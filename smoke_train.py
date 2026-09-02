"""Run a bounded one-epoch loss/optimizer smoke test without saving a model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from baseline.config import load_config
from baseline.data import build_dataloaders
from baseline.losses import build_loss
from baseline.models import build_model
from train import _resolve_device, _set_seed


@dataclass(frozen=True)
class TrainingSmokeResult:
    loss_name: str
    batches: int
    mean_loss: float
    pos_weight: list[float] | None


def run_training_smoke_test(
    config_path: str | Path, loss_name: str, max_batches: int | None = 2
) -> TrainingSmokeResult:
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    config = load_config(config_path)
    _set_seed(int(config.get("seed", 42)))
    device = _resolve_device(config.get("device", "auto"))
    train_loader, _, label_cols = build_dataloaders(**config["data"])
    model_config = config["model"]
    model = build_model(model_config["name"], len(label_cols), pretrained=False).to(device)
    frame = train_loader.dataset.frame
    positive_counts = torch.tensor(frame[label_cols].sum(axis=0).to_numpy(), dtype=torch.float32)
    negative_counts = torch.tensor(
        len(frame) - frame[label_cols].sum(axis=0).to_numpy(), dtype=torch.float32
    )
    criterion = (
        build_loss(loss_name, positive_counts, negative_counts)
        if loss_name != "bce"
        else build_loss("bce")
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("training", {}).get("learning_rate", 3e-4))
    )
    model.train()
    losses: list[float] = []
    for batch_index, (images, targets) in enumerate(train_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        if not torch.isfinite(loss):
            raise ValueError(f"{loss_name} produced a non-finite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    if not losses:
        raise ValueError("Training loader yielded no batches")
    pos_weight = criterion.pos_weight.detach().cpu().tolist() if criterion.pos_weight is not None else None
    result = TrainingSmokeResult(loss_name, len(losses), sum(losses) / len(losses), pos_weight)
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded training smoke epoch.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--loss", choices=("bce", "bce_pos_weight"), required=True)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional batch cap; omit for a complete train epoch.")
    args = parser.parse_args()
    run_training_smoke_test(args.config, args.loss, args.max_batches)


if __name__ == "__main__":
    main()
