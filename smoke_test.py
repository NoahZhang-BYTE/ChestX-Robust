"""Run one prepared NIH training batch through the baseline model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from baseline.config import load_config
from baseline.data import build_dataloaders
from baseline.models import build_model
from train import _resolve_device, _set_seed


@dataclass(frozen=True)
class SmokeTestResult:
    images_shape: tuple[int, ...]
    labels_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    loss: float


def run_smoke_test(config_path: str | Path) -> SmokeTestResult:
    """Load one training batch and verify it matches the model's raw logits."""
    config = load_config(config_path)
    _set_seed(int(config.get("seed", 42)))
    device = _resolve_device(config.get("device", "auto"))

    train_loader, _, label_cols = build_dataloaders(**config["data"])
    images, labels = next(iter(train_loader))
    model_config = config["model"]
    model = build_model(
        model_config["name"], len(label_cols), bool(model_config.get("pretrained", False))
    ).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(images.to(device))
        targets = labels.to(device)
        if logits.shape != targets.shape:
            raise ValueError(
                f"Logit shape {tuple(logits.shape)} does not match target shape {tuple(targets.shape)}"
            )
        loss = nn.BCEWithLogitsLoss()(logits, targets)

    if not torch.isfinite(loss):
        raise ValueError("BCEWithLogitsLoss produced a non-finite loss")

    print(f"images.shape={tuple(images.shape)}")
    print(f"labels.shape={tuple(labels.shape)}")
    print(f"images.dtype={images.dtype}")
    print(f"labels.dtype={labels.dtype}")
    print(f"labels[:3]={labels[:3]}")
    print(f"logits.shape={tuple(logits.shape)}")
    print(f"logits.dtype={logits.dtype}")
    print(f"loss={loss.item():.6f}")

    return SmokeTestResult(
        images_shape=tuple(images.shape),
        labels_shape=tuple(labels.shape),
        logits_shape=tuple(logits.shape),
        loss=loss.item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one multi-label NIH pipeline batch.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    run_smoke_test(args.config)


if __name__ == "__main__":
    main()
