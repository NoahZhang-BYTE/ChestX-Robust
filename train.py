from __future__ import annotations

import argparse
import torch

from baseline.config import load_config
from baseline.data import build_dataloaders
from baseline.engine import fit
from baseline.models import build_model
from baseline.reproducibility import seed_everything


def _set_seed(seed: int) -> None:
    seed_everything(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-label chest X-ray baseline")
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)), bool(config.get("deterministic", config.get("training", {}).get("deterministic", False))))
    device = _resolve_device(config.get("device", "auto"))
    deterministic = bool(config.get("deterministic", config.get("training", {}).get("deterministic", False)))
    if device.type == "cuda" and not deterministic:
        torch.backends.cudnn.benchmark = True

    data_config = config["data"]
    data_config = dict(data_config)
    data_config.setdefault("deterministic", deterministic)
    train_loader, val_loader, label_cols = build_dataloaders(**data_config)
    config["data"]["label_cols"] = label_cols
    model_config = config["model"]
    model = build_model(model_config["name"], len(label_cols), bool(model_config.get("pretrained", False)))
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    print(f"device={device} labels={label_cols} parameters={sum(p.numel() for p in model.parameters()):,}")
    fit(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
