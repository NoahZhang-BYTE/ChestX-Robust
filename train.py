from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from baseline.config import load_config
from baseline.data import build_dataloaders
from baseline.engine import fit
from baseline.models import build_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    _set_seed(int(config.get("seed", 42)))
    device = _resolve_device(config.get("device", "auto"))

    data_config = config["data"]
    train_loader, val_loader, label_cols = build_dataloaders(**data_config)
    config["data"]["label_cols"] = label_cols
    model_config = config["model"]
    model = build_model(model_config["name"], len(label_cols), bool(model_config.get("pretrained", False)))
    model.to(device)
    print(f"device={device} labels={label_cols} parameters={sum(p.numel() for p in model.parameters()):,}")
    fit(model, train_loader, val_loader, config, device)


if __name__ == "__main__":
    main()
