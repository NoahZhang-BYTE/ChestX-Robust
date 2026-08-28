from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .metrics import multilabel_metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp_enabled=False) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss = criterion(model(images), targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5, amp_enabled=False) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    target_batches, probability_batches = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
        target_batches.append(targets.cpu().numpy())
        probability_batches.append(torch.sigmoid(logits).cpu().numpy())
    targets = np.concatenate(target_batches, axis=0)
    probabilities = np.concatenate(probability_batches, axis=0)
    return total_loss / max(total_items, 1), multilabel_metrics(targets, probabilities, threshold)


def fit(model, train_loader, val_loader, config: dict[str, Any], device) -> list[dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"])
    )
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, float(config["training"]["threshold"]), amp_enabled)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, config, row)
        score = metrics["macro_auroc"] if np.isfinite(metrics["macro_auroc"]) else metrics["macro_f1"]
        if score > best_score:
            best_score = score
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, config, row)
        print(row)
    return history
