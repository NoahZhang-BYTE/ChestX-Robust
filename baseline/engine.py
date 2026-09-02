from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .losses import build_loss
from .metrics import multilabel_metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    scaler: torch.amp.GradScaler | None = None,
    best_metric: float | None = None,
    history: list[dict[str, Any]] | None = None,
    scheduler: Any | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "history": list(history or []),
        "config": config,
        "metrics": metrics,
        # Preserve the keys used by existing evaluation artifacts and scripts.
        "model_state": model_state,
        "optimizer_state": optimizer_state,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    _atomic_torch_save(payload, path)


def save_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            pd.DataFrame(history).to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"History temporary file was not written: {temporary_path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"Could not load resume checkpoint: {path}") from error
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Resume checkpoint is not a mapping: {path}")
    try:
        epoch = int(checkpoint["epoch"])
        model_state = checkpoint.get("model_state_dict", checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state_dict", checkpoint["optimizer_state"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Resume checkpoint is missing required training state: {path}") from error
    if epoch < 0 or not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise RuntimeError(f"Resume checkpoint contains invalid training state: {path}")
    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        raise RuntimeError(f"Resume checkpoint history is invalid: {path}")
    checkpoint["epoch"] = epoch
    checkpoint["model_state_dict"] = model_state
    checkpoint["optimizer_state_dict"] = optimizer_state
    checkpoint["history"] = history
    checkpoint["best_metric"] = checkpoint.get("best_metric")
    checkpoint["scaler_state_dict"] = checkpoint.get("scaler_state_dict")
    return checkpoint


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"Checkpoint temporary file was not written: {temporary_path}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


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


def build_training_criterion(train_loader, config: dict[str, Any], device) -> tuple[nn.BCEWithLogitsLoss, dict[str, Any]]:
    """Build the configured loss using label counts from the train loader only."""
    loss_config = config.get("training", {}).get("loss", {"name": "bce"})
    if isinstance(loss_config, str):
        loss_config = {"name": loss_config}
    if not isinstance(loss_config, dict):
        raise ValueError("training.loss must be a mapping or loss name")
    name = str(loss_config.get("name", "bce")).lower()
    cap = float(loss_config.get("cap", 20.0))
    if name == "bce":
        return build_loss("bce").to(device), {
            "name": name,
            "uses_pos_weight": False,
            "count_source": None,
            "cap": None,
            "per_label": [],
        }

    dataset = getattr(train_loader, "dataset", None)
    frame = getattr(dataset, "frame", None)
    label_cols = list(getattr(dataset, "label_cols", []))
    if frame is None or not label_cols:
        raise ValueError("Configured pos_weight requires a labeled train dataset")
    if "split" in frame.columns and not bool(frame["split"].eq("train").all()):
        raise ValueError("Configured pos_weight must be computed from train split rows only")
    positive_counts = torch.tensor(
        frame.loc[:, label_cols].sum(axis=0).to_numpy(dtype=np.float32), dtype=torch.float32
    )
    negative_counts = torch.tensor(
        len(frame) - positive_counts.numpy(), dtype=torch.float32
    )
    criterion = build_loss(name, positive_counts, negative_counts, cap=cap).to(device)
    weights = criterion.pos_weight.detach().cpu().numpy()
    raw_ratios = (negative_counts / positive_counts).numpy()
    sqrt_ratios = np.sqrt(raw_ratios)
    return criterion, {
        "name": name,
        "uses_pos_weight": True,
        "count_source": "train",
        "cap": cap,
        "per_label": [
            {
                "class": label,
                "positive_count": int(positive_counts[index].item()),
                "negative_count": int(negative_counts[index].item()),
                "negative_to_positive_ratio": float(raw_ratios[index]),
                "sqrt_ratio": float(sqrt_ratios[index]),
                "pos_weight": float(weights[index]),
            }
            for index, label in enumerate(label_cols)
        ],
    }


def fit(
    model,
    train_loader,
    val_loader,
    config: dict[str, Any],
    device,
    criterion=None,
    start_epoch: int = 1,
    history: list[dict[str, Any]] | None = None,
    best_metric: float | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    scaler_state_dict: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if criterion is None:
        criterion, _ = build_training_criterion(train_loader, config, device)
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"])
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scaler is not None and scaler_state_dict is not None:
        scaler.load_state_dict(scaler_state_dict)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf") if best_metric is None else float(best_metric)
    history = list(history or [])
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, float(config["training"]["threshold"]), amp_enabled)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        save_history(output_dir / "history.csv", history)
        score = metrics["macro_auroc"] if np.isfinite(metrics["macro_auroc"]) else metrics["macro_f1"]
        is_best = score > best_score
        if is_best:
            best_score = score
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, epoch, config, row,
            scaler=scaler, best_metric=best_score, history=history,
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, epoch, config, row,
                scaler=scaler, best_metric=best_score, history=history,
            )
        print(row)
    return history
