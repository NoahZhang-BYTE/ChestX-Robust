from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .losses import build_loss
from .metrics import multilabel_metrics
from .asl import AsymmetricLoss
from .reproducibility import capture_rng_state, restore_rng_state


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
    train_loader=None,
    val_loader=None,
    stale_epochs: int = 0,
    training_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
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
        "rng_state": rng_state if rng_state is not None else capture_rng_state(train_loader, val_loader),
        "stale_epochs": int(stale_epochs),
        "training_state": dict(training_state or {"stale_epochs": int(stale_epochs)}),
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
        model_state = checkpoint.get("model_state_dict")
        if model_state is None: model_state = checkpoint.get("model_state")
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is None: optimizer_state = checkpoint.get("optimizer_state")
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
    if "rng_state" not in checkpoint:
        import warnings
        warnings.warn("Legacy checkpoint has no RNG state; resumed run is not bitwise reproducible", RuntimeWarning)
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
    if name in {"asl", "asymmetric"}:
        return AsymmetricLoss(gamma_neg=float(loss_config.get("gamma_neg", 4.0)), gamma_pos=float(loss_config.get("gamma_pos", 1.0)), clip=float(loss_config.get("clip", 0.05))).to(device), {"name": name, "uses_pos_weight": False, "count_source": None, "cap": None, "per_label": []}
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
    scheduler=None,
    scheduler_state_dict: dict[str, Any] | None = None,
    resume_rng_state: dict[str, Any] | None = None,
    stale_epochs: int | None = None,
    resume_checkpoint: dict[str, Any] | None = None,
    max_epochs_this_call: int | None = None,
) -> list[dict[str, Any]]:
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        model_state = resume_checkpoint.get("model_state_dict") or resume_checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("Resume checkpoint is missing model state")
        model.load_state_dict(model_state, strict=True)
        history = list(resume_checkpoint.get("history", []))
        best_metric = resume_checkpoint.get("best_metric")
        optimizer_state_dict = resume_checkpoint.get("optimizer_state_dict")
        scaler_state_dict = resume_checkpoint.get("scaler_state_dict")
        scheduler_state_dict = resume_checkpoint.get("scheduler_state_dict")
        resume_rng_state = resume_checkpoint.get("rng_state")
        stale_epochs = resume_checkpoint.get("stale_epochs", (resume_checkpoint.get("training_state") or {}).get("stale_epochs", 0))
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
    if scheduler is None:
        sched_cfg = config.get("training", {}).get("scheduler")
        if sched_cfg is None and config.get("training", {}).get("scheduler_patience") is not None: sched_cfg = {"name":"plateau"}
        if sched_cfg and (isinstance(sched_cfg, str) and sched_cfg.lower() in {"reduce_on_plateau", "plateau"} or isinstance(sched_cfg, dict) and str(sched_cfg.get("name", "")).lower() in {"reduce_on_plateau", "plateau"}):
            sc = sched_cfg if isinstance(sched_cfg, dict) else {}
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=str(sc.get("mode", "max")), factor=float(sc.get("factor", config["training"].get("scheduler_factor",0.5))), patience=int(sc.get("patience", config["training"].get("scheduler_patience",2))), min_lr=float(sc.get("min_lr", config["training"].get("min_learning_rate",1e-6))))
    if scheduler is not None and scheduler_state_dict is not None: scheduler.load_state_dict(scheduler_state_dict)
    if resume_rng_state is not None: restore_rng_state(resume_rng_state, train_loader, val_loader)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf") if best_metric is None else float(best_metric)
    history = list(history or [])
    best_auroc = max((float(r.get("macro_auroc", float("-inf"))) for r in history), default=float("-inf"))
    best_auprc = max((float(r.get("macro_auprc", float("-inf"))) for r in history), default=float("-inf"))
    stale = int(config.get("training", {}).get("stale_epochs", 0) if stale_epochs is None else stale_epochs)
    training_cfg = config.get("training", {})
    scheduler_cfg = training_cfg.get("scheduler") if isinstance(training_cfg.get("scheduler"), dict) else {}
    early_cfg = training_cfg.get("early_stopping") if isinstance(training_cfg.get("early_stopping"), dict) else {}
    monitor_name = str(early_cfg.get("monitor", training_cfg.get("monitor", "macro_auroc")))
    checkpoint_monitor = str(training_cfg.get("checkpoint_monitor", "macro_auroc"))
    patience = int(early_cfg.get("patience", training_cfg.get("early_stopping_patience", 0)))
    min_delta = float(early_cfg.get("min_delta", training_cfg.get("early_stopping_min_delta", 0.0)))
    min_epoch = int(early_cfg.get("min_epoch", training_cfg.get("early_stopping_min_epoch", 1)))
    if max_epochs_this_call is None:
        end_epoch = int(config["training"]["epochs"])
    else:
        end_epoch = min(int(config["training"]["epochs"]), start_epoch + int(max_epochs_this_call) - 1)
    for epoch in range(start_epoch, end_epoch + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, float(config["training"]["threshold"]), amp_enabled)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        save_history(output_dir / "history.csv", history)
        if monitor_name not in metrics or checkpoint_monitor not in metrics:
            raise KeyError(f"Configured monitor metric missing: {monitor_name if monitor_name not in metrics else checkpoint_monitor}")
        score = float(metrics[checkpoint_monitor])
        monitor_score = float(metrics[monitor_name])
        previous_monitor = max((float(r.get(monitor_name, float("-inf"))) for r in history[:-1]), default=float("-inf"))
        is_best = score > best_score
        if is_best:
            best_score = score
        stale = 0 if monitor_score > previous_monitor + min_delta else stale + 1
        if scheduler is not None: scheduler.step(float(metrics.get(str(scheduler_cfg.get("monitor", monitor_name)), monitor_score)))
        epoch_rng_state = capture_rng_state(train_loader, val_loader)
        state = {"stale_epochs": int(stale), "best_metric": float(best_score), "monitor": monitor_name, "checkpoint_monitor": checkpoint_monitor}
        save_checkpoint(
            output_dir / "last.pt", model, optimizer, epoch, config, row,
            scaler=scaler, best_metric=best_score, history=history, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader, stale_epochs=stale, training_state=state, rng_state=epoch_rng_state,
        )
        if is_best:
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, epoch, config, row,
                scaler=scaler, best_metric=best_score, history=history, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader, stale_epochs=stale, training_state=state, rng_state=epoch_rng_state,
            )
        if float(metrics.get("macro_auroc", float("-inf"))) > best_auroc:
            best_auroc = float(metrics["macro_auroc"])
            save_checkpoint(output_dir / "best_macro_auroc.pt", model, optimizer, epoch, config, row, scaler=scaler, best_metric=best_auroc, history=history, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader, stale_epochs=stale, training_state=state, rng_state=epoch_rng_state)
        if float(metrics.get("macro_auprc", float("-inf"))) > best_auprc:
            best_auprc = float(metrics["macro_auprc"])
            save_checkpoint(output_dir / "best_macro_auprc.pt", model, optimizer, epoch, config, row, scaler=scaler, best_metric=best_auprc, history=history, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader, stale_epochs=stale, training_state=state, rng_state=epoch_rng_state)
        if patience and epoch >= min_epoch and stale >= patience:
            break
        print(row)
    return history
