"""Guarded B4 scheduler experiment; never launched implicitly by the workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from baseline.asl import AsymmetricLoss
from baseline.data import build_dataloaders
from baseline.engine import build_training_criterion, evaluate, load_training_checkpoint, restore_rng_state, save_checkpoint, save_history, train_one_epoch
from baseline.models import build_model
from formal_train import _enforce_commit_guard
from night_workflow import should_early_stop
from train import _resolve_device, _set_seed
from workflow_state import atomic_write_json, update_stage


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("B4 config root must be a mapping")
    return value


def _build_criterion(config: dict[str, Any], train_loader, device):
    loss = config.get("training", {}).get("loss", {"name": "bce"})
    name = str(loss.get("name", "bce")).lower() if isinstance(loss, dict) else str(loss).lower()
    if name in {"asymmetric", "asl"}:
        return AsymmetricLoss(
            gamma_neg=float(loss.get("gamma_neg", 4.0)),
            gamma_pos=float(loss.get("gamma_pos", 1.0)),
            clip=float(loss.get("clip", 0.05)),
        ).to(device)
    return build_training_criterion(train_loader, config, device)[0]


def _stale_auprc_epochs(history: list[dict[str, Any]], min_delta: float) -> int:
    best, stale = float("-inf"), 0
    for row in history:
        value = float(row["macro_auprc"])
        if value > best + min_delta:
            best, stale = value, 0
        else:
            stale += 1
    return stale


def run_b4_training(
    config_path: str | Path,
    resume_path: str | Path | None = None,
    workflow_state_path: str | Path = "workflow_state.json",
) -> list[dict[str, Any]]:
    _enforce_commit_guard()
    config = _read_yaml(Path(config_path))
    output_dir = Path(config["output_dir"])
    resume = load_training_checkpoint(resume_path) if resume_path is not None else None
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"B4 output directory already exists: {output_dir}")
    _set_seed(int(config.get("seed", 42)))
    device = _resolve_device(config.get("device", "auto"))
    train_loader, val_loader, labels = build_dataloaders(**config["data"])
    config["data"]["label_cols"] = labels
    model = build_model(config["model"]["name"], len(labels), bool(config["model"].get("pretrained", False))).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    criterion = _build_criterion(config, train_loader, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(config["training"].get("scheduler_factor", 0.5)),
        patience=int(config["training"].get("scheduler_patience", 2)),
        min_lr=float(config["training"].get("min_learning_rate", 1e-6)),
    )
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    history: list[dict[str, Any]] = []
    start_epoch, best_auroc, best_auprc, stale = 1, float("-inf"), float("-inf"), 0
    min_delta = float(config["training"].get("early_stopping_min_delta", 1e-4))
    if resume is not None:
        if resume.get("rng_state") is not None:
            restore_rng_state(resume["rng_state"])
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        if scaler is not None and resume.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume["scaler_state_dict"])
        if resume.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume["scheduler_state_dict"])
        history = list(resume.get("history", []))
        start_epoch = int(resume["epoch"]) + 1
        best_auroc = max((float(row.get("macro_auroc", float("-inf"))) for row in history), default=float("-inf"))
        best_auprc = max((float(row.get("macro_auprc", float("-inf"))) for row in history), default=float("-inf"))
        stale = _stale_auprc_epochs(history, min_delta)
        print(f"Resume checkpoint: {resume_path}; completed_epoch={resume['epoch']}; resume_from_epoch={start_epoch}; best_metric={best_auroc}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_dir / "config_snapshot.yaml"
    source_config = Path(config_path).read_text(encoding="utf-8")
    if config_snapshot.exists() and config_snapshot.read_text(encoding="utf-8") != source_config:
        raise RuntimeError(f"B4 config snapshot differs: {config_snapshot}")
    if not config_snapshot.exists():
        config_snapshot.write_text(source_config, encoding="utf-8", newline="\n")
    atomic_write_json(output_dir / "run_metadata.json", {
        "status": "running",
        "start_epoch": start_epoch,
        "max_epochs": int(config["training"].get("epochs", 20)),
        "monitor": "macro_auprc",
        "test_used": False,
    })
    stop_reason = "completed"
    for epoch in range(start_epoch, int(config["training"].get("epochs", 20)) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, 0.5, amp_enabled)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        monitor = float(metrics["macro_auprc"])
        scheduler.step(monitor)
        save_history(output_dir / "history.csv", history)
        auroc, auprc = float(metrics["macro_auroc"]), float(metrics["macro_auprc"])
        if auroc > best_auroc:
            best_auroc = auroc
            save_checkpoint(output_dir / "best_macro_auroc.pt", model, optimizer, epoch, config, row, scaler=scaler, best_metric=best_auroc, history=history, scheduler=scheduler)
        if auprc > best_auprc:
            best_auprc = auprc
            save_checkpoint(output_dir / "best_macro_auprc.pt", model, optimizer, epoch, config, row, scaler=scaler, best_metric=best_auprc, history=history, scheduler=scheduler)
        if auprc > max((float(previous["macro_auprc"]) for previous in history[:-1]), default=float("-inf")) + min_delta:
            stale = 0
        else:
            stale += 1
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, config, row, scaler=scaler, best_metric=best_auroc, history=history, scheduler=scheduler)
        print(row, flush=True)
        if should_early_stop(
            epoch,
            stale,
            int(config["training"].get("early_stopping_patience", 4)),
            int(config["training"].get("early_stopping_min_epoch", 5)),
        ):
            stop_reason = "early_stopping"
            print(f"early_stopping epoch={epoch} patience={stale}", flush=True)
            break
    atomic_write_json(output_dir / "run_metadata.json", {
        "status": "completed",
        "stop_reason": stop_reason,
        "completed_epoch": int(history[-1]["epoch"]),
        "best_macro_auroc": best_auroc,
        "best_macro_auprc": best_auprc,
        "monitor": "macro_auprc",
        "test_used": False,
    })
    # Ledger mutation is opt-in; historical wrappers must not infer a stage
    # from stale workflow state (B5 also reuses this implementation).
    stage_name = config.get("stage") or config.get("workflow_stage")
    if stage_name == "B4_training":
        update_stage(workflow_state_path, stage_name, "completed")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guarded B4 scheduler training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--workflow-state", default="workflow_state.json")
    args = parser.parse_args()
    run_b4_training(args.config, args.resume, args.workflow_state)


if __name__ == "__main__":
    main()
