"""Train B3 (ASL only) with the same guarded, resumable workflow as B2."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from baseline.asl import AsymmetricLoss
from baseline.data import build_dataloaders
from baseline.engine import evaluate, load_training_checkpoint, restore_rng_state, save_checkpoint, save_history, train_one_epoch
from baseline.models import build_model
from formal_train import _enforce_commit_guard
from train import _resolve_device, _set_seed
from workflow_state import update_stage


def _criterion(config: dict[str, Any]) -> AsymmetricLoss:
    loss_config = config.get("training", {}).get("loss", {})
    if str(loss_config.get("name", "")).lower() not in {"asymmetric", "asl"}:
        raise ValueError("B3 config must use asymmetric loss")
    return AsymmetricLoss(
        gamma_neg=float(loss_config.get("gamma_neg", 4.0)),
        gamma_pos=float(loss_config.get("gamma_pos", 1.0)),
        clip=float(loss_config.get("clip", 0.05)),
    )


def run_b3_training(
    config_path: str | Path = "configs/formal_b3_densenet121_asl.yaml",
    resume_path: str | Path | None = None,
    workflow_state_path: str | Path = "workflow_state.json",
) -> list[dict[str, Any]]:
    _enforce_commit_guard()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("B3 config root must be a mapping")
    output_dir = Path(config["output_dir"])
    resume = load_training_checkpoint(resume_path) if resume_path is not None else None
    if resume is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"B3 output directory already exists: {output_dir}")
    _set_seed(int(config.get("seed", 42)))
    device = _resolve_device(config.get("device", "auto"))
    train_loader, val_loader, labels = build_dataloaders(**config["data"])
    config["data"]["label_cols"] = labels
    model = build_model(config["model"]["name"], len(labels), bool(config["model"].get("pretrained", False))).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    criterion = _criterion(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    history: list[dict[str, Any]] = []
    start_epoch, best_auroc, best_auprc = 1, float("-inf"), float("-inf")
    if resume is not None:
        if resume.get("rng_state") is not None:
            restore_rng_state(resume["rng_state"])
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        if scaler is not None and resume.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume["scaler_state_dict"])
        history = list(resume.get("history", []))
        start_epoch = int(resume["epoch"]) + 1
        best_auroc = float(resume.get("best_metric") or float("-inf"))
        print(f"Resume checkpoint: {resume_path}; completed_epoch={resume['epoch']}; resume_from_epoch={start_epoch}; best_metric={best_auroc}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, float(config["training"].get("threshold", 0.5)), amp_enabled)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        history.append(row)
        save_history(output_dir / "history.csv", history)
        auroc = float(metrics["macro_auroc"])
        auprc = float(metrics["macro_auprc"])
        if auroc > best_auroc:
            best_auroc = auroc
            save_checkpoint(output_dir / "best_macro_auroc.pt", model, optimizer, epoch, config, row, scaler=scaler, best_metric=best_auroc, history=history)
        if auprc > best_auprc:
            best_auprc = auprc
            save_checkpoint(output_dir / "best_macro_auprc.pt", model, optimizer, epoch, config, row, scaler=scaler, best_metric=best_auprc, history=history)
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, config, row, scaler=scaler, best_metric=best_auroc, history=history)
        print(row, flush=True)
    stage_name = config.get("stage") or config.get("workflow_stage")
    if stage_name == "B3_training":
        update_stage(workflow_state_path, stage_name, "completed")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guarded B3 ASL training.")
    parser.add_argument("--config", default="configs/formal_b3_densenet121_asl.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--workflow-state", default="workflow_state.json")
    args = parser.parse_args()
    run_b3_training(args.config, args.resume, args.workflow_state)


if __name__ == "__main__":
    main()
