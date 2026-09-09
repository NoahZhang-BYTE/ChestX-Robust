"""Train one formal baseline from a frozen YAML config and write history.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from baseline.config import load_config
from baseline.data import build_dataloaders
from baseline.engine import build_training_criterion, fit, load_training_checkpoint
from baseline.models import build_model
from baseline.resources import check_commit_headroom, get_system_commit
from train import _resolve_device, _set_seed


def _enforce_commit_guard() -> None:
    result = check_commit_headroom(get_system_commit(), minimum_headroom_percent=20.0)
    stats = result.stats
    status = (
        "[PASS]\nSystem commit headroom is sufficient."
        if result.passed
        else "[BLOCKED]\nSystem commit headroom is too low."
    )
    print(
        f"{status}\n"
        f"Committed: {stats.committed_bytes / 1e9:.3f} GB\n"
        f"Limit: {stats.limit_bytes / 1e9:.3f} GB\n"
        f"Headroom: {stats.headroom_bytes / 1e9:.3f} GB\n"
        f"Headroom percent: {stats.headroom_percent:.3f}%",
        flush=True,
    )
    if not result.passed:
        raise RuntimeError(
            "Close memory-heavy processes or reboot before training."
        )


def run_formal_training(
    config_path: str | Path, resume_path: str | Path | None = None
) -> list[dict]:
    _enforce_commit_guard()
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    if resume_path is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Formal output directory already exists: {output_dir}")
    resume_checkpoint = load_training_checkpoint(resume_path) if resume_path is not None else None
    _set_seed(int(config.get("seed", 42)))
    device = _resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, label_cols = build_dataloaders(**config["data"])
    config["data"]["label_cols"] = label_cols
    criterion, loss_metadata = build_training_criterion(train_loader, config, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if loss_metadata["uses_pos_weight"]:
        pos_weight_path = output_dir / "pos_weights.csv"
        pd.DataFrame(loss_metadata["per_label"]).to_csv(pos_weight_path, index=False)
        print(
            f"loss={loss_metadata['name']} count_source=train cap={loss_metadata['cap']} "
            f"pos_weight_device={criterion.pos_weight.device} pos_weight_path={pos_weight_path}",
            flush=True,
        )
        print(pd.DataFrame(loss_metadata["per_label"]).to_string(index=False), flush=True)
    model_config = config["model"]
    model = build_model(
        model_config["name"], len(label_cols), bool(model_config.get("pretrained", False))
    ).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    print(
        f"formal={model_config['name']} device={device} labels={len(label_cols)} "
        f"batch_size={config['data']['batch_size']}",
        flush=True,
    )
    fit_kwargs = {}
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        completed_epoch = resume_checkpoint["epoch"]
        start_epoch = completed_epoch + 1
        best_metric = resume_checkpoint["best_metric"]
        print(
            f"Resume checkpoint:\n"
            f"path: {Path(resume_path)}\n"
            f"completed_epoch: {completed_epoch}\n"
            f"resume_from_epoch: {start_epoch}\n"
            f"best_metric: {best_metric}",
            flush=True,
        )
        fit_kwargs = {
            "start_epoch": start_epoch,
            "history": resume_checkpoint["history"],
            "best_metric": best_metric,
            "optimizer_state_dict": resume_checkpoint["optimizer_state_dict"],
            "scaler_state_dict": resume_checkpoint["scaler_state_dict"],
            "resume_rng_state": resume_checkpoint.get("rng_state"),
            "stale_epochs": resume_checkpoint.get("stale_epochs", 0),
        }
    history = fit(model, train_loader, val_loader, config, device, criterion=criterion, **fit_kwargs)
    print(f"history_rows={len(history)} history_path={output_dir / 'history.csv'}", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a formal baseline training job.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    run_formal_training(args.config, args.resume)


if __name__ == "__main__":
    main()
