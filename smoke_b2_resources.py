"""Run a bounded B2-equivalent training and validation resource smoke test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from baseline.config import load_config
from baseline.data import build_dataloaders
from baseline.engine import build_training_criterion
from baseline.models import build_model
from baseline.resources import get_system_commit
from train import _resolve_device, _set_seed


def collect_resource_snapshot(device: torch.device) -> dict[str, Any]:
    process: dict[str, int | None] = {"rss_bytes": None, "private_bytes": None}
    system_ram: dict[str, float | int | None] = {
        "used_bytes": None,
        "available_bytes": None,
        "percent": None,
    }
    try:
        import psutil

        current_process = psutil.Process(os.getpid())
        process_info = current_process.memory_info()
        process_full_info = current_process.memory_full_info()
        process = {
            "rss_bytes": int(process_info.rss),
            "private_bytes": getattr(process_full_info, "private", None),
        }
        virtual_memory = psutil.virtual_memory()
        system_ram = {
            "used_bytes": int(virtual_memory.used),
            "available_bytes": int(virtual_memory.available),
            "percent": float(virtual_memory.percent),
        }
    except ImportError:
        pass

    try:
        commit = get_system_commit()
        system_commit: dict[str, float | int | None] = {
            "current_bytes": commit.committed_bytes,
            "limit_bytes": commit.limit_bytes,
            "headroom_bytes": commit.headroom_bytes,
            "headroom_percent": commit.headroom_percent,
        }
    except OSError:
        system_commit = {
            "current_bytes": None,
            "limit_bytes": None,
            "headroom_bytes": None,
            "headroom_percent": None,
        }

    if device.type == "cuda":
        gpu: dict[str, int | None] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
    else:
        gpu = {"allocated_bytes": None, "reserved_bytes": None, "max_allocated_bytes": None}
    return {
        "process": process,
        "system_ram": system_ram,
        "system_commit": system_commit,
        "gpu": gpu,
    }


def _print_resource_record(
    phase: str, iteration: int, loss: float, device: torch.device
) -> None:
    record = {
        "epoch": 1,
        "phase": phase,
        "iteration": iteration,
        "loss": loss,
        **collect_resource_snapshot(device),
    }
    print(json.dumps(record, ensure_ascii=True), flush=True)


def run_b2_resource_smoke_test(
    config_path: str | Path,
    max_train_batches: int = 200,
    max_val_batches: int = 50,
    log_every: int = 25,
) -> dict[str, int]:
    if max_train_batches < 1 or max_val_batches < 1 or log_every < 1:
        raise ValueError("batch limits and log_every must be positive")
    config = load_config(config_path)
    _set_seed(int(config.get("seed", 42)))
    device = _resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, label_cols = build_dataloaders(**config["data"])
    config["data"]["label_cols"] = label_cols
    criterion, _ = build_training_criterion(train_loader, config, device)
    model_config = config["model"]
    model = build_model(
        model_config["name"], len(label_cols), bool(model_config.get("pretrained", False))
    ).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
        torch.cuda.reset_peak_memory_stats(device)
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    model.train()
    train_batches = 0
    for iteration, (images, targets) in enumerate(train_loader, start=1):
        if iteration > max_train_batches:
            break
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
        train_batches = iteration
        if iteration % log_every == 0 or iteration == max_train_batches:
            _print_resource_record("train", iteration, float(loss.item()), device)

    model.eval()
    val_batches = 0
    with torch.no_grad():
        for iteration, (images, targets) in enumerate(val_loader, start=1):
            if iteration > max_val_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = criterion(model(images), targets)
            val_batches = iteration
            if iteration % log_every == 0 or iteration == max_val_batches:
                _print_resource_record("val", iteration, float(loss.item()), device)
    return {"train_batches": train_batches, "val_batches": val_batches}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded B2 resource smoke test.")
    parser.add_argument("--config", default="configs/formal_b2_densenet121_sqrt_posweight.yaml")
    parser.add_argument("--max-train-batches", type=int, default=200)
    parser.add_argument("--max-val-batches", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    result = run_b2_resource_smoke_test(
        args.config, args.max_train_batches, args.max_val_batches, args.log_every
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
