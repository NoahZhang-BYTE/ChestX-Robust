"""Benchmark training throughput and CUDA memory for the formal baselines.

Each model/batch-size pair is initialized independently and runs only a fixed
warm-up followed by a fixed number of measured optimizer steps. No validation,
test, threshold fitting, or checkpoint writing is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from baseline.config import load_config
from baseline.data import build_dataloaders
from baseline.models import build_model
from train import _resolve_device, _set_seed


DEFAULT_CONFIGS = {
    "resnet18": Path("configs/formal_b0_resnet18.yaml"),
    "densenet121": Path("configs/formal_b1_densenet121.yaml"),
}
DEFAULT_BATCH_SIZES = (32, 48, 64)
WARMUP_STEPS = 30
MEASURED_STEPS = 200


def _amp_state(config: dict[str, Any], device: torch.device) -> bool:
    return bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"


def _run_one(
    model_name: str,
    config: dict[str, Any],
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    _set_seed(seed)
    device = _resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_config = dict(config["data"])
    data_config["batch_size"] = batch_size
    # Make the persisted split and shuffle seed explicit while preserving all
    # formal DataLoader settings.
    data_config["seed"] = seed
    train_loader = None
    model = None
    criterion = None
    optimizer = None
    loader_iter = None
    amp_enabled = _amp_state(config, device)
    train_loader, _, label_cols = build_dataloaders(**data_config)
    model_config = config["model"]
    model = build_model(
        model_name,
        len(label_cols),
        bool(model_config.get("pretrained", False)),
    ).to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=amp_enabled)
        if device.type == "cuda"
        else None
    )
    loader_iter = iter(train_loader)

    def step() -> float:
        nonlocal loader_iter
        try:
            images, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            images, targets = next(loader_iter)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            loss = criterion(model(images), targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return float(loss.detach().item())

    try:
        model.train()
        for _ in range(warmup_steps):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        losses = [step() for _ in range(measured_steps)]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        avg_step_ms = elapsed * 1000.0 / measured_steps
        images_per_sec = batch_size * measured_steps / elapsed
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
        else:
            peak_allocated = math.nan
            peak_reserved = math.nan
        return {
            "model": model_name,
            "batch_size": batch_size,
            "status": "PASS",
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "total_measured_time_sec": elapsed,
            "avg_step_ms": avg_step_ms,
            "images_per_sec": images_per_sec,
            "peak_allocated_gb": peak_allocated,
            "peak_reserved_gb": peak_reserved,
            "mean_loss": float(np.mean(losses)),
            "error": None,
            "seed": seed,
            "amp": amp_enabled,
            "num_workers": int(data_config.get("num_workers", 0)),
            "prefetch_factor": int(data_config.get("prefetch_factor", 2)),
            "persistent_workers": bool(data_config.get("persistent_workers", True)),
            "pin_memory": bool(torch.cuda.is_available()),
            "image_size": int(data_config.get("image_size", 224)),
            "csv_path": str(data_config["csv_path"]),
            "split_policy": "persisted train split; no validation/test evaluation",
        }
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "model": model_name,
            "batch_size": batch_size,
            "status": "OOM" if is_oom else "ERROR",
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "total_measured_time_sec": None,
            "avg_step_ms": None,
            "images_per_sec": None,
            "peak_allocated_gb": (
                torch.cuda.max_memory_allocated(device) / (1024**3)
                if device.type == "cuda"
                else None
            ),
            "peak_reserved_gb": (
                torch.cuda.max_memory_reserved(device) / (1024**3)
                if device.type == "cuda"
                else None
            ),
            "mean_loss": None,
            "error": repr(exc),
            "seed": seed,
            "amp": amp_enabled,
            "num_workers": int(data_config.get("num_workers", 0)),
            "prefetch_factor": int(data_config.get("prefetch_factor", 2)),
            "persistent_workers": bool(data_config.get("persistent_workers", True)),
            "pin_memory": bool(torch.cuda.is_available()),
            "image_size": int(data_config.get("image_size", 224)),
            "csv_path": str(data_config["csv_path"]),
            "split_policy": "persisted train split; no validation/test evaluation",
        }
    finally:
        del loader_iter, train_loader, optimizer, criterion, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "PASS"]
    if not successful:
        return {"fastest_batch_size": None, "lowest_memory_batch_size": None, "recommended_batch_size": None}
    fastest = max(successful, key=lambda row: row["images_per_sec"])
    lowest_memory = min(successful, key=lambda row: row["peak_allocated_gb"])
    ordered = sorted(successful, key=lambda row: row["batch_size"])
    recommended = fastest
    # Once the throughput gain is below 5%, prefer the preceding batch when
    # its peak allocation is meaningfully lower (at least 10% lower).
    for previous, current in zip(ordered, ordered[1:]):
        gain = current["images_per_sec"] / previous["images_per_sec"] - 1.0
        memory_increase = current["peak_allocated_gb"] / previous["peak_allocated_gb"] - 1.0
        if gain < 0.05 and memory_increase >= 0.10:
            recommended = previous
            break
    return {
        "fastest_batch_size": int(fastest["batch_size"]),
        "lowest_memory_batch_size": int(lowest_memory["batch_size"]),
        "recommended_batch_size": int(recommended["batch_size"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark formal baseline batch sizes.")
    parser.add_argument("--output-dir", default="outputs/batch_benchmark")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measured-steps", type=int, default=MEASURED_STEPS)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.measured_steps < 1:
        raise ValueError("warmup-steps must be non-negative and measured-steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_name, config_path in DEFAULT_CONFIGS.items():
        config = load_config(config_path)
        for batch_size in DEFAULT_BATCH_SIZES:
            print(f"starting model={model_name} batch_size={batch_size}", flush=True)
            row = _run_one(model_name, config, batch_size, args.warmup_steps, args.measured_steps)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)

    recommendations = {
        model_name: _recommend([row for row in rows if row["model"] == model_name])
        for model_name in DEFAULT_CONFIGS
    }
    payload = {
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "results": rows,
        "recommendations": recommendations,
        "fairness": {
            "configs": {name: str(path) for name, path in DEFAULT_CONFIGS.items()},
            "differences": ["model.name", "batch_size"],
            "evaluation": False,
            "threshold_fitting": False,
            "checkpoint_saved": False,
        },
    }
    (output_dir / "batch_benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    csv_fields = [
        "model", "batch_size", "status", "warmup_steps", "measured_steps",
        "total_measured_time_sec", "avg_step_ms", "images_per_sec",
        "peak_allocated_gb", "peak_reserved_gb", "mean_loss", "error",
    ]
    with (output_dir / "batch_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in csv_fields} for row in rows)
    print(json.dumps({"recommendations": recommendations}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
