"""Freeze validation thresholds and run the independent test for formal baselines.

This wrapper is intentionally separate from the historical ResNet18-only
threshold script. It reuses the same evaluation primitives and frozen-bundle
contract without changing their behavior.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision

from baseline.data import build_evaluation_loader, evaluation_preprocessing_spec
from baseline.evaluation import (
    collect_predictions,
    compute_auc_metrics,
    compute_f1_metrics,
    compute_per_class_metrics,
    find_per_class_thresholds,
)
from baseline.labels import LABEL_COLUMNS
from baseline.models import build_model
from baseline.nih import validate_labels_csv
from run_final_test import run_final_test
from tune_thresholds import (
    build_threshold_artifact,
    resolve_device,
    sha256_file,
    sha256_json,
    write_json,
)


def _loader_config(data_config: dict[str, Any]) -> dict[str, Any]:
    names = {
        "csv_path",
        "image_root",
        "image_col",
        "label_cols",
        "image_size",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
    }
    return {key: value for key, value in data_config.items() if key in names}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def freeze_validation(
    checkpoint_path: str | Path,
    frozen_dir: str | Path,
    device_value: str = "auto",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    frozen_dir = Path(frozen_dir).resolve()
    if frozen_dir.exists():
        raise FileExistsError(f"Frozen output directory already exists: {frozen_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state", "epoch", "config", "metrics"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint is missing required keys: {sorted(required)}")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a mapping")
    labels = list(config.get("data", {}).get("label_cols") or [])
    if labels != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint labels do not match canonical label order")

    data_config = dict(config["data"])
    labels_csv = Path(data_config["csv_path"]).resolve()
    image_root = Path(data_config["image_root"]).resolve()
    frame = validate_labels_csv(labels_csv, image_root)
    split_counts = {
        split: int(frame["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    loader, loader_labels = build_evaluation_loader(
        split="val", **_loader_config(data_config)
    )
    if loader_labels != labels:
        raise ValueError("Validation loader label order does not match checkpoint")

    model_config = config["model"]
    model = build_model(model_config["name"], len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = resolve_device(device_value)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    amp_enabled = bool(config.get("training", {}).get("amp", False)) and device.type == "cuda"
    targets, probabilities = collect_predictions(model, loader, device, amp_enabled)
    if targets.shape != (split_counts["val"], len(labels)):
        raise ValueError(f"Validation predictions have unexpected shape {targets.shape}")

    thresholds = find_per_class_thresholds(targets, probabilities)
    fallback_reasons = [
        None if np.unique(targets[:, i]).size == 2 else "single_class_validation_labels"
        for i in range(targets.shape[1])
    ]
    checkpoint_sha = sha256_file(checkpoint_path)
    labels_sha = sha256_file(labels_csv)
    threshold_artifact = build_threshold_artifact(
        labels,
        thresholds,
        checkpoint_sha256=checkpoint_sha,
        labels_csv_sha256=labels_sha,
        fallback_reasons=fallback_reasons,
    )
    auc = compute_auc_metrics(targets, probabilities, labels)
    fixed = compute_f1_metrics(targets, probabilities, 0.5)
    tuned = compute_f1_metrics(targets, probabilities, thresholds)
    validation_metrics = {
        "split": "val",
        "sample_count": int(targets.shape[0]),
        "num_classes": int(targets.shape[1]),
        "threshold_independent": auc,
        "f1_at_0_5": {
            **fixed,
            "per_class": compute_per_class_metrics(targets, probabilities, labels, 0.5),
        },
        "f1_tuned": {
            **tuned,
            "per_class": compute_per_class_metrics(targets, probabilities, labels, thresholds),
        },
        "threshold_source": "validation",
        "test_used_for_tuning": False,
    }
    frozen_dir.mkdir(parents=True, exist_ok=False)
    np.save(frozen_dir / "validation_targets.npy", targets.astype(np.uint8))
    np.save(frozen_dir / "validation_probabilities.npy", probabilities.astype(np.float32))
    write_json(frozen_dir / "thresholds.json", threshold_artifact)
    write_json(frozen_dir / "validation_metrics.json", validation_metrics)
    protocol = {
        "schema_version": 1,
        "status": "frozen",
        "checkpoint_selection_metric": "validation_macro_auroc",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "epoch": int(checkpoint["epoch"]),
            "validation_macro_auroc": float(checkpoint["metrics"]["macro_auroc"]),
        },
        "config": {"source": "checkpoint_embedded", "sha256": sha256_json(config)},
        "labels_csv": {"path": str(labels_csv), "sha256": labels_sha},
        "image_root": str(image_root),
        "label_cols": labels,
        "num_classes": len(labels),
        "split_counts": split_counts,
        "preprocessing": evaluation_preprocessing_spec(int(data_config.get("image_size", 224))),
        "threshold": {
            "source": "validation",
            "strategy": "per_class_max_f1",
            "artifact": "thresholds.json",
            "sha256": sha256_file(frozen_dir / "thresholds.json"),
            "values_by_label": threshold_artifact["thresholds_by_label"],
        },
        "test_used_for_tuning": False,
    }
    write_json(frozen_dir / "evaluation_protocol.json", protocol)
    metadata = {
        "schema_version": 1,
        "stage": "formal_validation_threshold_freeze",
        "device": str(device),
        "amp_enabled": amp_enabled,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_sha256": checkpoint_sha,
        "labels_csv_sha256": labels_sha,
    }
    write_json(frozen_dir / "validation_run_metadata.json", metadata)
    return {
        "frozen_dir": str(frozen_dir),
        "validation_metrics": validation_metrics,
        "thresholds": threshold_artifact["thresholds_by_label"],
        "protocol": protocol,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-test", action="store_true")
    args = parser.parse_args()
    result = freeze_validation(args.checkpoint, args.frozen_dir, args.device)
    if args.run_test:
        result["test"] = run_final_test(result["frozen_dir"], device_value=args.device)
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
