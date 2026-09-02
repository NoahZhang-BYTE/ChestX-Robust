"""Freeze validation-derived thresholds for the ResNet18 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
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


DEFAULT_CHECKPOINT = Path("checkpoints/imagenet_pretrained/best.pt")
DEFAULT_OUTPUT_DIR = Path("artifacts/nih_resnet18_imagenet_pretrained_best")


def build_threshold_artifact(
    label_cols: Sequence[str],
    thresholds: Sequence[float] | np.ndarray,
    *,
    checkpoint_sha256: str,
    labels_csv_sha256: str,
    fallback_reasons: Sequence[str | None],
) -> dict[str, Any]:
    """Build the ordered, validation-only threshold artifact."""
    labels = list(label_cols)
    values = np.asarray(thresholds, dtype=np.float64)
    reasons = list(fallback_reasons)
    if values.ndim != 1 or len(values) != len(labels):
        raise ValueError("thresholds must contain one value per label")
    if len(reasons) != len(labels):
        raise ValueError("fallback_reasons must contain one value per label")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("thresholds must be finite values between 0 and 1")
    threshold_list = [float(value) for value in values]
    return {
        "schema_version": 1,
        "source": "validation",
        "fit_split": "val",
        "strategy": "per_class_max_f1",
        "objective": "binary_f1",
        "tie_break": "highest_threshold",
        "degenerate_fallback": 0.5,
        "label_cols": labels,
        "thresholds": threshold_list,
        "thresholds_by_label": dict(zip(labels, threshold_list, strict=True)),
        "fallbacks": dict(zip(labels, reasons, strict=True)),
        "checkpoint_sha256": checkpoint_sha256,
        "labels_csv_sha256": labels_csv_sha256,
    }


def run_validation_tuning(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device_value: str = "auto",
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Evaluate val once, fit thresholds there, and atomically freeze artifacts."""
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Frozen output directory already exists: {output_dir}")
    checkpoint, config = _load_checkpoint(checkpoint_path)
    validate_designated_checkpoint(checkpoint_path, config, checkpoint["metrics"])
    labels = _checkpoint_labels(config)
    data_config = dict(config["data"])
    labels_csv = Path(data_config["csv_path"]).resolve()
    image_root = Path(data_config["image_root"]).resolve()
    frame = validate_labels_csv(labels_csv, image_root)
    split_counts = {
        split: int(frame["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    checkpoint_sha = sha256_file(checkpoint_path)
    labels_sha = sha256_file(labels_csv)
    config_sha = sha256_json(config)
    device = resolve_device(device_value)
    loader_config = _loader_config(
        data_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    val_loader, loader_labels = build_evaluation_loader(split="val", **loader_config)
    if loader_labels != labels:
        raise ValueError("Validation loader label order does not match checkpoint config")

    model_config = config["model"]
    model = build_model(model_config["name"], len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    amp_enabled = bool(config.get("training", {}).get("amp", False)) and device.type == "cuda"
    targets, probabilities = collect_predictions(
        model, val_loader, device, amp_enabled=amp_enabled
    )
    if targets.shape != (split_counts["val"], len(labels)):
        raise ValueError(
            f"Validation predictions have unexpected shape {targets.shape}"
        )

    fallback_reasons = [
        None if np.unique(targets[:, index]).size == 2 else "single_class_validation_labels"
        for index in range(targets.shape[1])
    ]
    thresholds = find_per_class_thresholds(targets, probabilities)
    threshold_artifact = build_threshold_artifact(
        labels,
        thresholds,
        checkpoint_sha256=checkpoint_sha,
        labels_csv_sha256=labels_sha,
        fallback_reasons=fallback_reasons,
    )
    auc_metrics = compute_auc_metrics(targets, probabilities, labels)
    reference_f1 = compute_f1_metrics(targets, probabilities, 0.5)
    tuned_f1 = compute_f1_metrics(targets, probabilities, thresholds)
    validation_metrics = {
        "split": "val",
        "sample_count": int(targets.shape[0]),
        "num_classes": int(targets.shape[1]),
        "threshold_independent": auc_metrics,
        "f1_at_0_5": {
            **reference_f1,
            "per_class": compute_per_class_metrics(
                targets, probabilities, labels, 0.5
            ),
        },
        "f1_tuned": {
            **tuned_f1,
            "per_class": compute_per_class_metrics(
                targets, probabilities, labels, thresholds
            ),
        },
        "undefined_metric_policy": (
            "Degenerate classes are excluded from macro AUROC/AUPRC; "
            "aggregate F1 uses sklearn zero_division=0."
        ),
    }

    stage = _make_stage(output_dir)
    try:
        np.save(stage / "validation_targets.npy", targets.astype(np.uint8))
        np.save(
            stage / "validation_probabilities.npy",
            probabilities.astype(np.float32),
        )
        write_json(stage / "thresholds.json", threshold_artifact)
        write_json(stage / "validation_metrics.json", validation_metrics)
        threshold_sha = sha256_file(stage / "thresholds.json")
        checkpoint_metrics = checkpoint.get("metrics") or {}
        image_size = int(data_config.get("image_size", 224))
        protocol = {
            "schema_version": 1,
            "status": "frozen",
            "checkpoint_selection_metric": "validation_macro_auroc",
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "epoch": int(checkpoint["epoch"]),
                "validation_macro_auroc": float(
                    checkpoint_metrics["macro_auroc"]
                ),
            },
            "config": {
                "source": "checkpoint_embedded",
                "sha256": config_sha,
            },
            "labels_csv": {"path": str(labels_csv), "sha256": labels_sha},
            "image_root": str(image_root),
            "label_cols": labels,
            "num_classes": len(labels),
            "split_counts": split_counts,
            "preprocessing": evaluation_preprocessing_spec(image_size),
            "threshold": {
                "source": "validation",
                "strategy": "per_class_max_f1",
                "artifact": "thresholds.json",
                "sha256": threshold_sha,
                "values_by_label": threshold_artifact["thresholds_by_label"],
            },
            "test_used_for_tuning": False,
        }
        write_json(stage / "evaluation_protocol.json", protocol)
        created_at = utc_now()
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in stage.iterdir()
            if path.is_file()
        }
        metadata = {
            "schema_version": 1,
            "stage": "validation_threshold_freeze",
            "evaluation_timestamp_utc": created_at,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "checkpoint_sha256": checkpoint_sha,
            "labels_csv_sha256": labels_sha,
            "evaluation_protocol_sha256": artifact_hashes[
                "evaluation_protocol.json"
            ],
            "artifact_sha256": artifact_hashes,
        }
        write_json(stage / "validation_run_metadata.json", metadata)
        stage.replace(output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "thresholds": threshold_artifact["thresholds_by_label"],
        "validation_metrics": validation_metrics,
        "checkpoint_sha256": checkpoint_sha,
        "labels_csv_sha256": labels_sha,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze validation-only thresholds for baseline v1."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_validation_tuning(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device_value=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary = result["validation_metrics"]
    print(
        json.dumps(
            {
                "output_dir": result["output_dir"],
                "thresholds": result["thresholds"],
                "threshold_independent": summary["threshold_independent"],
                "f1_at_0_5": summary["f1_at_0_5"],
                "f1_tuned": summary["f1_tuned"],
            },
            indent=2,
            default=_json_default,
        )
    )


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state", "epoch", "config", "metrics"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint is missing required keys: {sorted(required)}")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a mapping")
    return checkpoint, config


def _checkpoint_labels(config: dict[str, Any]) -> list[str]:
    labels = list(config.get("data", {}).get("label_cols") or [])
    if labels != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint must contain the canonical ordered 14 NIH labels")
    return labels


def validate_designated_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Require the designated ResNet18 best checkpoint used by baseline v1."""
    path = Path(checkpoint_path)
    if path.name != "best.pt" or path.parent.name != "imagenet_pretrained":
        raise ValueError(
            "Baseline v1 threshold freeze requires checkpoints/imagenet_pretrained/best.pt"
        )
    if str(config.get("model", {}).get("name", "")).lower() != "resnet18":
        raise ValueError("Baseline v1 threshold freeze requires a ResNet18 checkpoint")
    score = metrics.get("macro_auroc") if isinstance(metrics, dict) else None
    if not isinstance(score, (int, float)) or not np.isfinite(score):
        raise ValueError("best.pt must record a finite validation macro AUROC")


def _loader_config(
    data_config: dict[str, Any],
    *,
    batch_size: int | None,
    num_workers: int | None,
) -> dict[str, Any]:
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
    result = {key: value for key, value in data_config.items() if key in names}
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        result["batch_size"] = batch_size
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        result["num_workers"] = num_workers
    return result


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(value),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_stage(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


if __name__ == "__main__":
    main()
