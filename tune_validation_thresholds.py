"""Validation-only per-label F1 threshold fitting for B3 and B4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from baseline.data import build_evaluation_loader
from baseline.labels import LABEL_COLUMNS
from baseline.models import build_model
from train import _resolve_device
from tune_b2_thresholds import _best_threshold, _collect_validation, _load_checkpoint
from workflow_state import atomic_write_csv, atomic_write_json, update_stage


def fit_threshold_grid(targets: np.ndarray, probabilities: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    """Fit thresholds from supplied validation arrays; no split name is configurable."""
    targets = np.asarray(targets, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    labels = list(labels)
    if targets.ndim != 2 or probabilities.shape != targets.shape or targets.shape[1] != len(labels):
        raise ValueError("targets, probabilities, and labels must have matching 2D shapes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    rows: list[dict[str, Any]] = []
    thresholds: list[float] = []
    for index, label in enumerate(labels):
        target, probability = targets[:, index], probabilities[:, index]
        if np.unique(target).size != 2:
            raise ValueError(f"validation label is degenerate: {label}")
        threshold, tuned_f1 = _best_threshold(target, probability)
        fixed, tuned = probability >= 0.5, probability >= threshold
        rows.append({
            "label_index": index,
            "label_name": label,
            "positive_count": int(target.sum()),
            "prevalence": float(target.mean()),
            "fixed_threshold": 0.5,
            "fixed_precision": float(precision_score(target, fixed, zero_division=0)),
            "fixed_recall": float(recall_score(target, fixed, zero_division=0)),
            "fixed_f1": float(f1_score(target, fixed, zero_division=0)),
            "best_threshold": threshold,
            "tuned_precision": float(precision_score(target, tuned, zero_division=0)),
            "tuned_recall": float(recall_score(target, tuned, zero_division=0)),
            "tuned_f1": tuned_f1,
            "delta_f1": tuned_f1 - float(f1_score(target, fixed, zero_division=0)),
            "auroc": float(roc_auc_score(target, probability)),
            "auprc": float(average_precision_score(target, probability)),
        })
        thresholds.append(threshold)
    values = np.asarray(thresholds, dtype=np.float64)
    metrics = {
        "split": "val",
        "sample_count": int(targets.shape[0]),
        "macro_AUROC": float(np.mean([row["auroc"] for row in rows])),
        "micro_AUROC": float(roc_auc_score(targets.ravel(), probabilities.ravel())),
        "macro_AUPRC": float(np.mean([row["auprc"] for row in rows])),
        "micro_AUPRC": float(average_precision_score(targets.ravel(), probabilities.ravel())),
        "macro_F1_fixed05": float(f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0)),
        "micro_F1_fixed05": float(f1_score(targets, probabilities >= 0.5, average="micro", zero_division=0)),
        "sample_F1_fixed05": float(f1_score(targets, probabilities >= 0.5, average="samples", zero_division=0)),
        "macro_F1_tuned": float(f1_score(targets, probabilities >= values, average="macro", zero_division=0)),
        "micro_F1_tuned": float(f1_score(targets, probabilities >= values, average="micro", zero_division=0)),
        "sample_F1_tuned": float(f1_score(targets, probabilities >= values, average="samples", zero_division=0)),
        "threshold_independent_note": "AUROC and AUPRC do not change with threshold tuning.",
    }
    return {
        "fit_split": "val",
        "strategy": "grid_0.01_to_0.99_max_per_label_f1",
        "tie_break": "threshold closest to 0.5",
        "label_cols": labels,
        "thresholds": thresholds,
        "thresholds_by_label": dict(zip(labels, thresholds, strict=True)),
        "per_label": rows,
        "metrics": metrics,
    }


def tune_validation_thresholds(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    experiment: str,
    stage_name: str,
    workflow_state_path: str | Path = "workflow_state.json",
    device_value: str = "auto",
) -> dict[str, Any]:
    checkpoint_path, output_dir = Path(checkpoint_path).resolve(), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"threshold output already exists: {output_dir}")
    checkpoint, config = _load_checkpoint(checkpoint_path)
    labels = list(config.get("data", {}).get("label_cols") or LABEL_COLUMNS)
    if labels != list(LABEL_COLUMNS):
        raise ValueError("checkpoint label order does not match canonical labels")
    loader_keys = {"csv_path", "image_root", "image_col", "label_cols", "image_size", "batch_size", "num_workers", "prefetch_factor", "persistent_workers"}
    loader, loader_labels = build_evaluation_loader(split="val", **{key: value for key, value in config["data"].items() if key in loader_keys})
    if loader_labels != labels:
        raise ValueError("validation loader label order does not match checkpoint")
    device = _resolve_device(device_value)
    model = build_model(config["model"]["name"], len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=__import__("torch").channels_last)
    amp_enabled = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    targets, probabilities = _collect_validation(model, loader, device, amp_enabled)
    result = fit_threshold_grid(targets, probabilities, labels)
    artifact = {
        "schema_version": 1,
        "experiment": experiment,
        "source": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_used_for_tuning": False,
        **{key: result[key] for key in ("fit_split", "strategy", "tie_break", "label_cols", "thresholds", "thresholds_by_label")},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "y_true_val.npy", targets)
    np.save(output_dir / "y_prob_val.npy", probabilities)
    atomic_write_json(output_dir / "thresholds_validation.json", artifact)
    atomic_write_json(output_dir / "validation_metrics.json", result["metrics"])
    atomic_write_csv(output_dir / "threshold_tuning_val.csv", result["per_label"][0].keys(), result["per_label"])
    update_stage(workflow_state_path, stage_name, "completed")
    return {"artifact": artifact, "metrics": result["metrics"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit validation-only 0.01-grid thresholds for B3/B4.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--workflow-state", default="workflow_state.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(tune_validation_thresholds(args.checkpoint, args.output_dir, args.experiment, args.stage_name, args.workflow_state, args.device), indent=2))


if __name__ == "__main__":
    main()
