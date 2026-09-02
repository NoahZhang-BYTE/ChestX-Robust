"""Fit validation-only B2 thresholds on a fixed 0.01 grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from baseline.data import build_evaluation_loader
from baseline.labels import LABEL_COLUMNS
from baseline.models import build_model
from train import _resolve_device
from workflow_state import atomic_write_csv, atomic_write_json, update_stage


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"B2 checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"B2 checkpoint is invalid: {path}")
    state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    config = checkpoint.get("config")
    if not isinstance(state, dict) or not isinstance(config, dict):
        raise ValueError("B2 checkpoint must contain model state and config")
    checkpoint["model_state_dict"] = state
    return checkpoint, config


def _collect_validation(model: torch.nn.Module, loader, device: torch.device, amp_enabled: bool) -> tuple[np.ndarray, np.ndarray]:
    targets, probabilities = [], []
    model.eval()
    with torch.inference_mode():
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
            targets.append(batch_targets.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    if not targets:
        raise ValueError("validation loader produced no batches")
    return np.concatenate(targets, axis=0).astype(np.uint8), np.concatenate(probabilities, axis=0).astype(np.float32)


def _best_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.arange(0.01, 1.0, 0.01, dtype=np.float64)
    scores = np.asarray([f1_score(target, probability >= threshold, zero_division=0) for threshold in candidates])
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
    tied = candidates[best]
    threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    return threshold, float(scores[best[np.argmin(np.abs(tied - 0.5))]])


def tune_b2_thresholds(
    checkpoint_path: str | Path = "outputs/B2_densenet121_sqrt_posweight/best.pt",
    output_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight/thresholds",
    workflow_state_path: str | Path = "workflow_state.json",
    device_value: str = "auto",
    stage_name: str = "B2_threshold_tuning",
    experiment_name: str = "B2",
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"B2 threshold output already exists: {output_dir}")
    checkpoint, config = _load_checkpoint(checkpoint_path)
    labels = list(config.get("data", {}).get("label_cols") or LABEL_COLUMNS)
    if labels != list(LABEL_COLUMNS):
        raise ValueError("B2 checkpoint labels must match the canonical 14-label order")
    loader_config = {key: value for key, value in config["data"].items() if key in {
        "csv_path", "image_root", "image_col", "label_cols", "image_size", "batch_size", "num_workers", "prefetch_factor", "persistent_workers"
    }}
    loader, loader_labels = build_evaluation_loader(split="val", **loader_config)
    if loader_labels != labels:
        raise ValueError("validation label order does not match checkpoint")
    device = _resolve_device(device_value)
    model_config = config["model"]
    model = build_model(model_config["name"], len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    amp_enabled = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    targets, probabilities = _collect_validation(model, loader, device, amp_enabled)
    thresholds = []
    rows = []
    for index, label in enumerate(labels):
        target, probability = targets[:, index], probabilities[:, index]
        tuned_threshold, tuned_f1 = _best_threshold(target, probability)
        fixed_prediction = probability >= 0.5
        tuned_prediction = probability >= tuned_threshold
        auroc = float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else None
        auprc = float(average_precision_score(target, probability)) if np.unique(target).size == 2 else None
        fixed_f1 = float(f1_score(target, fixed_prediction, zero_division=0))
        rows.append({
            "label_index": index, "label_name": label,
            "positive_count": int(target.sum()), "prevalence": float(target.mean()),
            "fixed_threshold": 0.5,
            "fixed_precision": float(precision_score(target, fixed_prediction, zero_division=0)),
            "fixed_recall": float(recall_score(target, fixed_prediction, zero_division=0)),
            "fixed_f1": fixed_f1,
            "best_threshold": tuned_threshold,
            "precision_tuned": float(precision_score(target, tuned_prediction, zero_division=0)),
            "recall_tuned": float(recall_score(target, tuned_prediction, zero_division=0)),
            "f1_tuned": tuned_f1, "auroc": auroc, "auprc": auprc,
            "delta_f1": tuned_f1 - fixed_f1,
        })
        thresholds.append(tuned_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "y_true_val.npy", targets)
    np.save(output_dir / "y_prob_val.npy", probabilities)
    artifact = {
        "schema_version": 1, "source": "validation", "fit_split": "val",
        "experiment": experiment_name,
        "strategy": "grid_0.01_to_0.99_max_per_label_f1",
        "tie_break": "threshold closest to 0.5",
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]),
        "label_cols": labels, "thresholds": thresholds,
        "thresholds_by_label": dict(zip(labels, thresholds, strict=True)),
    }
    metrics = {
        "split": "val", "sample_count": int(targets.shape[0]),
        "macro_AUROC": float(np.nanmean([row["auroc"] for row in rows if row["auroc"] is not None])),
        "micro_AUROC": float(roc_auc_score(targets.ravel(), probabilities.ravel())),
        "macro_AUPRC": float(np.nanmean([row["auprc"] for row in rows if row["auprc"] is not None])),
        "micro_AUPRC": float(average_precision_score(targets.ravel(), probabilities.ravel())),
        "macro_F1_fixed05": float(f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0)),
        "micro_F1_fixed05": float(f1_score(targets, probabilities >= 0.5, average="micro", zero_division=0)),
        "sample_F1_fixed05": float(f1_score(targets, probabilities >= 0.5, average="samples", zero_division=0)),
        "macro_F1_tuned": float(f1_score(targets, probabilities >= np.asarray(thresholds), average="macro", zero_division=0)),
        "micro_F1_tuned": float(f1_score(targets, probabilities >= np.asarray(thresholds), average="micro", zero_division=0)),
        "sample_F1_tuned": float(f1_score(targets, probabilities >= np.asarray(thresholds), average="samples", zero_division=0)),
        "threshold_independent_note": "AUROC and AUPRC do not change with threshold tuning.",
    }
    atomic_write_json(output_dir / "thresholds.json", artifact)
    frozen_path = output_dir / "thresholds_frozen_for_test.json"
    atomic_write_json(frozen_path, artifact)
    frozen_sha256 = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    atomic_write_json(output_dir / "threshold_freeze.json", {
        "status": "frozen",
        "fit_split": "val",
        "frozen_thresholds_path": str(frozen_path.resolve()),
        "sha256": frozen_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_used_for_tuning": False,
    })
    atomic_write_json(output_dir / "validation_metrics.json", metrics)
    atomic_write_csv(output_dir / "threshold_tuning_val.csv", rows[0].keys(), rows)
    update_stage(workflow_state_path, stage_name, "completed")
    return {"output_dir": str(output_dir), "thresholds": thresholds, "frozen_thresholds_sha256": frozen_sha256, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune B2 thresholds on validation only.")
    parser.add_argument("--checkpoint", default="outputs/B2_densenet121_sqrt_posweight/best.pt")
    parser.add_argument("--output-dir", default="outputs/B2_densenet121_sqrt_posweight/thresholds")
    parser.add_argument("--workflow-state", default="workflow_state.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage-name", default="B2_threshold_tuning")
    parser.add_argument("--experiment-name", default="B2")
    args = parser.parse_args()
    print(json.dumps(tune_b2_thresholds(args.checkpoint, args.output_dir, args.workflow_state, args.device, args.stage_name, args.experiment_name), indent=2))


if __name__ == "__main__":
    main()
