"""Evaluate B2 on test using thresholds already frozen on validation."""

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


def evaluate_b2_frozen_test(
    checkpoint_path: str | Path = "outputs/B2_densenet121_sqrt_posweight/best.pt",
    thresholds_path: str | Path = "outputs/B2_densenet121_sqrt_posweight/thresholds/thresholds_frozen_for_test.json",
    output_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight/test",
    workflow_state_path: str | Path = "workflow_state.json",
    device_value: str = "auto",
    stage_name: str = "B2_test",
    experiment_name: str = "B2",
) -> dict[str, Any]:
    checkpoint_path, thresholds_path, output_dir = map(Path, (checkpoint_path, thresholds_path, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"B2 test output already exists: {output_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    artifact = json.loads(thresholds_path.read_text(encoding="utf-8"))
    freeze_path = thresholds_path.with_name("threshold_freeze.json")
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Threshold freeze manifest does not exist: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("fit_split") != "val":
        raise ValueError("Threshold freeze manifest is not a validation-only frozen artifact")
    if freeze.get("sha256") != hashlib.sha256(thresholds_path.read_bytes()).hexdigest():
        raise ValueError("Frozen thresholds SHA256 does not match threshold freeze manifest")
    labels = list(artifact["label_cols"])
    if labels != list(LABEL_COLUMNS) or artifact.get("fit_split") != "val":
        raise ValueError("Frozen B2 thresholds must be validation-only canonical thresholds")
    thresholds = np.asarray(artifact["thresholds"], dtype=np.float64)
    loader_config = {key: value for key, value in config["data"].items() if key in {
        "csv_path", "image_root", "image_col", "label_cols", "image_size", "batch_size", "num_workers", "prefetch_factor", "persistent_workers"
    }}
    loader, loader_labels = build_evaluation_loader(split="test", **loader_config)
    if loader_labels != labels:
        raise ValueError("test label order does not match frozen thresholds")
    device = _resolve_device(device_value)
    model = build_model(config["model"]["name"], len(labels), pretrained=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint["model_state"]), strict=True)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    amp_enabled = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
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
    y_true, y_prob = np.concatenate(targets).astype(np.uint8), np.concatenate(probabilities).astype(np.float32)
    fixed = y_prob >= 0.5
    tuned = y_prob >= thresholds.reshape(1, -1)
    def summary(prediction: np.ndarray) -> dict[str, float]:
        return {"macro_F1": float(f1_score(y_true, prediction, average="macro", zero_division=0)), "micro_F1": float(f1_score(y_true, prediction, average="micro", zero_division=0)), "sample_F1": float(f1_score(y_true, prediction, average="samples", zero_division=0))}
    auc = [roc_auc_score(y_true[:, i], y_prob[:, i]) if np.unique(y_true[:, i]).size == 2 else None for i in range(len(labels))]
    auprc = [average_precision_score(y_true[:, i], y_prob[:, i]) if np.unique(y_true[:, i]).size == 2 else None for i in range(len(labels))]
    per_label = []
    for i, label in enumerate(labels):
        per_label.append({
            "label_index": i, "label_name": label, "frozen_threshold": float(thresholds[i]),
            "fixed05_precision": float(precision_score(y_true[:, i], fixed[:, i], zero_division=0)),
            "fixed05_recall": float(recall_score(y_true[:, i], fixed[:, i], zero_division=0)),
            "fixed05_f1": float(f1_score(y_true[:, i], fixed[:, i], zero_division=0)),
            "tuned_precision": float(precision_score(y_true[:, i], tuned[:, i], zero_division=0)),
            "tuned_recall": float(recall_score(y_true[:, i], tuned[:, i], zero_division=0)),
            "tuned_f1": float(f1_score(y_true[:, i], tuned[:, i], zero_division=0)),
            "auroc": auc[i], "auprc": auprc[i],
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "y_true_test.npy", y_true)
    np.save(output_dir / "y_prob_test.npy", y_prob)
    summary_payload = {
        "experiment": experiment_name, "split": "test", "checkpoint": str(checkpoint_path),
        "test_used_for_tuning": False,
        "macro_AUROC": float(np.nanmean(auc)), "micro_AUROC": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "macro_AUPRC": float(np.nanmean(auprc)), "micro_AUPRC": float(average_precision_score(y_true.ravel(), y_prob.ravel())),
        "fixed05": summary(fixed), "tuned": summary(tuned),
        "frozen_thresholds_path": str(thresholds_path.resolve()),
        "frozen_thresholds_sha256": freeze["sha256"],
    }
    atomic_write_json(output_dir / "test_fixed05_metrics.json", {**summary_payload, "metrics": summary(fixed)})
    atomic_write_json(output_dir / "test_tuned_metrics.json", {**summary_payload, "metrics": summary(tuned)})
    atomic_write_csv(output_dir / "test_per_label_comparison.csv", per_label[0].keys(), per_label)
    update_stage(workflow_state_path, stage_name, "completed")
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B2 test with frozen validation thresholds.")
    parser.add_argument("--checkpoint", default="outputs/B2_densenet121_sqrt_posweight/best.pt")
    parser.add_argument("--thresholds", default="outputs/B2_densenet121_sqrt_posweight/thresholds/thresholds_frozen_for_test.json")
    parser.add_argument("--output-dir", default="outputs/B2_densenet121_sqrt_posweight/test")
    parser.add_argument("--workflow-state", default="workflow_state.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage-name", default="B2_test")
    parser.add_argument("--experiment-name", default="B2")
    args = parser.parse_args()
    print(json.dumps(evaluate_b2_frozen_test(args.checkpoint, args.thresholds, args.output_dir, args.workflow_state, args.device, args.stage_name, args.experiment_name), indent=2))


if __name__ == "__main__":
    main()
