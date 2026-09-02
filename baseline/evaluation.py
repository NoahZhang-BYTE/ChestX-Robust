"""Prediction collection and metrics for multi-label evaluation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


def collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device | str,
    amp_enabled: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run eval/no-grad inference and return targets plus sigmoid probabilities."""
    device = torch.device(device)
    target_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
            target_batches.append(targets.detach().cpu().numpy())
            probability_batches.append(torch.sigmoid(logits).detach().cpu().numpy())
    if not target_batches:
        raise ValueError("loader produced no batches")
    return _validate_arrays(
        np.concatenate(target_batches, axis=0),
        np.concatenate(probability_batches, axis=0),
    )


def compute_auc_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_cols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return per-class and macro/micro AUROC and AUPRC."""
    targets, probabilities = _validate_arrays(targets, probabilities)
    labels = _validate_labels(label_cols, targets.shape[1])
    per_class: list[dict[str, float | str]] = []
    aurocs: list[float] = []
    auprcs: list[float] = []
    for index, label in enumerate(labels):
        target = targets[:, index]
        probability = probabilities[:, index]
        if np.unique(target).size < 2:
            auroc = float("nan")
            auprc = float("nan")
        else:
            auroc = float(roc_auc_score(target, probability))
            auprc = float(average_precision_score(target, probability))
            aurocs.append(auroc)
            auprcs.append(auprc)
        per_class.append({"label": label, "auroc": auroc, "auprc": auprc})
    return {
        "macro_auroc": _mean_or_nan(aurocs),
        "micro_auroc": _micro_auc(roc_auc_score, targets, probabilities),
        "macro_auprc": _mean_or_nan(auprcs),
        "micro_auprc": _micro_auc(average_precision_score, targets, probabilities),
        "per_class": per_class,
    }


def compute_f1_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: float | Sequence[float] = 0.5,
) -> dict[str, float]:
    """Return aggregate F1 scores for scalar or per-class thresholds."""
    targets, probabilities = _validate_arrays(targets, probabilities)
    values = _normalize_thresholds(thresholds, targets.shape[1])
    predictions = probabilities >= values.reshape(1, -1)
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "sample_f1": float(f1_score(targets, predictions, average="samples", zero_division=0)),
    }


def compute_per_class_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_cols: Sequence[str] | None = None,
    thresholds: float | Sequence[float] = 0.5,
) -> list[dict[str, float | int | str | bool]]:
    """Return threshold-independent and threshold-dependent metrics per class."""
    targets, probabilities = _validate_arrays(targets, probabilities)
    labels = _validate_labels(label_cols, targets.shape[1])
    threshold_values = _normalize_thresholds(thresholds, targets.shape[1])
    auc_rows = compute_auc_metrics(targets, probabilities, labels)["per_class"]
    auc_by_label = {str(row["label"]): row for row in auc_rows}
    rows: list[dict[str, float | int | str | bool]] = []
    for index, label in enumerate(labels):
        target = targets[:, index]
        probability = probabilities[:, index]
        prediction = probability >= threshold_values[index]
        positive_count = int(target.sum())
        negative_count = int(target.size - positive_count)
        true_negative = int(np.logical_and(target == 0, ~prediction).sum())
        specificity = float(true_negative / negative_count) if negative_count else float("nan")
        rows.append(
            {
                "label": label,
                "threshold": float(threshold_values[index]),
                "auroc": float(auc_by_label[label]["auroc"]),
                "auprc": float(auc_by_label[label]["auprc"]),
                "f1": float(f1_score(target, prediction, zero_division=0)),
                "precision": float(precision_score(target, prediction, zero_division=0)),
                "recall": float(recall_score(target, prediction, zero_division=0)),
                "specificity": specificity,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "total_count": int(target.size),
                "degenerate": positive_count == 0 or negative_count == 0,
            }
        )
    return rows


def find_per_class_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Fit validation F1-maximizing thresholds from PR-curve candidates."""
    targets, probabilities = _validate_arrays(targets, probabilities)
    thresholds = np.empty(targets.shape[1], dtype=np.float64)
    for index in range(targets.shape[1]):
        target = targets[:, index]
        probability = probabilities[:, index]
        if np.unique(target).size < 2:
            warnings.warn(
                f"class index {index} has degenerate labels; falling back to threshold 0.5",
                RuntimeWarning,
                stacklevel=2,
            )
            thresholds[index] = 0.5
            continue
        precision, recall, candidates = precision_recall_curve(target, probability)
        f1_values = _f1_from_precision_recall(precision[:-1], recall[:-1])
        best = np.flatnonzero(
            np.isclose(f1_values, np.nanmax(f1_values), rtol=0.0, atol=1e-12)
        )
        thresholds[index] = float(candidates[best[-1]])
    return _normalize_thresholds(thresholds, targets.shape[1])


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    split: str = "val",
    threshold: float | Sequence[float] = 0.5,
    threshold_path: str | Path | None = None,
    fit_thresholds: bool = False,
) -> dict[str, Any]:
    """Compatibility API for evaluating one named split without file output."""
    if fit_thresholds and split != "val":
        raise ValueError("Threshold fitting is only allowed on val, never test")
    if split == "test":
        raise ValueError(
            "Direct test evaluation is forbidden; use the frozen protocol runner"
        )

    from .config import load_config
    from .data import build_evaluation_loader
    from .metrics import load_threshold_artifact
    from .models import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config") or {}
    config = load_config(config_path) if config_path is not None else checkpoint_config
    if not config:
        raise ValueError("Checkpoint does not contain a config; config_path is required")
    data_config = dict(config["data"])
    evaluation_keys = {
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
    loader, label_cols = build_evaluation_loader(
        split=split,
        **{key: value for key, value in data_config.items() if key in evaluation_keys},
    )
    model_config = config["model"]
    model = build_model(model_config["name"], len(label_cols), pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device_value = config.get("device", "cpu")
    device = torch.device(
        "cuda" if device_value == "auto" and torch.cuda.is_available() else device_value
    )
    model.to(device)
    amp_enabled = bool(config.get("training", {}).get("amp", False)) and device.type == "cuda"
    targets, probabilities = collect_predictions(
        model, loader, device, amp_enabled=amp_enabled
    )

    thresholds: float | Sequence[float] = threshold
    artifact = None
    if threshold_path is not None:
        artifact = load_threshold_artifact(threshold_path, label_cols)
        thresholds = artifact["thresholds"]
    if fit_thresholds:
        thresholds = find_per_class_thresholds(targets, probabilities)
    auc = compute_auc_metrics(targets, probabilities, label_cols)
    metrics = {
        **{key: value for key, value in auc.items() if key != "per_class"},
        **compute_f1_metrics(targets, probabilities, thresholds),
        "per_label": compute_per_class_metrics(targets, probabilities, label_cols, thresholds),
    }
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "num_samples": int(targets.shape[0]),
        "thresholds": (
            float(thresholds)
            if np.isscalar(thresholds)
            else list(np.asarray(thresholds, dtype=float))
        ),
        "metrics": metrics,
    }
    if artifact is not None:
        result["threshold_artifact"] = artifact
    return result


def _validate_arrays(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 2 or probabilities.ndim != 2 or targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must be 2D arrays with matching shapes")
    if targets.shape[0] == 0 or targets.shape[1] == 0:
        raise ValueError("targets and probabilities must not be empty")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must be between 0 and 1")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("targets must contain only 0 or 1")
    return targets.astype(np.int64, copy=False), probabilities


def _validate_labels(label_cols: Sequence[str] | None, num_classes: int) -> list[str]:
    labels = (
        list(label_cols)
        if label_cols is not None
        else [f"label_{index}" for index in range(num_classes)]
    )
    if len(labels) != num_classes:
        raise ValueError("label_cols must contain one label per class")
    return labels


def _normalize_thresholds(
    thresholds: float | Sequence[float], num_classes: int
) -> np.ndarray:
    values = np.asarray(
        [thresholds] if np.isscalar(thresholds) else thresholds, dtype=np.float64
    )
    if values.ndim != 1 or (values.size != 1 and values.size != num_classes):
        raise ValueError("thresholds must be scalar or contain one threshold per class")
    if values.size == 1:
        values = np.repeat(values, num_classes)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("thresholds must be finite values between 0 and 1")
    return values


def _mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _micro_auc(metric, targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size != 2:
        return float("nan")
    return float(metric(targets.ravel(), probabilities.ravel()))


def _f1_from_precision_recall(
    precision: np.ndarray, recall: np.ndarray
) -> np.ndarray:
    denominator = precision + recall
    return np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
