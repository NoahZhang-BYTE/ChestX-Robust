from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def normalize_thresholds(threshold: float | Sequence[float], num_labels: int) -> np.ndarray:
    if num_labels < 1:
        raise ValueError("num_labels must be positive")
    scalar = np.isscalar(threshold)
    values = np.asarray([threshold] if scalar else threshold, dtype=np.float64)
    if values.ndim != 1 or (not scalar and values.size != num_labels):
        raise ValueError("thresholds must contain one threshold per label")
    if scalar:
        values = np.repeat(values, num_labels)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("thresholds must be between 0 and 1")
    return values


def apply_thresholds(probabilities: np.ndarray, thresholds: float | Sequence[float]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    values = normalize_thresholds(thresholds, probabilities.shape[1])
    return (probabilities >= values.reshape(1, -1)).astype(np.int64)


def fit_per_class_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_cols: Sequence[str],
    fit_split: str = "val",
    objective: str = "binary_f1",
) -> dict[str, Any]:
    if fit_split != "val":
        raise ValueError("fit_split must be 'val'")
    if objective != "binary_f1":
        raise ValueError("objective must be 'binary_f1'")
    targets, probabilities = _validate_arrays(targets, probabilities)
    labels = list(label_cols)
    if len(labels) != targets.shape[1]:
        raise ValueError("label_cols must match the number of labels")

    thresholds: list[float] = []
    available: list[bool] = []
    reasons: list[str | None] = []
    for column in range(targets.shape[1]):
        target = targets[:, column]
        probability = probabilities[:, column]
        positive = int(target.sum())
        negative = int(target.size - positive)
        if positive == 0:
            thresholds.append(1.0)
            available.append(False)
            reasons.append("no_positive")
            continue
        if negative == 0:
            thresholds.append(1.0)
            available.append(False)
            reasons.append("no_negative")
            continue
        candidates = np.unique(np.concatenate(([0.0, 1.0], probability.astype(np.float64))))
        scores = np.array([f1_score(target, probability >= candidate, zero_division=0) for candidate in candidates])
        best_score = scores.max()
        best_candidates = candidates[np.isclose(scores, best_score, rtol=0.0, atol=1e-12)]
        thresholds.append(float(np.round(best_candidates[-1], 12)))
        available.append(True)
        reasons.append(None)

    return {
        "schema_version": 1,
        "fit_split": fit_split,
        "objective": objective,
        "label_cols": labels,
        "thresholds": thresholds,
        "available": available,
        "reasons": reasons,
    }


def load_threshold_artifact(path: str | Path, label_cols: Sequence[str]) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read threshold artifact: {path}") from error
    if not isinstance(artifact, dict):
        raise ValueError("Threshold artifact must be a JSON object")
    if artifact.get("fit_split") != "val":
        raise ValueError("Threshold artifact must be fitted on val")
    if artifact.get("label_cols") != list(label_cols):
        raise ValueError("Threshold artifact label order does not match evaluation labels")
    values = normalize_thresholds(artifact.get("thresholds", []), len(label_cols))
    artifact["thresholds"] = values.tolist()
    return artifact


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | Sequence[float] = 0.5,
    label_cols: Sequence[str] | None = None,
) -> dict[str, Any]:
    targets, probabilities = _validate_arrays(targets, probabilities)
    labels = list(label_cols) if label_cols is not None else [f"label_{i}" for i in range(targets.shape[1])]
    if len(labels) != targets.shape[1]:
        raise ValueError("label_cols must match the number of labels")
    thresholds = normalize_thresholds(threshold, targets.shape[1])
    predictions = apply_thresholds(probabilities, thresholds)

    per_label: list[dict[str, Any]] = []
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    f1_values: list[float] = []
    for column, label in enumerate(labels):
        target = targets[:, column]
        probability = probabilities[:, column]
        prediction = predictions[:, column]
        positive_count = int(target.sum())
        negative_count = int(target.size - positive_count)
        degenerate = positive_count == 0 or negative_count == 0
        entry: dict[str, Any] = {
            "label": label,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "prevalence": float(target.mean()) if target.size else float("nan"),
            "threshold": float(thresholds[column]),
            "degenerate": degenerate,
        }
        if degenerate:
            entry.update({"precision": None, "recall": None, "f1": None, "auroc": None, "auprc": None})
        else:
            entry.update(
                {
                    "precision": float(precision_score(target, prediction, zero_division=0)),
                    "recall": float(recall_score(target, prediction, zero_division=0)),
                    "f1": float(f1_score(target, prediction, zero_division=0)),
                    "auroc": float(roc_auc_score(target, probability)),
                    "auprc": float(average_precision_score(target, probability)),
                }
            )
            auroc_values.append(entry["auroc"])
            auprc_values.append(entry["auprc"])
            f1_values.append(entry["f1"])
        per_label.append(entry)

    return {
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else float("nan"),
        "micro_auroc": _safe_micro_score(roc_auc_score, targets, probabilities),
        "macro_auprc": float(np.mean(auprc_values)) if auprc_values else float("nan"),
        "micro_auprc": _safe_micro_score(average_precision_score, targets, probabilities),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "sample_f1": float(f1_score(targets, predictions, average="samples", zero_division=0)),
        "per_label": per_label,
    }


def _safe_micro_score(metric, targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float("nan")
    return float(metric(targets.ravel(), probabilities.ravel()))


def _validate_arrays(targets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 2 or probabilities.shape != targets.shape:
        raise ValueError("targets and probabilities must be 2D arrays with matching shapes")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("targets must contain only 0 or 1")
    return targets, probabilities
