from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _safe_macro_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    scores = []
    for column in range(targets.shape[1]):
        if np.unique(targets[:, column]).size < 2:
            continue
        scores.append(roc_auc_score(targets[:, column], probabilities[:, column]))
    return float(np.mean(scores)) if scores else float("nan")


def _safe_micro_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float("nan")
    return float(roc_auc_score(targets.ravel(), probabilities.ravel()))


def multilabel_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    targets = np.asarray(targets).astype(np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if targets.ndim != 2 or probabilities.shape != targets.shape:
        raise ValueError("targets and probabilities must be 2D arrays with matching shapes")
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "macro_auroc": _safe_macro_auroc(targets, probabilities),
        "micro_auroc": _safe_micro_auroc(targets, probabilities),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "sample_f1": float(f1_score(targets, predictions, average="samples", zero_division=0)),
    }
