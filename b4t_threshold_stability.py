"""Validation-only 5-fold threshold stability analysis for B4."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from baseline.evaluation import (
    compute_auc_metrics,
    compute_f1_metrics,
    compute_per_class_metrics,
    find_per_class_thresholds,
)
from baseline.labels import LABEL_COLUMNS


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_ready(value), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    validation_dir: str | Path,
    output_dir: str | Path,
    test_dir: str | Path | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    validation_dir = Path(validation_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    targets = np.load(validation_dir / "validation_targets.npy")
    probabilities = np.load(validation_dir / "validation_probabilities.npy")
    if targets.ndim != 2 or probabilities.shape != targets.shape:
        raise ValueError("validation arrays must be 2D and shape-matched")
    if targets.shape[1] != len(LABEL_COLUMNS):
        raise ValueError("validation arrays do not match canonical label count")
    full_thresholds = find_per_class_thresholds(targets, probabilities)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_thresholds: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    for fold, (fit_indices, heldout_indices) in enumerate(splitter.split(targets), start=1):
        thresholds = find_per_class_thresholds(targets[fit_indices], probabilities[fit_indices])
        fold_thresholds.append(thresholds)
        metrics = compute_f1_metrics(targets[heldout_indices], probabilities[heldout_indices], thresholds)
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": int(fit_indices.size),
                "heldout_count": int(heldout_indices.size),
                **metrics,
            }
        )

    threshold_matrix = np.vstack(fold_thresholds)
    median_thresholds = np.median(threshold_matrix, axis=0)
    mean_thresholds = np.mean(threshold_matrix, axis=0)
    stability_rows = []
    for index, label in enumerate(LABEL_COLUMNS):
        values = threshold_matrix[:, index]
        stability_rows.append(
            {
                "label": label,
                "full_val_threshold": float(full_thresholds[index]),
                "fold_1_threshold": float(values[0]),
                "fold_2_threshold": float(values[1]),
                "fold_3_threshold": float(values[2]),
                "fold_4_threshold": float(values[3]),
                "fold_5_threshold": float(values[4]),
                "mean_threshold": float(mean_thresholds[index]),
                "median_threshold": float(median_thresholds[index]),
                "std_threshold": float(np.std(values, ddof=1)),
                "min_threshold": float(np.min(values)),
                "max_threshold": float(np.max(values)),
                "range_threshold": float(np.max(values) - np.min(values)),
            }
        )

    full_metrics = {
        "threshold_independent": compute_auc_metrics(targets, probabilities, LABEL_COLUMNS),
        "single_full_val": {
            "thresholds": full_thresholds.tolist(),
            **compute_f1_metrics(targets, probabilities, full_thresholds),
        },
        "kfold_mean": {
            "thresholds": mean_thresholds.tolist(),
            **compute_f1_metrics(targets, probabilities, mean_thresholds),
        },
        "kfold_median": {
            "thresholds": median_thresholds.tolist(),
            **compute_f1_metrics(targets, probabilities, median_thresholds),
        },
        "fixed_0_5": {
            "thresholds": [0.5] * len(LABEL_COLUMNS),
            **compute_f1_metrics(targets, probabilities, 0.5),
        },
    }

    test_metrics = None
    if test_dir is not None:
        test_dir = Path(test_dir).resolve()
        summary = json.loads((test_dir / "test_summary.json").read_text(encoding="utf-8"))
        if summary.get("test_used_for_tuning") is not False:
            raise ValueError("test artifact does not declare test_used_for_tuning=false")
        test_targets = np.load(test_dir / "test_targets.npy")
        test_probabilities = np.load(test_dir / "test_probabilities.npy")
        test_metrics = {
            "test_used_for_tuning": False,
            "fixed_0_5": compute_f1_metrics(test_targets, test_probabilities, 0.5),
            "single_full_val": compute_f1_metrics(test_targets, test_probabilities, full_thresholds),
            "kfold_mean": compute_f1_metrics(test_targets, test_probabilities, mean_thresholds),
            "kfold_median": compute_f1_metrics(test_targets, test_probabilities, median_thresholds),
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "threshold_stability.csv", stability_rows)
    _write_csv(output_dir / "fold_metrics.csv", fold_rows)
    _write_json(
        output_dir / "threshold_stability_summary.json",
        {
            "schema_version": 1,
            "stage": "B4T_validation_threshold_stability",
            "validation_dir": str(validation_dir),
            "test_dir": str(test_dir) if test_dir is not None else None,
            "fit_split": "val",
            "test_used_for_tuning": False,
            "n_splits": n_splits,
            "seed": seed,
            "labels": list(LABEL_COLUMNS),
            "full_validation": full_metrics,
            "fold_metrics": fold_rows,
            "test_evaluation": test_metrics,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {"output_dir": str(output_dir), "full_validation": full_metrics, "test_evaluation": test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(_json_ready(run_analysis(**vars(args))), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
