"""Diagnose class imbalance and per-class errors for frozen B1 evaluation.

This analysis consumes a validation-frozen evaluation bundle. It never fits a
threshold, evaluates a new checkpoint, or modifies any formal training result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from baseline.evaluation import (
    compute_auc_metrics,
    compute_f1_metrics,
    compute_per_class_metrics,
)
from run_final_test import load_and_validate_frozen_bundle


DEFAULT_FROZEN_DIR = Path("artifacts/B1_densenet121_eval")
DEFAULT_OUTPUT_DIR = Path("artifacts/P2A_diagnosis")


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
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _spearman(x: pd.Series, y: pd.Series) -> dict[str, float | int | None]:
    mask = x.notna() & y.notna()
    x_valid = x.loc[mask].to_numpy(dtype=np.float64)
    y_valid = y.loc[mask].to_numpy(dtype=np.float64)
    if len(x_valid) < 3 or np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return {"n": int(len(x_valid)), "rho": None, "p_value": None}
    result = spearmanr(x_valid, y_valid)
    return {
        "n": int(len(x_valid)),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _rank_bottom(frame: pd.DataFrame, column: str, count: int = 5) -> list[dict[str, Any]]:
    return (
        frame.sort_values([column, "class"], ascending=[True, True])
        .head(count)
        .to_dict(orient="records")
    )


def _pattern_flags(test_frame: pd.DataFrame) -> dict[str, Any]:
    """Use dataset-relative cutoffs and persist them with the exploratory flags."""
    thresholds = {
        "auroc_median": float(test_frame["auroc"].median()),
        "auprc_q1": float(test_frame["auprc"].quantile(0.25)),
        "f1_q1": float(test_frame["f1"].quantile(0.25)),
        "precision_median": float(test_frame["precision"].median()),
        "precision_q1": float(test_frame["precision"].quantile(0.25)),
        "recall_median": float(test_frame["recall"].median()),
        "recall_q1": float(test_frame["recall"].quantile(0.25)),
        "prevalence_q1": float(test_frame["train_prevalence"].quantile(0.25)),
    }
    flags = {
        "high_auroc_low_f1": test_frame.loc[
            (test_frame["auroc"] >= thresholds["auroc_median"])
            & (test_frame["f1"] <= thresholds["f1_q1"]),
            "class",
        ].tolist(),
        "low_auroc_auprc_f1": test_frame.loc[
            (test_frame["auprc"] <= thresholds["auprc_q1"])
            & (test_frame["f1"] <= thresholds["f1_q1"])
            & (test_frame["auroc"] <= thresholds["auroc_median"]),
            "class",
        ].tolist(),
        "high_precision_low_recall": test_frame.loc[
            (test_frame["precision"] >= thresholds["precision_median"])
            & (test_frame["recall"] <= thresholds["recall_q1"]),
            "class",
        ].tolist(),
        "high_recall_low_precision": test_frame.loc[
            (test_frame["recall"] >= thresholds["recall_median"])
            & (test_frame["precision"] <= thresholds["precision_q1"]),
            "class",
        ].tolist(),
    }
    rare = set(test_frame.loc[test_frame["train_prevalence"] <= thresholds["prevalence_q1"], "class"])
    bottom_f1_classes = {row["class"] for row in _rank_bottom(test_frame, "f1")}
    flags["lowest_quartile_prevalence"] = sorted(rare)
    flags["low_prevalence_bottom_f1_overlap"] = sorted(rare.intersection(bottom_f1_classes))
    return {"cutoffs": thresholds, "flags": flags}


def _distribution(frame: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    train = frame.loc[frame["split"].eq("train"), labels]
    total = len(train)
    if total == 0:
        raise ValueError("The persisted train split is empty")
    positive = train.sum(axis=0).astype(int)
    result = pd.DataFrame(
        {
            "class": labels,
            "train_positive_count": [int(positive[label]) for label in labels],
        }
    )
    result["train_total_count"] = total
    result["train_negative_count"] = result["train_total_count"] - result["train_positive_count"]
    result["train_prevalence"] = result["train_positive_count"] / result["train_total_count"]
    result["negative_to_positive_ratio"] = (
        result["train_negative_count"] / result["train_positive_count"]
    )
    if not np.isfinite(result["negative_to_positive_ratio"]).all():
        raise ValueError("At least one class has zero positive training samples")
    return result.sort_values(["train_prevalence", "class"], ascending=[True, True]).reset_index(drop=True)


def _diagnostic_frame(
    targets: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    auc_rows = compute_auc_metrics(targets, probabilities, labels)["per_class"]
    f1_rows = compute_per_class_metrics(targets, probabilities, labels, thresholds)
    auc = pd.DataFrame(auc_rows).rename(columns={"label": "class"})
    f1 = pd.DataFrame(f1_rows).rename(columns={"label": "class", "positive_count": "positive_support"})
    f1["prevalence"] = f1["positive_support"] / f1["total_count"]
    columns = [
        "class",
        "positive_support",
        "total_count",
        "prevalence",
        "threshold",
        "precision",
        "recall",
        "f1",
    ]
    return auc.merge(f1[columns], on="class", validate="one_to_one")


def run_diagnosis(
    frozen_dir: str | Path = DEFAULT_FROZEN_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    frozen_dir = Path(frozen_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Diagnosis output directory already exists: {output_dir}")

    protocol, threshold_artifact = load_and_validate_frozen_bundle(frozen_dir)
    labels = list(protocol["label_cols"])
    thresholds = np.asarray(threshold_artifact["thresholds"], dtype=np.float64)
    labels_csv = Path(protocol["labels_csv"]["path"])
    frame = pd.read_csv(labels_csv)
    required = {"split", *labels}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"labels.csv is missing required columns: {missing}")

    distribution = _distribution(frame, labels)
    distribution_by_class = distribution.set_index("class")
    val_targets = np.load(frozen_dir / "validation_targets.npy")
    val_probabilities = np.load(frozen_dir / "validation_probabilities.npy")
    test_dir = frozen_dir / "test"
    test_targets = np.load(test_dir / "test_targets.npy")
    test_probabilities = np.load(test_dir / "test_probabilities.npy")
    val_frame = _diagnostic_frame(val_targets, val_probabilities, labels, thresholds)
    test_frame = _diagnostic_frame(test_targets, test_probabilities, labels, thresholds)
    for split_frame in (val_frame, test_frame):
        split_frame["train_positive_count"] = split_frame["class"].map(distribution_by_class["train_positive_count"])
        split_frame["train_prevalence"] = split_frame["class"].map(distribution_by_class["train_prevalence"])
        split_frame["negative_to_positive_ratio"] = split_frame["class"].map(distribution_by_class["negative_to_positive_ratio"])
        split_frame.sort_values("class", inplace=True)

    val_auc = compute_auc_metrics(val_targets, val_probabilities, labels)
    test_auc = compute_auc_metrics(test_targets, test_probabilities, labels)
    val_fixed = compute_f1_metrics(val_targets, val_probabilities, 0.5)
    val_tuned = compute_f1_metrics(val_targets, val_probabilities, thresholds)
    test_tuned = compute_f1_metrics(test_targets, test_probabilities, thresholds)
    correlations = {
        f"prevalence_vs_{column}": _spearman(test_frame["train_prevalence"], test_frame[column])
        for column in ("auroc", "auprc", "f1", "precision", "recall")
    }
    patterns = _pattern_flags(test_frame)
    rankings = {
        "test_f1_lowest_5": _rank_bottom(test_frame, "f1"),
        "test_auroc_lowest_5": _rank_bottom(test_frame, "auroc"),
        "test_auprc_lowest_5": _rank_bottom(test_frame, "auprc"),
        "test_recall_lowest_5": _rank_bottom(test_frame, "recall"),
        "test_precision_lowest_5": _rank_bottom(test_frame, "precision"),
        "test_threshold_lowest_5": _rank_bottom(test_frame, "threshold"),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)
    val_frame.to_csv(output_dir / "val_per_class_diagnosis.csv", index=False)
    test_frame.to_csv(output_dir / "test_per_class_diagnosis.csv", index=False)
    _write_json(
        output_dir / "correlation_summary.json",
        {
            "analysis": "Spearman rank correlation across 14 labels; descriptive only, not causal.",
            "predictor": "train_prevalence",
            "outcomes": correlations,
        },
    )
    _write_json(
        output_dir / "diagnosis_summary.json",
        {
            "analysis_scope": {
                "checkpoint": protocol["checkpoint"],
                "frozen_dir": str(frozen_dir),
                "threshold_source": threshold_artifact["source"],
                "threshold_fit_split": threshold_artifact["fit_split"],
                "test_used_for_tuning": False,
            },
            "overall": {
                "validation": {
                    "macro_auprc": val_auc["macro_auprc"],
                    "micro_auprc": val_auc["micro_auprc"],
                    "f1_at_0_5": val_fixed,
                    "f1_tuned": val_tuned,
                },
                "test": {
                    "macro_auprc": test_auc["macro_auprc"],
                    "micro_auprc": test_auc["micro_auprc"],
                    "f1_tuned": test_tuned,
                },
            },
            "rankings": rankings,
            "patterns": patterns,
            "correlations": correlations,
            "rarest_5": distribution.head(5).to_dict(orient="records"),
            "most_common_5": distribution.tail(5).sort_values(
                ["train_prevalence", "class"], ascending=[False, True]
            ).to_dict(orient="records"),
        },
    )
    return {
        "output_dir": str(output_dir),
        "validation": {"macro_auprc": val_auc["macro_auprc"], "micro_auprc": val_auc["micro_auprc"]},
        "test": {"macro_auprc": test_auc["macro_auprc"], "micro_auprc": test_auc["micro_auprc"]},
        "correlations": correlations,
        "rankings": rankings,
        "patterns": patterns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose B1 class imbalance and per-class errors.")
    parser.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    print(json.dumps(_json_ready(run_diagnosis(args.frozen_dir, args.output_dir)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
