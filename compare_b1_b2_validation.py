"""Compare B1 and B2 on their saved validation predictions under one grid protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from baseline.labels import LABEL_COLUMNS
from tune_b2_thresholds import _best_threshold
from workflow_state import atomic_write_csv, atomic_write_json


def _load_pair(targets_path: Path, probabilities_path: Path) -> tuple[np.ndarray, np.ndarray]:
    targets = np.load(targets_path).astype(np.uint8)
    probabilities = np.load(probabilities_path).astype(np.float32)
    if targets.ndim != 2 or targets.shape != probabilities.shape or targets.shape[1] != len(LABEL_COLUMNS):
        raise ValueError(f"Invalid saved validation arrays: {targets_path}, {probabilities_path}")
    return targets, probabilities


def _evaluate(name: str, targets: np.ndarray, probabilities: np.ndarray) -> tuple[dict, list[dict]]:
    thresholds = np.asarray([_best_threshold(targets[:, index], probabilities[:, index])[0] for index in range(targets.shape[1])])
    fixed_prediction = probabilities >= 0.5
    tuned_prediction = probabilities >= thresholds.reshape(1, -1)
    rows: list[dict] = []
    for index, label in enumerate(LABEL_COLUMNS):
        target = targets[:, index]
        probability = probabilities[:, index]
        rows.append({
            "model": name,
            "label_index": index,
            "label_name": label,
            "threshold": float(thresholds[index]),
            "auroc": float(roc_auc_score(target, probability)),
            "auprc": float(average_precision_score(target, probability)),
            "fixed05_precision": float(precision_score(target, fixed_prediction[:, index], zero_division=0)),
            "fixed05_recall": float(recall_score(target, fixed_prediction[:, index], zero_division=0)),
            "fixed05_f1": float(f1_score(target, fixed_prediction[:, index], zero_division=0)),
            "tuned_precision": float(precision_score(target, tuned_prediction[:, index], zero_division=0)),
            "tuned_recall": float(recall_score(target, tuned_prediction[:, index], zero_division=0)),
            "tuned_f1": float(f1_score(target, tuned_prediction[:, index], zero_division=0)),
        })
    summary = {
        "model": name,
        "sample_count": int(targets.shape[0]),
        "macro_auroc": float(np.mean([row["auroc"] for row in rows])),
        "micro_auroc": float(roc_auc_score(targets.ravel(), probabilities.ravel())),
        "macro_auprc": float(np.mean([row["auprc"] for row in rows])),
        "micro_auprc": float(average_precision_score(targets.ravel(), probabilities.ravel())),
        "macro_f1_fixed05": float(f1_score(targets, fixed_prediction, average="macro", zero_division=0)),
        "micro_f1_fixed05": float(f1_score(targets, fixed_prediction, average="micro", zero_division=0)),
        "sample_f1_fixed05": float(f1_score(targets, fixed_prediction, average="samples", zero_division=0)),
        "macro_f1_tuned": float(f1_score(targets, tuned_prediction, average="macro", zero_division=0)),
        "micro_f1_tuned": float(f1_score(targets, tuned_prediction, average="micro", zero_division=0)),
        "sample_f1_tuned": float(f1_score(targets, tuned_prediction, average="samples", zero_division=0)),
    }
    return summary, rows


def compare_b1_b2_validation(
    b1_dir: str | Path = "artifacts/B1_densenet121_eval",
    b2_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight/thresholds",
    output_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight/post_analysis",
) -> dict:
    b1_dir, b2_dir, output_dir = map(Path, (b1_dir, b2_dir, output_dir))
    b1_targets, b1_probabilities = _load_pair(b1_dir / "validation_targets.npy", b1_dir / "validation_probabilities.npy")
    b2_targets, b2_probabilities = _load_pair(b2_dir / "y_true_val.npy", b2_dir / "y_prob_val.npy")
    if not np.array_equal(b1_targets, b2_targets):
        raise ValueError("B1 and B2 validation targets differ; comparison would not be fair")
    b1_summary, b1_rows = _evaluate("B1_densenet121_bce", b1_targets, b1_probabilities)
    b2_summary, b2_rows = _evaluate("B2_densenet121_sqrt_posweight", b2_targets, b2_probabilities)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [b1_summary, b2_summary]
    by_label = {(row["model"], row["label_name"]): row for row in [*b1_rows, *b2_rows]}
    label_comparison = []
    for index, label in enumerate(LABEL_COLUMNS):
        b1_row = by_label[("B1_densenet121_bce", label)]
        b2_row = by_label[("B2_densenet121_sqrt_posweight", label)]
        label_comparison.append({
            "label_index": index,
            "label_name": label,
            "b1_auroc": b1_row["auroc"], "b2_auroc": b2_row["auroc"], "delta_auroc": b2_row["auroc"] - b1_row["auroc"],
            "b1_auprc": b1_row["auprc"], "b2_auprc": b2_row["auprc"], "delta_auprc": b2_row["auprc"] - b1_row["auprc"],
            "b1_threshold": b1_row["threshold"], "b2_threshold": b2_row["threshold"],
            "b1_tuned_f1": b1_row["tuned_f1"], "b2_tuned_f1": b2_row["tuned_f1"], "delta_tuned_f1": b2_row["tuned_f1"] - b1_row["tuned_f1"],
            "b1_tuned_precision": b1_row["tuned_precision"], "b2_tuned_precision": b2_row["tuned_precision"],
            "b1_tuned_recall": b1_row["tuned_recall"], "b2_tuned_recall": b2_row["tuned_recall"],
        })
    rare_names = {"Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema"}
    atomic_write_csv(output_dir / "B1_vs_B2_validation_summary.csv", rows[0].keys(), rows)
    atomic_write_csv(output_dir / "B1_vs_B2_validation_per_label.csv", label_comparison[0].keys(), label_comparison)
    atomic_write_csv(output_dir / "B1_vs_B2_rare5_validation.csv", label_comparison[0].keys(), [row for row in label_comparison if row["label_name"] in rare_names])
    result = {
        "split": "val",
        "same_targets_verified": True,
        "test_used": False,
        "threshold_protocol": "per-label F1 grid 0.01..0.99; ties choose threshold nearest 0.5",
        "B1": b1_summary,
        "B2": b2_summary,
        "B2_minus_B1": {key: b2_summary[key] - b1_summary[key] for key in b1_summary if key.startswith(("macro_", "micro_", "sample_"))},
    }
    atomic_write_json(output_dir / "B1_vs_B2_validation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved B1/B2 validation arrays without test data.")
    parser.add_argument("--b1-dir", default="artifacts/B1_densenet121_eval")
    parser.add_argument("--b2-dir", default="outputs/B2_densenet121_sqrt_posweight/thresholds")
    parser.add_argument("--output-dir", default="outputs/B2_densenet121_sqrt_posweight/post_analysis")
    args = parser.parse_args()
    print(json.dumps(compare_b1_b2_validation(args.b1_dir, args.b2_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
