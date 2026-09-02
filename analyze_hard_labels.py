"""Create non-destructive error tables for the predefined difficult labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.labels import LABEL_COLUMNS
from workflow_state import atomic_write_csv, atomic_write_json, update_stage


def analyze_hard_labels(
    labels_csv: str | Path = "data/labels.csv",
    arrays_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight/test",
    thresholds_path: str | Path = "outputs/B2_densenet121_sqrt_posweight/thresholds/thresholds.json",
    output_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight/post_analysis/hard_labels",
    workflow_state_path: str | Path = "workflow_state.json",
    stage_name: str = "hard_label_analysis",
) -> dict[str, int]:
    frame = pd.read_csv(labels_csv)
    y_true = np.load(Path(arrays_dir) / "y_true_test.npy")
    y_prob = np.load(Path(arrays_dir) / "y_prob_test.npy")
    thresholds_path = Path(thresholds_path)
    artifact = json.loads(thresholds_path.read_text(encoding="utf-8"))
    freeze_path = thresholds_path.with_name("threshold_freeze.json")
    if not freeze_path.is_file():
        raise FileNotFoundError(f"Threshold freeze manifest does not exist: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("fit_split") != "val":
        raise ValueError("Hard-label analysis requires validation-only frozen thresholds")
    if freeze.get("sha256") != hashlib.sha256(thresholds_path.read_bytes()).hexdigest():
        raise ValueError("Frozen threshold SHA256 does not match its manifest")
    thresholds = np.asarray(artifact["thresholds"], dtype=np.float64)
    subset = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    if len(subset) != len(y_true):
        raise ValueError("test labels and prediction arrays have different lengths")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    expected_targets = subset.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.uint8)
    if y_true.shape != expected_targets.shape or not np.array_equal(y_true, expected_targets):
        raise ValueError("test labels do not align with the saved prediction arrays")
    for index, label in ((3, "Infiltration"), (6, "Pneumonia"), (13, "Hernia")):
        prediction = y_prob[:, index] >= thresholds[index]
        rows = []
        for sample_index in range(len(y_true)):
            coexisting = [name for other, name in enumerate(LABEL_COLUMNS) if other != index and y_true[sample_index, other]]
            error_type = (
                "TP" if y_true[sample_index, index] and prediction[sample_index]
                else "FP" if prediction[sample_index]
                else "FN" if y_true[sample_index, index]
                else "TN"
            )
            rows.append({
                "image_path": str(subset.iloc[sample_index]["path"]),
                "probability": float(y_prob[sample_index, index]),
                "ground_truth": int(y_true[sample_index, index]),
                "predicted": int(prediction[sample_index]),
                "threshold": float(thresholds[index]),
                "coexisting_labels": json.dumps(coexisting),
                "error_type": error_type,
            })
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(label_dir / "all_samples.csv", rows[0].keys(), rows)
        for error_type in ("TP", "FP", "FN", "TN"):
            subset_rows = [row for row in rows if row["error_type"] == error_type]
            atomic_write_csv(label_dir / f"{error_type}.csv", rows[0].keys(), subset_rows)
        counts[label] = len(rows)
    atomic_write_json(output_dir / "summary.json", {"labels": counts, "threshold_source": "validation"})
    update_stage(workflow_state_path, stage_name, "completed")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Infiltration and Pneumonia errors.")
    parser.add_argument("--labels-csv", default="data/labels.csv")
    parser.add_argument("--arrays-dir", default="outputs/B2_densenet121_sqrt_posweight/test")
    parser.add_argument("--thresholds", default="outputs/B2_densenet121_sqrt_posweight/thresholds/thresholds_frozen_for_test.json")
    parser.add_argument("--output-dir", default="outputs/B2_densenet121_sqrt_posweight/post_analysis/hard_labels")
    parser.add_argument("--workflow-state", default="workflow_state.json")
    parser.add_argument("--stage-name", default="hard_label_analysis")
    args = parser.parse_args()
    print(json.dumps(analyze_hard_labels(args.labels_csv, args.arrays_dir, args.thresholds, args.output_dir, args.workflow_state, args.stage_name), indent=2))


if __name__ == "__main__":
    main()
