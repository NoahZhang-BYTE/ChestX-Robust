import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_cooccurrence import analyze_cooccurrence
from tune_b2_thresholds import _best_threshold


def test_b2_threshold_grid_tie_breaks_toward_half():
    threshold, score = _best_threshold(
        np.array([1, 1, 0, 0]), np.array([0.4, 0.6, 0.4, 0.6])
    )

    assert threshold == 0.4
    assert score == 2 / 3


def test_cooccurrence_writes_named_matrix_and_summary(tmp_path: Path):
    rows = []
    for split, first, second in (("train", 1, 1), ("train", 1, 0), ("val", 0, 1)):
        rows.append({"split": split, "Atelectasis": first, "Cardiomegaly": second, **{name: 0 for name in (
            "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        )}})
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"current_stage": "cooccurrence_analysis", "stages": {stage: "pending" for stage in (
        "B2_training", "B2_analysis", "B2_threshold_tuning", "B2_test", "hard_label_analysis", "cooccurrence_analysis", "B3_training", "B3_analysis", "B3_threshold_tuning", "B3_test", "B2_vs_B3", "B4_training", "B4_analysis", "B5_confirmation"
    )}}), encoding="utf-8")
    output_dir = tmp_path / "cooccurrence"

    analyze_cooccurrence(csv_path, output_dir, state_path)

    matrix = pd.read_csv(output_dir / "label_cooccurrence.csv")
    assert matrix.loc[0, "Atelectasis"] == 2
    assert matrix.loc[0, "Cardiomegaly"] == 1
    assert (output_dir / "label_cooccurrence.png").is_file()
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["Infiltration"][0][1] == 0.0
