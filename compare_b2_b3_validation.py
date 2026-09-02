"""Create the prespecified B2/B3 decision from validation artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from night_workflow import choose_b4_winner
from workflow_state import atomic_write_csv, atomic_write_json, update_stage


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _draw_comparison(path: Path, summary: list[dict[str, float]]) -> None:
    width, height, margin = 900, 360, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    keys = ("macro_AUPRC", "macro_AUROC", "macro_F1_tuned")
    for index, key in enumerate(keys):
        values = next(row for row in summary if row["metric"] == key)
        x = margin + index * 270
        draw.text((x, 35), key, fill="black")
        for offset, name, color in ((0, "B2", "#1565c0"), (70, "B3", "#c62828")):
            value = float(values[name])
            top = 300 - int(value * 240)
            draw.rectangle((x + offset, top, x + offset + 45, 300), fill=color)
            draw.text((x + offset, 305), name, fill="black")
            draw.text((x + offset, top - 20), f"{value:.3f}", fill="black")
        draw.line((x - 10, 300, x + 130, 300), fill="black")
    image.save(path, format="PNG")


def compare_b2_b3_validation(
    b2_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight",
    b3_dir: str | Path = "outputs/B3_densenet121_asl",
    output_dir: str | Path = "outputs/comparisons/B2_vs_B3_validation",
    workflow_state_path: str | Path = "workflow_state.json",
) -> dict:
    b2_dir, b3_dir, output_dir = map(Path, (b2_dir, b3_dir, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"comparison output already exists: {output_dir}")
    b2_metrics = _read_json(b2_dir / "thresholds" / "validation_metrics.json")
    b3_metrics = _read_json(b3_dir / "thresholds" / "validation_metrics.json")
    b2_targets = np.load(b2_dir / "thresholds" / "y_true_val.npy")
    b3_targets = np.load(b3_dir / "thresholds" / "y_true_val.npy")
    if not np.array_equal(b2_targets, b3_targets):
        raise ValueError("B2/B3 validation targets differ; comparison is not valid")
    b2_per = pd.read_csv(b2_dir / "thresholds" / "threshold_tuning_val.csv")
    b3_per = pd.read_csv(b3_dir / "thresholds" / "threshold_tuning_val.csv")
    # B2 artifacts predate the B3 naming convention (f1_tuned vs tuned_f1).
    # Normalize the legacy column before the one-to-one merge; metrics remain
    # unchanged and both files are still validation-only artifacts.
    if "tuned_f1" not in b2_per.columns and "f1_tuned" in b2_per.columns:
        b2_per = b2_per.rename(columns={"f1_tuned": "tuned_f1"})
    if "tuned_f1" not in b3_per.columns and "f1_tuned" in b3_per.columns:
        b3_per = b3_per.rename(columns={"f1_tuned": "tuned_f1"})
    merged = b2_per.merge(b3_per, on=["label_index", "label_name"], suffixes=("_B2", "_B3"), validate="one_to_one")
    b2_label = {row.label_name: {"auroc": float(row.auroc_B2), "auprc": float(row.auprc_B2), "tuned_f1": float(row.tuned_f1_B2)} for row in merged.itertuples()}
    b3_label = {row.label_name: {"auroc": float(row.auroc_B3), "auprc": float(row.auprc_B3), "tuned_f1": float(row.tuned_f1_B3)} for row in merged.itertuples()}
    decision = choose_b4_winner(b2_metrics, b3_metrics, b2_label, b3_label)
    summary_rows = []
    for metric in ("macro_AUPRC", "macro_AUROC", "micro_AUPRC", "micro_AUROC", "macro_F1_tuned", "micro_F1_tuned", "sample_F1_tuned"):
        summary_rows.append({"metric": metric, "B2": float(b2_metrics[metric]), "B3": float(b3_metrics[metric]), "delta_B3_minus_B2": float(b3_metrics[metric]) - float(b2_metrics[metric])})
    per_label_rows = []
    for row in merged.itertuples():
        per_label_rows.append({
            "label_index": int(row.label_index), "label_name": row.label_name,
            "B2_AUROC": float(row.auroc_B2), "B3_AUROC": float(row.auroc_B3), "delta_AUROC": float(row.auroc_B3 - row.auroc_B2),
            "B2_AUPRC": float(row.auprc_B2), "B3_AUPRC": float(row.auprc_B3), "delta_AUPRC": float(row.auprc_B3 - row.auprc_B2),
            "B2_tuned_F1": float(row.tuned_f1_B2), "B3_tuned_F1": float(row.tuned_f1_B3), "delta_tuned_F1": float(row.tuned_f1_B3 - row.tuned_f1_B2),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "summary.csv", summary_rows[0].keys(), summary_rows)
    atomic_write_csv(output_dir / "per_label.csv", per_label_rows[0].keys(), per_label_rows)
    _draw_comparison(output_dir / "comparison.png", summary_rows)
    decision.update({
        "comparison_split": "val",
        "same_validation_targets_verified": True,
        "selection_priority": ["macro AUPRC", "macro AUROC", "tuned macro F1", "Pneumonia/Infiltration/Hernia"],
        "Infiltration_ASL_response": {
            "B2_AUROC": b2_label["Infiltration"]["auroc"], "B3_AUROC": b3_label["Infiltration"]["auroc"],
            "B2_AUPRC": b2_label["Infiltration"]["auprc"], "B3_AUPRC": b3_label["Infiltration"]["auprc"],
        },
        "Hernia": {"B2_AUPRC": b2_label["Hernia"]["auprc"], "B3_AUPRC": b3_label["Hernia"]["auprc"]},
    })
    atomic_write_json(output_dir / "decision.json", decision)
    report = (
        "# B2 vs B3 Validation\n\n"
        f"Winner: **{decision['winner']}**\n\n"
        f"Rule: `{decision['rule']}`. The decision uses validation only; test was not read.\n\n"
        f"B3-B2 macro AUPRC: {decision['delta_B3_minus_B2']['macro_AUPRC']:.6f}; "
        f"macro AUROC: {decision['delta_B3_minus_B2']['macro_AUROC']:.6f}; "
        f"tuned macro F1: {decision['delta_B3_minus_B2']['macro_F1_tuned']:.6f}.\n"
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    update_stage(workflow_state_path, "B2_vs_B3", "completed")
    update_stage(workflow_state_path, "B2_vs_B3_validation", "completed")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B2/B3 from validation-only artifacts.")
    parser.add_argument("--b2-dir", default="outputs/B2_densenet121_sqrt_posweight")
    parser.add_argument("--b3-dir", default="outputs/B3_densenet121_asl")
    parser.add_argument("--output-dir", default="outputs/comparisons/B2_vs_B3_validation")
    parser.add_argument("--workflow-state", default="workflow_state.json")
    args = parser.parse_args()
    print(json.dumps(compare_b2_b3_validation(args.b2_dir, args.b3_dir, args.output_dir, args.workflow_state), indent=2))


if __name__ == "__main__":
    main()
