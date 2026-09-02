"""Compare completed B2 and B3 artifacts without selecting on test alone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from workflow_state import atomic_write_csv, atomic_write_json, update_stage


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def compare_b2_b3(
    b2_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight",
    b3_dir: str | Path = "outputs/B3_densenet121_asl",
    output_dir: str | Path = "outputs/B2_vs_B3",
    workflow_state_path: str | Path = "workflow_state.json",
) -> dict[str, Any]:
    b2_dir, b3_dir, output_dir = map(Path, (b2_dir, b3_dir, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"comparison output already exists: {output_dir}")
    b2_val = _read_json(b2_dir / "thresholds" / "validation_metrics.json")
    b3_val = _read_json(b3_dir / "thresholds" / "validation_metrics.json")
    b2_test = _read_json(b2_dir / "test" / "B2_test_tuned.json")
    b3_test = _read_json(b3_dir / "test" / "B3_test_tuned.json")
    rows = []
    for metric in ("macro_AUPRC", "macro_AUROC", "macro_F1_tuned", "micro_F1_tuned"):
        b2_value = b2_val.get(metric, b2_test.get(metric, b2_test.get("tuned", {}).get(metric)))
        b3_value = b3_val.get(metric, b3_test.get(metric, b3_test.get("tuned", {}).get(metric)))
        rows.append({"metric": metric, "B2": b2_value, "B3": b3_value, "delta_B3_minus_B2": None if b2_value is None or b3_value is None else float(b3_value) - float(b2_value)})
    b2_auprc = float(b2_val.get("macro_AUPRC", float("nan")))
    b3_auprc = float(b3_val.get("macro_AUPRC", float("nan")))
    b2_auroc = float(b2_val.get("macro_AUROC", float("nan")))
    b3_auroc = float(b3_val.get("macro_AUROC", float("nan")))
    winner = "B3" if b3_auprc > b2_auprc and b3_auroc >= b2_auroc else "B2" if b2_auprc > b3_auprc and b2_auroc >= b3_auroc else "TIE / INCONCLUSIVE"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_dir / "b2_vs_b3_summary.csv", rows[0].keys(), rows)
    b2_per_label = pd.read_csv(b2_dir / "thresholds" / "threshold_tuning_val.csv")
    b3_per_label = pd.read_csv(b3_dir / "thresholds" / "threshold_tuning_val.csv")
    merged = b2_per_label.merge(b3_per_label, on=["label_index", "label_name"], suffixes=("_B2", "_B3"))
    per_label_rows = []
    for _, row in merged.iterrows():
        per_label_rows.append({
            "label_index": int(row["label_index"]), "label_name": row["label_name"],
            "auroc_B2": row.get("auroc_B2"), "auroc_B3": row.get("auroc_B3"),
            "auprc_B2": row.get("auprc_B2"), "auprc_B3": row.get("auprc_B3"),
            "f1_tuned_B2": row.get("f1_tuned_B2"), "f1_tuned_B3": row.get("f1_tuned_B3"),
        })
    atomic_write_csv(output_dir / "b2_vs_b3_per_label.csv", per_label_rows[0].keys(), per_label_rows)
    report = {"winner": winner, "priority": ["macro AUPRC", "macro AUROC", "tuned macro F1", "rare-label performance", "stability"], "validation": {"B2": b2_val, "B3": b3_val}, "test": {"B2": b2_test, "B3": b3_test}, "test_used_for_selection": False}
    atomic_write_json(output_dir / "b2_vs_b3_report.json", report)
    (output_dir / "b2_vs_b3_report.md").write_text(f"# B2 vs B3\n\nWinner: **{winner}**\n\nSelection prioritizes validation macro AUPRC, then validation macro AUROC. Test metrics are reported after threshold freeze and are not used for selection.\n", encoding="utf-8")
    update_stage(workflow_state_path, "B2_vs_B3", "completed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare B2 and B3 artifacts.")
    parser.add_argument("--b2-dir", default="outputs/B2_densenet121_sqrt_posweight")
    parser.add_argument("--b3-dir", default="outputs/B3_densenet121_asl")
    parser.add_argument("--output-dir", default="outputs/B2_vs_B3")
    parser.add_argument("--workflow-state", default="workflow_state.json")
    args = parser.parse_args()
    print(json.dumps(compare_b2_b3(args.b2_dir, args.b3_dir, args.output_dir, args.workflow_state), indent=2))


if __name__ == "__main__":
    main()
