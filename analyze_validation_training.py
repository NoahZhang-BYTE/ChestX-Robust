"""Analyze a completed B3 or B4 run using validation metrics only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from analyze_b2_training import _checkpoint_inventory, _finite, _plot_curves
from workflow_state import atomic_write_csv, atomic_write_json, update_stage


REQUIRED_COLUMNS = ("train_loss", "val_loss", "macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc", "macro_f1")


def _row_for_epoch(history: pd.DataFrame, epoch: int) -> pd.Series:
    rows = history.loc[history["epoch"].eq(epoch)]
    if len(rows) != 1:
        raise ValueError(f"history must contain exactly one row for epoch {epoch}")
    return rows.iloc[0]


def _matching_checkpoint(inventory: Sequence[dict[str, Any]], filename: str, epoch: int, metric: str, value: float) -> dict[str, Any]:
    candidate = next((row for row in inventory if row.get("filename") == filename), None)
    if candidate is None or not candidate.get("loadable"):
        raise ValueError(f"Required loadable checkpoint is missing: {filename}")
    if int(candidate["epoch"]) != epoch:
        raise ValueError(f"{filename} epoch does not match best validation {metric} epoch")
    checkpoint_value = candidate.get(metric)
    if checkpoint_value is None or not math.isclose(float(checkpoint_value), value, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{filename} {metric} does not match history")
    return candidate


def select_validation_checkpoints(history: pd.DataFrame, inventory: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select only from validation macro AUPRC first, then macro AUROC."""
    for column in ("epoch", "macro_auprc", "macro_auroc"):
        if column not in history:
            raise ValueError(f"history is missing {column}")
    auprc_row = history.loc[history["macro_auprc"].idxmax()]
    auroc_row = history.loc[history["macro_auroc"].idxmax()]
    primary = _matching_checkpoint(
        inventory,
        "best_macro_auprc.pt",
        int(auprc_row["epoch"]),
        "macro_auprc",
        float(auprc_row["macro_auprc"]),
    )
    secondary = _matching_checkpoint(
        inventory,
        "best_macro_auroc.pt",
        int(auroc_row["epoch"]),
        "macro_auroc",
        float(auroc_row["macro_auroc"]),
    )
    return {
        "selected_filename": primary["filename"],
        "selected_epoch": int(primary["epoch"]),
        "selection_reason": "best validation macro AUPRC; macro AUROC is the secondary criterion",
        "macro_auprc": float(primary["macro_auprc"]),
        "macro_auroc": float(primary["macro_auroc"]),
        "secondary_filename": secondary["filename"],
        "secondary_epoch": int(secondary["epoch"]),
        "secondary_macro_auprc": float(secondary["macro_auprc"]),
        "secondary_macro_auroc": float(secondary["macro_auroc"]),
        "test_used_for_selection": False,
    }


def analyze_validation_training(
    output_dir: str | Path,
    experiment: str,
    stage_name: str,
    workflow_state_path: str | Path = "workflow_state.json",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    history_path = output_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"history.csv does not exist: {history_path}")
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError("history.csv is empty")
    for column in REQUIRED_COLUMNS:
        if column not in history or not all(_finite(value) for value in history[column]):
            raise ValueError(f"history has missing, NaN, or Inf values in {column}")
    epochs = history["epoch"].astype(int).tolist()
    if epochs != list(range(1, max(epochs) + 1)):
        raise ValueError("history epochs must be consecutive and start at 1")
    inventory = _checkpoint_inventory(output_dir)
    selection = select_validation_checkpoints(history, inventory)
    post_dir = output_dir / "post_analysis"
    post_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(post_dir / "checkpoint_inventory.csv", inventory[0].keys(), inventory)
    rows = []
    for _, row in history.iterrows():
        item = {"epoch": int(row["epoch"])}
        for column in REQUIRED_COLUMNS:
            item[column] = float(row[column])
        item["train_val_loss_gap"] = item["val_loss"] - item["train_loss"]
        rows.append(item)
    atomic_write_csv(post_dir / "epoch_summary.csv", rows[0].keys(), rows)
    best_auprc = history.loc[history["macro_auprc"].idxmax()]
    best_auroc = history.loc[history["macro_auroc"].idxmax()]
    minimum_loss = history.loc[history["val_loss"].idxmin()]
    _plot_curves(
        history,
        post_dir / "training_curves.png",
        {
            "best_macro_auroc": int(best_auroc["epoch"]),
            "best_macro_auprc": int(best_auprc["epoch"]),
            "minimum_val_loss": int(minimum_loss["epoch"]),
            "epoch_10_degradation": int(history["epoch"].max()),
        },
    )
    final = history.iloc[-1]
    summary = {
        "experiment": experiment,
        "completed_epoch": int(final["epoch"]),
        "best_epoch_macro_auprc": int(best_auprc["epoch"]),
        "best_macro_auprc": float(best_auprc["macro_auprc"]),
        "best_epoch_macro_auroc": int(best_auroc["epoch"]),
        "best_macro_auroc": float(best_auroc["macro_auroc"]),
        "minimum_val_loss_epoch": int(minimum_loss["epoch"]),
        "minimum_val_loss": float(minimum_loss["val_loss"]),
        "final_train_val_loss_gap": float(final["val_loss"] - final["train_loss"]),
        "selection_metric": "validation macro AUPRC",
        "test_used": False,
    }
    atomic_write_json(post_dir / "analysis_summary.json", summary)
    selection["selected_checkpoint"] = str((output_dir / selection["selected_filename"]).resolve())
    atomic_write_json(post_dir / "checkpoint_selection.json", selection)
    update_stage(workflow_state_path, stage_name, "completed")
    return {"summary": summary, "selection": selection}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed B3/B4 validation history.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--workflow-state", default="workflow_state.json")
    args = parser.parse_args()
    print(json.dumps(analyze_validation_training(args.output_dir, args.experiment, args.stage_name, args.workflow_state), indent=2))


if __name__ == "__main__":
    main()
