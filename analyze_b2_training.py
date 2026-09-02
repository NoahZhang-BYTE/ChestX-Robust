"""Validate and summarize a completed B2 training run without changing it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

from baseline.engine import load_training_checkpoint
from baseline.labels import LABEL_COLUMNS
from workflow_state import atomic_write_json, atomic_write_csv, update_stage


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _plot_curves(history: pd.DataFrame, path: Path, markers: dict[str, int]) -> None:
    width, height, margin = 1200, 800, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    panels = [
        ("train_loss", "val_loss", "Loss"),
        ("macro_auroc", "macro_auprc", "AUROC / AUPRC"),
        ("macro_f1", "micro_auroc", "F1 / micro AUROC"),
    ]
    epochs = history["epoch"].tolist()
    for index, (first, second, title) in enumerate(panels):
        top = margin + index * 240
        bottom = top + 175
        left, right = margin, width - margin
        values = []
        for column in (first, second):
            if column in history:
                values.extend([float(value) for value in history[column] if _finite(value)])
        if not values:
            continue
        low, high = min(values), max(values)
        span = max(high - low, 1e-9)
        draw.text((left, top - 25), title, fill="black")
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        for column, color in ((first, "#1565c0"), (second, "#c62828")):
            if column not in history:
                continue
            points = []
            for epoch, value in zip(epochs, history[column]):
                if not _finite(value):
                    continue
                x = left + (float(epoch) - min(epochs)) / max(max(epochs) - min(epochs), 1) * (right - left)
                y = bottom - (float(value) - low) / span * (bottom - top)
                points.append((int(x), int(y)))
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for point in points:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        for marker_name, epoch in markers.items():
            x = left + (float(epoch) - min(epochs)) / max(max(epochs) - min(epochs), 1) * (right - left)
            color = {"best_macro_auroc": "#2e7d32", "best_macro_auprc": "#6a1b9a", "minimum_val_loss": "#ef6c00", "epoch_10_degradation": "#c62828"}[marker_name]
            draw.line((int(x), top, int(x), bottom), fill=color, width=1)
        if index == 0:
            draw.text((right - 380, top - 25), "green=best AUROC purple=best AUPRC orange=min val loss red=epoch 10", fill="black")
    image.save(path, format="PNG")


def _checkpoint_inventory(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted([*output_dir.glob("*.pt"), *output_dir.glob("*.pth")]):
        row: dict[str, Any] = {
            "filename": path.name,
            "epoch": None,
            "macro_auroc": None,
            "macro_auprc": None,
            "best_metric": None,
            "file_size": path.stat().st_size,
            "loadable": False,
            "notes": "",
        }
        try:
            checkpoint = load_training_checkpoint(path)
            metrics = checkpoint.get("metrics", {})
            row.update(
                epoch=int(checkpoint["epoch"]),
                macro_auroc=metrics.get("macro_auroc"),
                macro_auprc=metrics.get("macro_auprc"),
                best_metric=checkpoint.get("best_metric"),
                loadable=True,
                notes="metadata loaded read-only",
            )
        except Exception as error:  # Inventory must record, not hide, unreadable files.
            row["notes"] = f"load failed: {type(error).__name__}: {error}"
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No .pt or .pth checkpoints found in {output_dir}")
    return rows


def _write_immutable_text(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Existing immutable artifact differs: {path}")
        return
    path.write_text(text, encoding="utf-8", newline="\n")


def _freeze_b2_run(output_dir: Path, config_source: Path) -> None:
    complete_text = (
        "experiment=B2_densenet121_sqrt_posweight\n"
        "completed_epochs=10\n"
        "auditable_history_epochs=3-10\n"
        "status=completed\n"
    )
    _write_immutable_text(output_dir / "B2_COMPLETE.txt", complete_text)
    if not config_source.is_file():
        raise FileNotFoundError(f"B2 source config does not exist: {config_source}")
    snapshot = output_dir / "config_snapshot.yaml"
    source_bytes = config_source.read_bytes()
    if snapshot.exists():
        if snapshot.read_bytes() != source_bytes:
            raise RuntimeError(f"Existing immutable config snapshot differs: {snapshot}")
    else:
        snapshot.write_bytes(source_bytes)


def analyze_b2(
    output_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight",
    config_source: str | Path = "configs/formal_b2_densenet121_sqrt_posweight.yaml",
    workflow_state_path: str | Path = "workflow_state.json",
    experiment_name: str = "B2",
    stage_name: str = "B2_analysis",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    config_source = Path(config_source)
    history_path = output_dir / "history.csv"
    checkpoint_path = output_dir / "last.pt"
    if not history_path.is_file():
        raise FileNotFoundError(f"B2 history.csv does not exist: {history_path}")
    checkpoint = load_training_checkpoint(checkpoint_path)
    history = pd.read_csv(history_path)
    expected_epochs = list(range(3, 11))
    if history.empty or history["epoch"].astype(int).tolist() != expected_epochs:
        raise RuntimeError("B2 analysis requires a completed epoch 10 history")
    numeric_columns = [
        "train_loss", "val_loss", "macro_auroc", "macro_auprc", "macro_f1", "micro_auroc"
    ]
    if any(column not in history for column in numeric_columns):
        raise RuntimeError(f"B2 history is missing required metrics: {numeric_columns}")
    if not all(_finite(value) for column in numeric_columns for value in history[column]):
        raise RuntimeError("B2 history contains NaN or Inf")
    if int(checkpoint["epoch"]) != 10:
        raise RuntimeError("B2 last checkpoint does not contain epoch 10")

    _freeze_b2_run(output_dir, config_source)
    inventory = _checkpoint_inventory(output_dir)
    atomic_write_csv(output_dir / "checkpoint_inventory.csv", inventory[0].keys(), inventory)
    selected = next((row for row in inventory if row["filename"] == "best.pt"), None)
    if selected is None or not selected["loadable"] or selected["epoch"] != 9:
        raise RuntimeError("[B2 CHECKPOINT WARNING] No loadable epoch 9 best.pt is available; stopping before validation/test")
    if not math.isclose(float(selected["macro_auroc"]), float(history.loc[history["epoch"].eq(9), "macro_auroc"].iloc[0]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("best.pt epoch 9 macro AUROC does not match auditable history")
    if not math.isclose(float(selected["macro_auprc"]), float(history.loc[history["epoch"].eq(9), "macro_auprc"].iloc[0]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("best.pt epoch 9 macro AUPRC does not match auditable history")

    update_stage(workflow_state_path, "B2_training", "completed")
    update_stage(workflow_state_path, stage_name, "running")
    stage = output_dir / "post_analysis"
    stage.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in history.iterrows():
        item = {"epoch": int(row["epoch"])}
        for column in numeric_columns:
            item[column] = float(row[column])
        item["train_val_loss_gap"] = item["val_loss"] - item["train_loss"]
        rows.append(item)
    atomic_write_csv(stage / "epoch_summary.csv", rows[0].keys(), rows)
    best_auroc_row = history.loc[history["macro_auroc"].idxmax()]
    best_auprc_row = history.loc[history["macro_auprc"].idxmax()]
    min_loss_row = history.loc[history["val_loss"].idxmin()]
    _plot_curves(
        history,
        stage / "training_curves.png",
        {
            "best_macro_auroc": int(best_auroc_row["epoch"]),
            "best_macro_auprc": int(best_auprc_row["epoch"]),
            "minimum_val_loss": int(min_loss_row["epoch"]),
            "epoch_10_degradation": 10,
        },
    )

    best = {}
    for metric in ("macro_auroc", "macro_auprc", "macro_f1"):
        best_row = history.loc[history[metric].idxmax()]
        best[f"best_epoch_{metric if metric != 'macro_f1' else 'macro_f1_fixed05'}"] = int(best_row["epoch"])
        best[f"best_{metric if metric != 'macro_f1' else 'macro_f1_fixed05'}"] = float(best_row[metric])
    last_three = history.tail(3)
    auroc_delta = float(last_three["macro_auroc"].iloc[-1] - last_three["macro_auroc"].iloc[0])
    auprc_delta = float(last_three["macro_auprc"].iloc[-1] - last_three["macro_auprc"].iloc[0])
    loss_gap = float(history["val_loss"].iloc[-1] - history["train_loss"].iloc[-1])
    overfitting = bool(
        history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]
        and history["val_loss"].iloc[-1] > history["val_loss"].iloc[0]
        and auroc_delta < 0
    )
    summary = {
        "experiment": experiment_name,
        "checkpoint": str(checkpoint_path),
        "completed_epoch": 10,
        "auditable_history_epochs": "3-10",
        "label_mapping": {f"label_{index}": name for index, name in enumerate(LABEL_COLUMNS)},
        **best,
        "final_train_val_loss_gap": loss_gap,
        "last_3_epoch_macro_auroc_delta": auroc_delta,
        "last_3_epoch_macro_auprc_delta": auprc_delta,
        "overfitting_detected": overfitting,
        "minimum_val_loss_epoch": int(min_loss_row["epoch"]),
        "minimum_val_loss": float(min_loss_row["val_loss"]),
        "epoch_10_validation_degradation": {
            "vs_epoch_9_val_loss_delta": float(history.loc[history["epoch"].eq(10), "val_loss"].iloc[0] - history.loc[history["epoch"].eq(9), "val_loss"].iloc[0]),
            "vs_epoch_9_macro_auroc_delta": float(history.loc[history["epoch"].eq(10), "macro_auroc"].iloc[0] - history.loc[history["epoch"].eq(9), "macro_auroc"].iloc[0]),
            "vs_epoch_9_macro_auprc_delta": float(history.loc[history["epoch"].eq(10), "macro_auprc"].iloc[0] - history.loc[history["epoch"].eq(9), "macro_auprc"].iloc[0]),
        },
        "interpretation": "AUROC/AUPRC trends are threshold-independent; fixed-0.5 F1 is reported separately.",
    }
    atomic_write_json(stage / "analysis_summary.json", summary)
    atomic_write_json(stage / "checkpoint_selection.json", {
        "selected_checkpoint": str((output_dir / "best.pt").resolve()),
        "selected_epoch": 9,
        "selection_reason": "best observed macro AUROC and macro AUPRC",
        "macro_auroc": float(selected["macro_auroc"]),
        "macro_auprc": float(selected["macro_auprc"]),
        "test_used_for_selection": False,
    })
    update_stage(workflow_state_path, stage_name, "completed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a completed B2 history.")
    parser.add_argument("--output-dir", default="outputs/B2_densenet121_sqrt_posweight")
    parser.add_argument("--config-source", default="configs/formal_b2_densenet121_sqrt_posweight.yaml")
    parser.add_argument("--workflow-state", default="workflow_state.json")
    parser.add_argument("--experiment-name", default="B2")
    parser.add_argument("--stage-name", default="B2_analysis")
    args = parser.parse_args()
    print(json.dumps(analyze_b2(args.output_dir, args.config_source, args.workflow_state, args.experiment_name, args.stage_name), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
