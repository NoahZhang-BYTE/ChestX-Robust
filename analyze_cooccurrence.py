"""Analyze train-split label co-occurrence without changing labels or splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from baseline.labels import LABEL_COLUMNS
from workflow_state import atomic_write_csv, atomic_write_json, update_stage


def analyze_cooccurrence(
    labels_csv: str | Path = "data/labels.csv",
    output_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight/post_analysis/cooccurrence",
    workflow_state_path: str | Path = "workflow_state.json",
    stage_name: str = "cooccurrence_analysis",
) -> dict[str, str]:
    frame = pd.read_csv(labels_csv)
    train = frame.loc[frame["split"].eq("train"), list(LABEL_COLUMNS)].to_numpy(dtype=np.int64)
    if train.size == 0:
        raise ValueError("train split is empty")
    matrix = train.T @ train
    totals = train.sum(axis=0)
    conditional = np.divide(matrix, totals[:, None], out=np.zeros_like(matrix, dtype=float), where=totals[:, None] != 0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows = [{"label_index": i, "label_name": label, **{name: int(matrix[i, j]) for j, name in enumerate(LABEL_COLUMNS)}} for i, label in enumerate(LABEL_COLUMNS)]
    conditional_rows = [{"label_index": i, "label_name": label, **{name: float(conditional[i, j]) for j, name in enumerate(LABEL_COLUMNS)}} for i, label in enumerate(LABEL_COLUMNS)]
    atomic_write_csv(output_dir / "label_cooccurrence.csv", matrix_rows[0].keys(), matrix_rows)
    atomic_write_csv(output_dir / "label_conditional_probability.csv", conditional_rows[0].keys(), conditional_rows)
    scale = 255.0 / max(float(conditional.max()), 1e-9)
    image = Image.new("RGB", (700, 700), "white")
    draw = ImageDraw.Draw(image)
    origin, cell = 150, 32
    for i in range(len(LABEL_COLUMNS)):
        for j in range(len(LABEL_COLUMNS)):
            value = int(min(255, conditional[i, j] * scale))
            draw.rectangle((origin + j * cell, origin + i * cell, origin + (j + 1) * cell, origin + (i + 1) * cell), fill=(255 - value, 255 - value, 255))
        draw.text((10, origin + i * cell + 8), f"{i}: {LABEL_COLUMNS[i][:16]}", fill="black")
        draw.text((origin + i * cell + 3, 105), str(i), fill="black")
    image.save(output_dir / "label_cooccurrence.png", format="PNG")
    summary = {
        label: sorted(
            ((LABEL_COLUMNS[j], float(conditional[i, j])) for j in range(len(LABEL_COLUMNS)) if j != i),
            key=lambda pair: pair[1], reverse=True,
        )[:3]
        for i, label in enumerate(LABEL_COLUMNS)
        if label in {"Infiltration", "Pneumonia", "Hernia"}
    }
    atomic_write_json(output_dir / "summary.json", summary)
    update_stage(workflow_state_path, stage_name, "completed")
    return {"output_dir": str(output_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze train label co-occurrence.")
    parser.add_argument("--labels-csv", default="data/labels.csv")
    parser.add_argument("--output-dir", default="outputs/B2_densenet121_sqrt_posweight/post_analysis/cooccurrence")
    parser.add_argument("--workflow-state", default="workflow_state.json")
    parser.add_argument("--stage-name", default="cooccurrence_analysis")
    args = parser.parse_args()
    print(json.dumps(analyze_cooccurrence(args.labels_csv, args.output_dir, args.workflow_state, args.stage_name), indent=2))


if __name__ == "__main__":
    main()
