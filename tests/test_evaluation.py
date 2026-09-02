from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from baseline.evaluation import evaluate_checkpoint
from baseline.labels import LABEL_COLUMNS
from baseline.models import build_model


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        name = f"{index}.png"
        Image.fromarray(np.full((16, 16, 3), index * 40, dtype=np.uint8)).save(image_root / name)
        rows.append({"image_id": name, "path": name, "split": split, "patient_id": index, **{label: int(label == "Atelectasis" and index == 0) for label in LABEL_COLUMNS}})
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config = {
        "data": {"csv_path": str(csv_path), "image_root": str(image_root), "image_col": "path", "label_cols": list(LABEL_COLUMNS), "image_size": 16, "batch_size": 1, "num_workers": 0, "val_split": 0.2},
        "model": {"name": "resnet18", "pretrained": False},
        "device": "cpu",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, csv_path


def test_evaluate_checkpoint_returns_complete_validation_report(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    model = build_model("resnet18", len(LABEL_COLUMNS), pretrained=False)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": yaml.safe_load(config_path.read_text()), "metrics": {"macro_f1": 0.1}}, checkpoint)
    result = evaluate_checkpoint(checkpoint, config_path, split="val", threshold=0.5)
    assert result["split"] == "val"
    assert {"macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc", "macro_f1", "micro_f1", "sample_f1", "per_label"} <= result["metrics"].keys()


def test_evaluate_checkpoint_rejects_test_threshold_fit(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    model = build_model("resnet18", len(LABEL_COLUMNS), pretrained=False)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": yaml.safe_load(config_path.read_text())}, checkpoint)
    with pytest.raises(ValueError, match="test.*fit|fit.*test"):
        evaluate_checkpoint(checkpoint, config_path, split="test", fit_thresholds=True)


def test_evaluate_checkpoint_rejects_direct_test_evaluation(tmp_path):
    config_path, _ = _write_fixture(tmp_path)
    with pytest.raises(ValueError, match="test.*frozen|frozen.*test"):
        evaluate_checkpoint(tmp_path / "unused.pt", config_path, split="test")
