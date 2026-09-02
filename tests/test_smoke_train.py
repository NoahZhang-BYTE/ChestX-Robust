from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from baseline.labels import LABEL_COLUMNS
from smoke_train import run_training_smoke_test


def test_one_epoch_smoke_supports_bce_and_capped_pos_weight(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        name = f"{index}.png"
        Image.fromarray(np.full((32, 32, 3), index * 20, dtype=np.uint8)).save(image_root / name)
        rows.append({"image_id": name, "path": name, "split": split, "patient_id": index, **{label: int(index == 0) for label in LABEL_COLUMNS}})
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"seed": 42, "device": "cpu", "data": {"csv_path": str(csv_path), "image_root": str(image_root), "image_col": "path", "label_cols": list(LABEL_COLUMNS), "image_size": 32, "batch_size": 2, "num_workers": 0}, "model": {"name": "resnet18", "pretrained": False}}), encoding="utf-8")
    b0 = run_training_smoke_test(config_path, "bce", max_batches=1)
    b1 = run_training_smoke_test(config_path, "bce_pos_weight", max_batches=1)
    assert b0.pos_weight is None
    assert b1.pos_weight is not None and max(b1.pos_weight) <= 20


def test_training_smoke_can_consume_the_complete_train_epoch(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        name = f"{index}.png"
        Image.fromarray(np.full((32, 32, 3), index * 20, dtype=np.uint8)).save(image_root / name)
        rows.append({"image_id": name, "path": name, "split": split, "patient_id": index, **{label: int(index == 0) for label in LABEL_COLUMNS}})
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"seed": 42, "device": "cpu", "data": {"csv_path": str(csv_path), "image_root": str(image_root), "image_col": "path", "label_cols": list(LABEL_COLUMNS), "image_size": 32, "batch_size": 2, "num_workers": 0}, "model": {"name": "resnet18", "pretrained": False}}), encoding="utf-8")
    result = run_training_smoke_test(config_path, "bce", max_batches=None)
    assert result.batches == 1
