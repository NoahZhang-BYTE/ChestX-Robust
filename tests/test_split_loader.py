from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from baseline.data import build_split_loader


def _split_csv(tmp_path: Path, leaked: bool = False) -> Path:
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        filename = f"image_{index}.png"
        Image.fromarray(np.full((8, 8, 3), index, dtype=np.uint8)).save(tmp_path / filename)
        rows.append(
            {
                "image_id": filename,
                "path": filename,
                "split": split,
                "patient_id": 0 if leaked and split == "val" else index,
                "finding_a": index % 2,
            }
        )
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_named_split_loader_reads_only_requested_split(tmp_path):
    loader, labels = build_split_loader(
        _split_csv(tmp_path), tmp_path, split="test", image_col="path", label_cols=["finding_a"], image_size=8
    )
    assert labels == ["finding_a"]
    assert len(loader.dataset) == 1
    assert loader.dataset.frame["split"].tolist() == ["test"]


def test_named_split_loader_rejects_patient_leakage(tmp_path):
    with pytest.raises(ValueError, match="Patient leakage"):
        build_split_loader(
            _split_csv(tmp_path, leaked=True), tmp_path, split="val", image_col="path", label_cols=["finding_a"]
        )
