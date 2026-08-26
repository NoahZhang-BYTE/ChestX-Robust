from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from baseline.data import MultiLabelImageDataset, build_dataloaders
from baseline.engine import save_checkpoint
from baseline.metrics import multilabel_metrics
from baseline.models import build_model


def _fixture_csv(tmp_path: Path) -> Path:
    rows = []
    for index, labels in enumerate(((1, 0), (0, 1), (1, 1))):
        filename = f"image_{index}.png"
        Image.fromarray(np.full((24, 24, 3), index * 60, dtype=np.uint8)).save(tmp_path / filename)
        rows.append({"image": filename, "finding_a": labels[0], "finding_b": labels[1]})
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_dataset_returns_image_and_multihot_target(tmp_path):
    csv_path = _fixture_csv(tmp_path)
    frame = pd.read_csv(csv_path)
    dataset = MultiLabelImageDataset(
        frame=frame,
        image_root=tmp_path,
        image_col="image",
        label_cols=["finding_a", "finding_b"],
    )

    image, target = dataset[0]

    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert torch.equal(target, torch.tensor([1.0, 0.0]))


def test_resnet18_has_multilabel_output_head():
    model = build_model("resnet18", num_classes=3, pretrained=False)
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 3)


def test_dataloaders_infer_labels_when_config_list_is_empty(tmp_path):
    csv_path = _fixture_csv(tmp_path)

    train_loader, val_loader, label_cols = build_dataloaders(
        csv_path=csv_path,
        image_root=tmp_path,
        image_col="image",
        label_cols=[],
        val_split=1 / 3,
        batch_size=2,
    )

    assert label_cols == ["finding_a", "finding_b"]
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1


def test_metrics_handle_single_class_label_without_crashing():
    targets = np.array([[1, 0], [1, 0]])
    probabilities = np.array([[0.9, 0.2], [0.8, 0.1]])

    metrics = multilabel_metrics(targets, probabilities)

    assert set(("macro_auroc", "micro_auroc", "macro_f1", "micro_f1", "sample_f1")) <= metrics.keys()
    assert np.isfinite(metrics["macro_f1"])
    assert np.isnan(metrics["macro_auroc"])


def test_checkpoint_contains_training_state(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=4,
        config={"model": "resnet18"},
        metrics={"macro_f1": 0.5},
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert {"model_state", "optimizer_state", "epoch", "config", "metrics"} <= checkpoint.keys()
    assert checkpoint["epoch"] == 4
