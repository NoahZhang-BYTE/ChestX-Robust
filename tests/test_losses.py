from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from baseline.engine import build_training_criterion
from baseline.losses import (
    build_loss,
    compute_capped_pos_weight,
    compute_sqrt_capped_pos_weight,
)


def test_bce_factory_matches_plain_bce():
    criterion = build_loss("bce", torch.tensor([2.0, 3.0]))
    expected = torch.nn.BCEWithLogitsLoss()
    assert criterion.pos_weight is None
    assert torch.allclose(criterion(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])), expected(torch.tensor([[0.0, 1.0]]), torch.tensor([[1.0, 0.0]])))


def test_pos_weight_is_capped_and_finite():
    weights = compute_capped_pos_weight(torch.tensor([1.0, 2.0]), torch.tensor([100.0, 10.0]), cap=20.0)
    assert weights.tolist() == [20.0, 5.0]
    assert torch.isfinite(weights).all()


def test_sqrt_pos_weight_applies_sqrt_before_cap():
    weights = compute_sqrt_capped_pos_weight(
        torch.tensor([1.0, 2.0]), torch.tensor([10000.0, 10.0]), cap=20.0
    )
    assert torch.allclose(weights, torch.tensor([20.0, 5.0**0.5]))
    criterion = build_loss(
        "bce_sqrt_pos_weight", torch.tensor([1.0]), torch.tensor([81.0]), cap=20.0
    )
    assert torch.allclose(criterion.pos_weight, torch.tensor([9.0]))


def test_sqrt_pos_weight_metadata_uses_train_loader_rows_and_label_order():
    dataset = SimpleNamespace(
        frame=pd.DataFrame(
            {
                "split": ["train", "train", "train", "train"],
                "second": [0, 1, 0, 0],
                "first": [1, 1, 0, 0],
            }
        ),
        label_cols=["second", "first"],
    )
    criterion, metadata = build_training_criterion(
        SimpleNamespace(dataset=dataset),
        {"training": {"loss": {"name": "bce_sqrt_pos_weight", "cap": 20.0}}},
        torch.device("cpu"),
    )
    assert criterion.pos_weight.device.type == "cpu"
    assert metadata["count_source"] == "train"
    assert [row["class"] for row in metadata["per_label"]] == ["second", "first"]
    assert [row["positive_count"] for row in metadata["per_label"]] == [1, 2]
    assert torch.allclose(criterion.pos_weight, torch.tensor([3.0**0.5, 1.0]))


def test_pos_weight_rejects_a_loader_that_contains_non_train_rows():
    dataset = SimpleNamespace(
        frame=pd.DataFrame({"split": ["train", "val"], "label": [1, 1]}),
        label_cols=["label"],
    )
    with pytest.raises(ValueError, match="train split"):
        build_training_criterion(
            SimpleNamespace(dataset=dataset),
            {"training": {"loss": {"name": "bce_sqrt_pos_weight"}}},
            torch.device("cpu"),
        )


def test_unknown_loss_is_rejected():
    with pytest.raises(ValueError, match="loss name"):
        build_loss("focal", torch.ones(2))
