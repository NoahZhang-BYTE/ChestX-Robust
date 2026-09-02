import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from baseline.evaluation import (
    collect_predictions,
    compute_auc_metrics,
    compute_f1_metrics,
    compute_per_class_metrics,
    find_per_class_thresholds,
)


def test_collect_predictions_returns_sigmoid_probabilities_in_eval_mode():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    model.train()
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[0.0, 1.0], [2.0, -1.0]]),
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        ),
        batch_size=1,
        shuffle=False,
    )

    targets, probabilities = collect_predictions(model, loader, "cpu")

    assert model.training is False
    np.testing.assert_array_equal(targets, [[0, 1], [1, 0]])
    np.testing.assert_allclose(
        probabilities,
        [[0.5, 0.7310586], [0.8807971, 0.2689414]],
        atol=2e-7,
    )


def test_auc_and_f1_metrics_report_perfect_prediction_scores():
    targets = np.array([[1, 0], [0, 1], [1, 1], [0, 1]])
    probabilities = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.8], [0.2, 0.8]])

    auc = compute_auc_metrics(targets, probabilities, ["a", "b"])
    f1 = compute_f1_metrics(targets, probabilities, [0.5, 0.5])

    assert auc["macro_auroc"] == pytest.approx(1.0)
    assert auc["micro_auprc"] == pytest.approx(1.0)
    assert [row["label"] for row in auc["per_class"]] == ["a", "b"]
    assert f1 == {"macro_f1": 1.0, "micro_f1": 1.0, "sample_f1": 1.0}


def test_per_class_metrics_support_vector_thresholds_and_counts():
    targets = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    probabilities = np.array([[0.6, 0.2], [0.4, 0.6], [0.8, 0.1], [0.2, 0.9]])

    rows = compute_per_class_metrics(
        targets, probabilities, ["a", "b"], thresholds=[0.5, 0.7]
    )

    assert rows[0]["threshold"] == 0.5
    assert rows[1]["threshold"] == 0.7
    assert rows[0]["positive_count"] == 2
    assert rows[0]["negative_count"] == 2
    assert rows[0]["total_count"] == 4
    assert rows[0]["specificity"] == pytest.approx(1.0)


def test_threshold_search_returns_valid_values_and_degenerate_fallback():
    targets = np.column_stack((np.array([0, 0, 1, 1]), np.zeros(4, dtype=int)))
    probabilities = np.column_stack(
        (np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.1, 0.2, 0.3, 0.4]))
    )

    with pytest.warns(RuntimeWarning, match="degenerate"):
        thresholds = find_per_class_thresholds(targets, probabilities)

    assert thresholds.shape == (2,)
    assert np.all((thresholds >= 0.0) & (thresholds <= 1.0))
    assert thresholds[1] == 0.5


@pytest.mark.parametrize(
    ("targets", "probabilities", "message"),
    [
        (np.zeros((2, 1)), np.zeros((3, 1)), "matching shapes"),
        (np.zeros((2, 1)), np.array([[0.5], [np.nan]]), "finite"),
        (np.zeros((2, 1)), np.array([[0.5], [np.inf]]), "finite"),
    ],
)
def test_metrics_reject_shape_mismatch_and_nonfinite_probabilities(
    targets, probabilities, message
):
    with pytest.raises(ValueError, match=message):
        compute_auc_metrics(targets, probabilities)
