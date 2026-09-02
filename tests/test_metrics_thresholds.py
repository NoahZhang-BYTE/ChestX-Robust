import numpy as np
import pytest

from baseline.metrics import (
    apply_thresholds,
    fit_per_class_thresholds,
    load_threshold_artifact,
    multilabel_metrics,
    normalize_thresholds,
)


def test_scalar_and_vector_thresholds_are_equivalent():
    targets = np.array([[1, 0], [0, 1], [1, 1]])
    probabilities = np.array([[0.9, 0.2], [0.4, 0.8], [0.6, 0.7]])
    assert multilabel_metrics(targets, probabilities, 0.5)["macro_f1"] == multilabel_metrics(
        targets, probabilities, [0.5, 0.5]
    )["macro_f1"]


def test_threshold_vector_keeps_label_order():
    probabilities = np.array([[0.30, 0.30]])
    assert apply_thresholds(probabilities, [0.20, 0.40]).tolist() == [[1, 0]]


def test_fit_requires_val_and_marks_single_class():
    with pytest.raises(ValueError, match="fit_split must be 'val'"):
        fit_per_class_thresholds(np.array([[1], [0]]), np.array([[0.8], [0.2]]), ["x"], "test")
    artifact = fit_per_class_thresholds(
        np.zeros((3, 1), dtype=int), np.full((3, 1), 0.5), ["x"]
    )
    assert artifact["thresholds"] == [1.0]
    assert artifact["available"] == [False]
    assert artifact["reasons"] == ["no_positive"]


def test_threshold_tie_uses_highest_value():
    targets = np.array([[1], [0], [0], [1]])
    probabilities = np.array([[0.9], [0.8], [0.7], [0.6]])
    artifact = fit_per_class_thresholds(targets, probabilities, ["x"])
    assert artifact["thresholds"] == [0.9]


def test_per_label_metrics_include_prevalence_and_null_degenerate_scores():
    targets = np.array([[1, 0], [0, 0]])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.3]])
    metrics = multilabel_metrics(targets, probabilities, 0.5, ["a", "b"])
    assert metrics["per_label"][0]["prevalence"] == 0.5
    assert metrics["per_label"][0]["positive_count"] == 1
    assert metrics["per_label"][1]["degenerate"] is True
    for key in ("precision", "recall", "f1", "auroc", "auprc"):
        assert metrics["per_label"][1][key] is None


def test_threshold_artifact_round_trips_and_rejects_label_order(tmp_path):
    artifact = fit_per_class_thresholds(
        np.array([[1, 0], [0, 1]]), np.array([[0.8, 0.2], [0.3, 0.9]]), ["a", "b"]
    )
    path = tmp_path / "thresholds.json"
    path.write_text(__import__("json").dumps(artifact), encoding="utf-8")
    assert load_threshold_artifact(path, ["a", "b"])["thresholds"] == artifact["thresholds"]
    with pytest.raises(ValueError, match="label order"):
        load_threshold_artifact(path, ["b", "a"])


def test_normalize_thresholds_rejects_wrong_length_and_out_of_range():
    with pytest.raises(ValueError, match="one threshold per label"):
        normalize_thresholds([0.5], 2)
    with pytest.raises(ValueError, match="between 0 and 1"):
        normalize_thresholds([1.1, 0.5], 2)
