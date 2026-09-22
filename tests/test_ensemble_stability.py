from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline.labels import LABEL_COLUMNS
from scripts.analysis.analyze_ensemble_stability import (
    GLOBAL_WEIGHT,
    _bootstrap,
    _candidate_probabilities,
    _groups,
    _load_family,
    _metrics,
    _validation_patient_ids,
    run_analysis,
)


def _targets(rows: int = 8) -> np.ndarray:
    targets = np.zeros((rows, len(LABEL_COLUMNS)), dtype=np.int64)
    targets[: rows // 2, 0] = 1
    targets[::2, 1] = 1
    return targets


def _write_labels_csv(path: Path, targets: np.ndarray, patient_ids: list[int]) -> None:
    frame = pd.DataFrame({"image_id": [f"img_{i}" for i in range(len(targets))], "path": [f"{i}.png" for i in range(len(targets))], "split": "val", "patient_id": patient_ids})
    for index, label in enumerate(LABEL_COLUMNS):
        frame[label] = targets[:, index]
    frame.to_csv(path, index=False)


def _write_thresholds(path: Path, value: float = 0.5) -> None:
    path.write_text(json.dumps({"source": "validation", "label_cols": list(LABEL_COLUMNS), "thresholds": [value] * len(LABEL_COLUMNS)}), encoding="utf-8")


def _write_weights(path: Path, weights: np.ndarray) -> None:
    pd.DataFrame({"label": list(LABEL_COLUMNS), "b4_weight": weights, "b5_weight": 1.0 - weights}).to_csv(path, index=False)


def test_candidate_weights_span_global_per_class_and_shrinkage():
    b4 = np.full((4, len(LABEL_COLUMNS)), 0.8)
    b5 = np.full((4, len(LABEL_COLUMNS)), 0.2)
    per_class = np.linspace(0.0, 1.0, len(LABEL_COLUMNS))

    candidates = _candidate_probabilities(b4, b5, per_class)

    assert set(candidates) == {"global_0.5", "global_0.4", "per_class", "per_class_shrinkage_0.25", "per_class_shrinkage_0.50", "per_class_shrinkage_0.75"}
    np.testing.assert_allclose(candidates["global_0.4"][1], np.full(len(LABEL_COLUMNS), GLOBAL_WEIGHT))
    np.testing.assert_allclose(candidates["per_class"][1], per_class)
    for shrinkage in (0.25, 0.5, 0.75):
        expected = GLOBAL_WEIGHT + shrinkage * (per_class - GLOBAL_WEIGHT)
        np.testing.assert_allclose(candidates[f"per_class_shrinkage_{shrinkage:.2f}"][1], expected)


def test_patient_groups_keep_all_rows_of_each_patient():
    patient_ids = np.asarray([1, 2, 1, 2, 3])

    groups = _groups(patient_ids)

    assert sorted(len(group) for group in groups) == [1, 2, 2]
    assert sorted(np.concatenate(groups).tolist()) == [0, 1, 2, 3, 4]


def test_validation_patient_ids_requires_matching_target_order(tmp_path: Path):
    targets = _targets()
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, targets, patient_ids=[1, 1, 2, 2, 3, 3, 4, 4])

    assert _validation_patient_ids(labels_csv, targets).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]

    shuffled = targets.copy()
    shuffled[[0, 1]] = shuffled[[1, 0]]
    with pytest.raises(ValueError, match="ordering"):
        _validation_patient_ids(labels_csv, shuffled)


def test_single_member_family_is_reported_as_single_source(tmp_path: Path):
    targets = _targets()
    member = tmp_path / "b4.npy"
    np.save(member, np.full(targets.shape, 0.5))

    mean, sources = _load_family([member], targets, "B4 family")

    np.testing.assert_allclose(mean, np.full(targets.shape, 0.5))
    assert len(sources) == 1


def test_family_average_uses_all_members(tmp_path: Path):
    targets = _targets()
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    np.save(first, np.zeros(targets.shape))
    np.save(second, np.ones(targets.shape))

    mean, sources = _load_family([first, second], targets, "B4 family")

    np.testing.assert_allclose(mean, np.full(targets.shape, 0.5))
    assert len(sources) == 2


def test_paired_bootstrap_delta_is_antisymmetric_for_the_baseline():
    targets = _targets(rows=40)
    thresholds = np.full(len(LABEL_COLUMNS), 0.5)
    candidates = {
        "global_0.4": (np.full(targets.shape, 0.5), np.full(len(LABEL_COLUMNS), GLOBAL_WEIGHT)),
        "global_0.5": (np.full(targets.shape, 0.5), np.full(len(LABEL_COLUMNS), 0.5)),
    }
    groups = _groups(np.repeat(np.arange(10), 4))

    summary, _raw = _bootstrap(targets, candidates, thresholds, groups, draws=5, seed=7)

    baseline = summary.loc[(summary["candidate"] == "global_0.4") & (summary["metric"] == "macro_auprc")].iloc[0]
    assert baseline["delta_mean_vs_global_0.4"] == 0.0
    assert baseline["probability_delta_positive"] == 0.0


def test_run_analysis_never_reads_test_and_writes_protocol(tmp_path: Path):
    targets = _targets()
    labels_csv = tmp_path / "labels.csv"
    _write_labels_csv(labels_csv, targets, patient_ids=[1, 1, 2, 2, 3, 3, 4, 4])
    thresholds = tmp_path / "thresholds.json"
    _write_thresholds(thresholds)
    weights = tmp_path / "per_class_weights.csv"
    _write_weights(weights, np.linspace(0.2, 0.8, len(LABEL_COLUMNS)))
    b4 = tmp_path / "b4.npy"
    b5 = tmp_path / "b5.npy"
    np.save(b4, np.full(targets.shape, 0.6))
    np.save(b5, np.full(targets.shape, 0.4))
    val_targets = tmp_path / "val_targets.npy"
    np.save(val_targets, targets)
    output = tmp_path / "out"

    result = run_analysis(
        b4_probs=b4, b5_probs=b5, val_targets=val_targets,
        labels_csv=labels_csv, thresholds=thresholds, per_class_weights_path=weights,
        output_dir=output, bootstrap_draws=5, seed=3,
    )

    protocol = json.loads((output / "analysis_protocol.json").read_text(encoding="utf-8"))
    assert protocol["test_read"] is False
    assert protocol["selection_or_tuning_split"] == "val"
    assert protocol["bootstrap"]["unit"] == "patient"
    assert (output / "patient_bootstrap_summary.csv").is_file()
    assert (output / "complementarity_by_label.csv").is_file()
    assert {"candidate", "b4_weight_mean", "macro_auroc", "macro_auprc", "macro_f1"} <= set(pd.read_csv(output / "validation_candidate_metrics.csv").columns)
