import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from run_final_test import build_parser as build_final_test_parser
from run_final_test import bind_frozen_data_paths
from run_final_test import load_and_validate_frozen_bundle
from run_final_test import run_final_test
from tune_thresholds import build_threshold_artifact
from tune_thresholds import validate_designated_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_threshold_artifact_is_validation_only_and_keeps_label_order():
    labels = ["a", "b"]
    artifact = build_threshold_artifact(
        labels,
        np.array([0.25, 0.75]),
        checkpoint_sha256="1" * 64,
        labels_csv_sha256="2" * 64,
        fallback_reasons=[None, None],
    )

    assert artifact["source"] == "validation"
    assert artifact["fit_split"] == "val"
    assert artifact["label_cols"] == labels
    assert artifact["thresholds"] == [0.25, 0.75]
    assert artifact["thresholds_by_label"] == {"a": 0.25, "b": 0.75}


def test_final_test_cli_does_not_accept_tuning_inputs():
    option_strings = {
        option
        for action in build_final_test_parser()._actions
        for option in action.option_strings
    }

    assert "--split" not in option_strings
    assert "--threshold" not in option_strings
    assert "--checkpoint" not in option_strings
    assert "--output-dir" not in option_strings


def test_final_test_binds_loader_to_frozen_absolute_data_paths(tmp_path):
    frozen_csv = (tmp_path / "frozen-labels.csv").resolve()
    frozen_root = (tmp_path / "frozen-images").resolve()
    data_config = {"csv_path": "data/labels.csv", "image_root": "data/images"}
    protocol = {
        "labels_csv": {"path": str(frozen_csv)},
        "image_root": str(frozen_root),
    }

    bound = bind_frozen_data_paths(data_config, protocol)

    assert Path(bound["csv_path"]) == frozen_csv
    assert Path(bound["image_root"]) == frozen_root


def test_threshold_freeze_rejects_non_best_or_non_resnet_checkpoint():
    with pytest.raises(ValueError, match="best.pt"):
        validate_designated_checkpoint(
            Path("checkpoints/imagenet_pretrained/last.pt"),
            {"model": {"name": "resnet18"}},
            {"macro_auroc": 0.8},
        )
    with pytest.raises(ValueError, match="ResNet18"):
        validate_designated_checkpoint(
            Path("checkpoints/imagenet_pretrained/best.pt"),
            {"model": {"name": "densenet121"}},
            {"macro_auroc": 0.8},
        )


def test_final_test_api_rejects_alternate_output_directory(tmp_path):
    with pytest.raises(ValueError, match="fixed test output"):
        run_final_test(tmp_path, output_dir=tmp_path / "another")


def test_final_test_module_does_not_import_or_call_threshold_fitting():
    source = Path("run_final_test.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "find_per_class_thresholds" not in imported | called
    assert "fit_per_class_thresholds" not in imported | called


def test_frozen_bundle_rejects_modified_threshold_artifact(tmp_path):
    checkpoint = tmp_path / "best.pt"
    labels_csv = tmp_path / "labels.csv"
    checkpoint.write_bytes(b"checkpoint")
    labels_csv.write_bytes(b"labels")
    thresholds = {
        "schema_version": 1,
        "source": "validation",
        "fit_split": "val",
        "strategy": "per_class_max_f1",
        "label_cols": ["a", "b"],
        "thresholds": [0.25, 0.75],
        "thresholds_by_label": {"a": 0.25, "b": 0.75},
        "checkpoint_sha256": _sha256(checkpoint),
        "labels_csv_sha256": _sha256(labels_csv),
    }
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(json.dumps(thresholds), encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "status": "frozen",
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "labels_csv": {"path": str(labels_csv), "sha256": _sha256(labels_csv)},
        "label_cols": ["a", "b"],
        "num_classes": 2,
        "threshold": {
            "source": "validation",
            "strategy": "per_class_max_f1",
            "artifact": "thresholds.json",
            "sha256": _sha256(threshold_path),
        },
        "test_used_for_tuning": False,
    }
    (tmp_path / "evaluation_protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )

    loaded_protocol, loaded_thresholds = load_and_validate_frozen_bundle(tmp_path)
    assert loaded_protocol["status"] == "frozen"
    assert loaded_thresholds["thresholds"] == [0.25, 0.75]

    thresholds["thresholds"][0] = 0.5
    threshold_path.write_text(json.dumps(thresholds), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold artifact SHA256"):
        load_and_validate_frozen_bundle(tmp_path)
