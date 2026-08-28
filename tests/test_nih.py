from pathlib import Path
import subprocess
import sys
from math import isfinite

import pandas as pd
import pytest
import yaml
from PIL import Image

from baseline.labels import LABEL_COLUMNS, NUM_CLASSES
from baseline.nih import _split_patients, prepare_nih_metadata, validate_labels_csv
from smoke_test import run_smoke_test


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-an-image")


def write_nih_fixture(tmp_path: Path, finding: str = "Cardiomegaly|Effusion") -> Path:
    image_root = tmp_path / "images"
    _write_image(image_root / "images_001" / "a.png")
    _write_image(image_root / "images_002" / "b.png")
    _write_image(image_root / "images_003" / "c.png")
    metadata_path = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame(
        {
            "Image Index": ["a.png", "b.png", "c.png"],
            "Finding Labels": [finding, "No Finding", "Mass"],
            "Patient ID": [1, 1, 2],
        }
    ).to_csv(metadata_path, index=False)
    return metadata_path


def test_prepare_nih_writes_multihot_labels_and_patient_safe_splits(tmp_path):
    metadata = write_nih_fixture(tmp_path)
    frame = prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")

    assert list(frame.columns) == ["image_id", "path", "split", "patient_id", *LABEL_COLUMNS]
    assert frame.loc[frame.image_id.eq("a.png"), ["Cardiomegaly", "Effusion"]].iloc[0].tolist() == [1, 1]
    assert frame.loc[frame.image_id.eq("b.png"), list(LABEL_COLUMNS)].iloc[0].tolist() == [0] * NUM_CLASSES
    assert frame.loc[frame.image_id.eq("a.png"), "path"].iloc[0] == "images_001/a.png"
    assert set(frame.split) <= {"train", "val", "test"}
    assert frame.groupby("patient_id").split.nunique().max() == 1
    assert pd.read_csv(tmp_path / "labels.csv").equals(frame)


def test_prepare_nih_rejects_unknown_non_empty_finding(tmp_path):
    metadata = write_nih_fixture(tmp_path, finding="Imaginary disease")

    with pytest.raises(ValueError, match="Unknown finding labels"):
        prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")


def test_prepare_nih_maps_no_finding_to_an_all_zero_target(tmp_path):
    metadata = write_nih_fixture(tmp_path, finding="No Finding|Mass")

    frame = prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")

    assert frame.loc[frame.image_id.eq("a.png"), list(LABEL_COLUMNS)].iloc[0].tolist() == [0] * NUM_CLASSES


def test_prepare_nih_rejects_missing_and_ambiguous_images(tmp_path):
    metadata = write_nih_fixture(tmp_path)
    (tmp_path / "images" / "images_002" / "b.png").unlink()

    with pytest.raises(ValueError, match="Missing image"):
        prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")

    _write_image(tmp_path / "images" / "images_004" / "b.png")
    _write_image(tmp_path / "images" / "images_002" / "b.png")

    with pytest.raises(ValueError, match="Ambiguous image"):
        prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")


def test_prepare_nih_rejects_repeated_image_ids(tmp_path):
    metadata = write_nih_fixture(tmp_path)
    frame = pd.read_csv(metadata)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(metadata, index=False)

    with pytest.raises(ValueError, match="Duplicate Image Index"):
        prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")


def test_prepare_nih_preserves_official_test_membership_and_derives_validation_from_train(tmp_path):
    metadata = write_nih_fixture(tmp_path)
    train_list = tmp_path / "train_val_list.txt"
    test_list = tmp_path / "test_list.txt"
    train_list.write_text("a.png\nb.png\n", encoding="utf-8")
    test_list.write_text("c.png\n", encoding="utf-8")

    frame = prepare_nih_metadata(
        metadata,
        tmp_path / "images",
        tmp_path / "labels.csv",
        train_list_path=train_list,
        test_list_path=test_list,
    )

    assert set(frame.loc[frame.image_id.eq("c.png"), "split"]) == {"test"}
    assert set(frame.loc[frame.image_id.isin(["a.png", "b.png"]), "split"]) <= {"train", "val"}


def test_prepare_nih_requires_all_metadata_columns(tmp_path):
    metadata = write_nih_fixture(tmp_path)
    pd.read_csv(metadata).drop(columns="Patient ID").to_csv(metadata, index=False)

    with pytest.raises(ValueError, match="Missing required metadata columns"):
        prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")


def test_prepare_nih_rejects_schema_valid_empty_metadata(tmp_path):
    metadata = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame(columns=["Image Index", "Finding Labels", "Patient ID"]).to_csv(
        metadata, index=False
    )
    (tmp_path / "images").mkdir()

    with pytest.raises(ValueError, match="NIH metadata contains no rows"):
        prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")


def test_prepare_nih_keeps_a_single_patient_in_train(tmp_path):
    image_root = tmp_path / "images"
    _write_image(image_root / "nested" / "a.png")
    metadata = tmp_path / "Data_Entry_2017.csv"
    pd.DataFrame(
        {
            "Image Index": ["a.png"],
            "Finding Labels": ["No Finding"],
            "Patient ID": [1],
        }
    ).to_csv(metadata, index=False)

    frame = prepare_nih_metadata(metadata, image_root, tmp_path / "labels.csv")

    assert frame.split.tolist() == ["train"]


def test_patient_splitting_preserves_mixed_patient_id_types():
    splits = _split_patients([1, "patient-2"], seed=42)

    assert len(splits) == 2
    assert set(splits) == {"train", "test"}


def prepare_fixture_labels_csv(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    _write_image(image_root / "nested" / "a.png")
    _write_image(image_root / "nested" / "b.png")
    _write_image(image_root / "nested" / "c.png")
    frame = pd.DataFrame(
        {
            "image_id": ["a.png", "b.png", "c.png"],
            "path": ["nested/a.png", "nested/b.png", "nested/c.png"],
            "split": ["train", "val", "test"],
            "patient_id": [1, 2, 3],
        }
    )
    for label in LABEL_COLUMNS:
        frame[label] = 0
    frame.loc[0, "Atelectasis"] = 1
    labels_csv = tmp_path / "labels.csv"
    frame.to_csv(labels_csv, index=False)
    return labels_csv


def test_validator_reports_all_splits_and_labels_and_rejects_patient_leakage(tmp_path, capsys):
    labels_csv = prepare_fixture_labels_csv(tmp_path)

    returned = validate_labels_csv(labels_csv, tmp_path / "images")

    assert returned.equals(pd.read_csv(labels_csv))
    output = capsys.readouterr().out
    assert "Train: 1 images" in output
    assert "Val: 1 images" in output
    assert "Test: 1 images" in output
    assert "Atelectasis: 1 / 1 = 100.0%" in output
    assert "Cardiomegaly: 0 / 1 = 0.0%" in output
    for label in LABEL_COLUMNS:
        assert f"{label}:" in output

    leaked = pd.read_csv(labels_csv)
    leaked.loc[1, "patient_id"] = 1
    leaked.to_csv(labels_csv, index=False)
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_labels_csv(labels_csv, tmp_path / "images")


def test_validator_reports_zero_row_splits(tmp_path, capsys):
    labels_csv = prepare_fixture_labels_csv(tmp_path)
    frame = pd.read_csv(labels_csv)
    frame["split"] = "train"
    frame.to_csv(labels_csv, index=False)

    validate_labels_csv(labels_csv, tmp_path / "images")

    output = capsys.readouterr().out
    assert "Val: 0 images" in output
    assert "Test: 0 images" in output
    assert output.count("0 / 0 = 0.0%") == 2 * NUM_CLASSES


def test_validator_rejects_schema_valid_empty_labels_csv(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame(
        columns=["image_id", "path", "split", "patient_id", *LABEL_COLUMNS]
    ).to_csv(labels_csv, index=False)
    (tmp_path / "images").mkdir()

    with pytest.raises(ValueError, match="contains no rows"):
        validate_labels_csv(labels_csv, tmp_path / "images")


def test_validator_rejects_boolean_label_values(tmp_path):
    labels_csv = prepare_fixture_labels_csv(tmp_path)
    frame = pd.read_csv(labels_csv)
    frame["Atelectasis"] = [True, False, True]
    frame.to_csv(labels_csv, index=False)

    with pytest.raises(ValueError, match="0 or 1"):
        validate_labels_csv(labels_csv, tmp_path / "images")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="Hernia"), "required columns"),
        (lambda frame: frame.assign(extra=1), "exactly"),
        (lambda frame: frame.assign(Atelectasis=2), "0 or 1"),
        (lambda frame: frame.assign(split="development"), "split"),
        (lambda frame: frame.assign(image_id="a.png"), "Duplicate image_id"),
        (lambda frame: frame.assign(path="../outside.png"), "relative path"),
        (
            lambda frame: frame.assign(path="C:/absolute/path.png"),
            "relative path",
        ),
        (lambda frame: frame.assign(path="missing.png"), "does not exist"),
        (
            lambda frame: frame.assign(
                patient_id=frame["patient_id"].mask(frame.index == 0)
            ),
            "missing values",
        ),
    ],
)
def test_validator_rejects_malformed_labels_csv(tmp_path, mutate, message):
    labels_csv = prepare_fixture_labels_csv(tmp_path)
    frame = mutate(pd.read_csv(labels_csv))
    frame.to_csv(labels_csv, index=False)

    with pytest.raises(ValueError, match=message):
        validate_labels_csv(labels_csv, tmp_path / "images")


def test_validate_data_cli_validates_csv_and_prints_statistics(tmp_path):
    labels_csv = prepare_fixture_labels_csv(tmp_path)
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "validate_data.py",
            "--csv",
            str(labels_csv),
            "--data-root",
            str(tmp_path / "images"),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Train: 1 images" in result.stdout


def _write_valid_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color="white").save(path)


def write_smoke_config(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    for image_name in ("train-a.png", "train-b.png", "val.png"):
        _write_valid_image(image_root / image_name)

    frame = pd.DataFrame(
        {
            "image_id": ["train-a.png", "train-b.png", "val.png"],
            "path": ["train-a.png", "train-b.png", "val.png"],
            "split": ["train", "train", "val"],
            "patient_id": [1, 2, 3],
        }
    )
    for label in LABEL_COLUMNS:
        frame[label] = 0
    frame.loc[0, "Atelectasis"] = 1
    labels_csv = tmp_path / "labels.csv"
    frame.to_csv(labels_csv, index=False)

    config = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "csv_path": str(labels_csv),
            "image_root": str(image_root),
            "image_col": "path",
            "label_cols": [],
            "image_size": 32,
            "val_split": 0.2,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {"name": "resnet18", "pretrained": False},
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_smoke_pipeline_returns_matching_nih_target_and_logit_shapes(tmp_path, capsys):
    config_path = write_smoke_config(tmp_path)

    result = run_smoke_test(config_path)

    assert result.images_shape == (2, 3, 32, 32)
    assert result.labels_shape == (2, NUM_CLASSES)
    assert result.logits_shape == result.labels_shape
    assert isfinite(result.loss)
    assert result.loss >= 0
    assert "logits.dtype=torch.float32" in capsys.readouterr().out
