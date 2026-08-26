from pathlib import Path

import pandas as pd
import pytest

from baseline.labels import LABEL_COLUMNS, NUM_CLASSES
from baseline.nih import prepare_nih_metadata


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
