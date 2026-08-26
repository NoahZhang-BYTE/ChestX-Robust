"""Preparation utilities for official NIH ChestX-ray14 metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline.labels import LABEL_COLUMNS


_METADATA_COLUMNS = ("Image Index", "Finding Labels", "Patient ID")
_OUTPUT_COLUMNS = ("image_id", "path", "split", "patient_id", *LABEL_COLUMNS)


def prepare_nih_metadata(
    metadata_path: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    seed: int = 42,
    train_list_path: str | Path | None = None,
    test_list_path: str | Path | None = None,
) -> pd.DataFrame:
    """Convert NIH metadata into a portable CSV with patient-level splits."""
    metadata = pd.read_csv(metadata_path)
    _validate_metadata(metadata)

    image_root = Path(image_root).resolve()
    paths_by_image_id = _index_images(image_root)
    image_ids = metadata["Image Index"].astype(str)
    resolved_paths = [_resolve_image_path(image_id, paths_by_image_id) for image_id in image_ids]

    labels = [_encode_findings(value) for value in metadata["Finding Labels"]]
    patient_ids = metadata["Patient ID"].tolist()
    splits = _assign_splits(
        image_ids.tolist(), patient_ids, seed, train_list_path, test_list_path
    )

    frame = pd.DataFrame(
        {
            "image_id": image_ids,
            "path": [path.relative_to(image_root).as_posix() for path in resolved_paths],
            "split": splits,
            "patient_id": patient_ids,
        }
    )
    for label, values in zip(LABEL_COLUMNS, zip(*labels, strict=True)):
        frame[label] = values

    frame = frame.loc[:, _OUTPUT_COLUMNS]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _validate_metadata(metadata: pd.DataFrame) -> None:
    missing = [column for column in _METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {', '.join(missing)}")
    if metadata.loc[:, _METADATA_COLUMNS].isna().any().any():
        raise ValueError("Required metadata columns contain missing values")
    if metadata["Image Index"].astype(str).duplicated().any():
        raise ValueError("Duplicate Image Index values are ambiguous")


def _index_images(image_root: Path) -> dict[str, list[Path]]:
    if not image_root.is_dir():
        raise ValueError(f"Image root does not exist or is not a directory: {image_root}")
    paths_by_name: dict[str, list[Path]] = {}
    for path in image_root.rglob("*"):
        if path.is_file():
            paths_by_name.setdefault(path.name, []).append(path.resolve())
    return paths_by_name


def _resolve_image_path(image_id: str, paths_by_image_id: dict[str, list[Path]]) -> Path:
    candidates = paths_by_image_id.get(Path(image_id).name, [])
    if not candidates:
        raise ValueError(f"Missing image for Image Index '{image_id}'")
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous image for Image Index '{image_id}'")
    return candidates[0]


def _encode_findings(findings: object) -> tuple[int, ...]:
    tokens = [token.strip() for token in str(findings).split("|") if token.strip()]
    unknown = sorted(set(tokens).difference((*LABEL_COLUMNS, "No Finding")))
    if unknown:
        raise ValueError(f"Unknown finding labels: {', '.join(unknown)}")
    if "No Finding" in tokens:
        return (0,) * len(LABEL_COLUMNS)
    return tuple(int(label in tokens) for label in LABEL_COLUMNS)


def _assign_splits(
    image_ids: list[str],
    patient_ids: list[object],
    seed: int,
    train_list_path: str | Path | None,
    test_list_path: str | Path | None,
) -> list[str]:
    if (train_list_path is None) != (test_list_path is None):
        raise ValueError("train_list_path and test_list_path must be supplied together")
    if train_list_path is None:
        return _split_patients(patient_ids, seed)

    train_images = _read_image_list(Path(train_list_path))
    test_images = _read_image_list(Path(test_list_path))
    metadata_images = set(image_ids)
    _validate_official_membership(metadata_images, train_images, test_images)

    train_patient_ids = [
        patient_id for image_id, patient_id in zip(image_ids, patient_ids, strict=True)
        if image_id in train_images
    ]
    test_patient_ids = {
        patient_id for image_id, patient_id in zip(image_ids, patient_ids, strict=True)
        if image_id in test_images
    }
    if set(train_patient_ids).intersection(test_patient_ids):
        raise ValueError("Official train/test lists leak patients")

    train_splits = _split_train_patients(train_patient_ids, seed)
    return ["test" if image_id in test_images else train_splits[patient_id] for image_id, patient_id in zip(image_ids, patient_ids, strict=True)]


def _read_image_list(path: Path) -> set[str]:
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate image entries in official list: {path}")
    return set(entries)


def _validate_official_membership(
    metadata_images: set[str], train_images: set[str], test_images: set[str]
) -> None:
    overlap = train_images.intersection(test_images)
    if overlap:
        raise ValueError("Official train and test lists overlap")
    listed_images = train_images.union(test_images)
    unknown = listed_images.difference(metadata_images)
    if unknown:
        raise ValueError("Official lists contain image IDs absent from metadata")
    unlisted = metadata_images.difference(listed_images)
    if unlisted:
        raise ValueError("Official lists do not cover every metadata image")


def _split_patients(patient_ids: list[object], seed: int) -> list[str]:
    patients = _shuffled_unique_patients(patient_ids, seed)
    train_end = max(1, int(len(patients) * 0.8)) if patients else 0
    val_end = train_end + int(len(patients) * 0.1)
    assignments = {
        **{patient_id: "train" for patient_id in patients[:train_end]},
        **{patient_id: "val" for patient_id in patients[train_end:val_end]},
        **{patient_id: "test" for patient_id in patients[val_end:]},
    }
    return [assignments[patient_id] for patient_id in patient_ids]


def _split_train_patients(patient_ids: list[object], seed: int) -> dict[object, str]:
    patients = _shuffled_unique_patients(patient_ids, seed)
    train_end = max(1, int(len(patients) * 0.9)) if patients else 0
    return {
        **{patient_id: "train" for patient_id in patients[:train_end]},
        **{patient_id: "val" for patient_id in patients[train_end:]},
    }


def _shuffled_unique_patients(patient_ids: list[object], seed: int) -> list[object]:
    patients = sorted(set(patient_ids), key=lambda patient_id: str(patient_id))
    return np.random.default_rng(seed).permutation(patients).tolist()
