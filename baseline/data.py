from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from .reproducibility import make_generator, seed_worker
from torchvision import transforms

from baseline.labels import LABEL_COLUMNS


class MultiLabelImageDataset(Dataset):
    """CSV-backed dataset with one image path column and binary label columns."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path = ".",
        image_col: str = "image",
        label_cols: Sequence[str] = (),
        transform=None,
    ) -> None:
        if image_col not in frame.columns:
            raise ValueError(f"Missing image column: {image_col}")
        missing = [column for column in label_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")
        if not label_cols:
            raise ValueError("label_cols must contain at least one label")

        self.frame = frame.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.image_col = image_col
        self.label_cols = list(label_cols)
        self.transform = transform or transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image_path = Path(str(row[self.image_col]))
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        target = torch.tensor(
            row[self.label_cols].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        return image_tensor, target


def _split_frame(frame: pd.DataFrame, val_split: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("At least two samples are required for a train/validation split")
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_count = max(1, int(round(len(shuffled) * val_split)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled.iloc[val_count:].reset_index(drop=True), shuffled.iloc[:val_count].reset_index(drop=True)


def build_dataloaders(
    csv_path: str | Path,
    image_root: str | Path = ".",
    image_col: str = "image",
    label_cols: Sequence[str] | None = None,
    val_split: float = 0.2,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    persistent_workers: bool = True,
    seed: int = 42,
    deterministic: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
    frame = pd.read_csv(csv_path)
    if not label_cols:
        if all(column in frame.columns for column in LABEL_COLUMNS):
            label_cols = list(LABEL_COLUMNS)
        else:
            metadata_columns = {image_col, "image_id", "path", "split", "patient_id"}
            label_cols = [column for column in frame.columns if column not in metadata_columns]
    label_cols = list(label_cols)
    if "split" in frame.columns:
        _validate_persisted_splits(frame)
        train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
        val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
        if train_frame.empty or val_frame.empty:
            raise ValueError("CSV split column must contain non-empty train and val rows")
    else:
        if "patient_id" in frame.columns:
            raise ValueError("Patient-addressable CSV requires a persisted split column")
        train_frame, val_frame = _split_frame(frame, val_split, seed)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    train_dataset = MultiLabelImageDataset(train_frame, image_root, image_col, label_cols, train_transform)
    val_dataset = MultiLabelImageDataset(val_frame, image_root, image_col, label_cols, val_transform)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "worker_init_fn": seed_worker, "generator": make_generator(seed)}
    if num_workers:
        if prefetch_factor is None or prefetch_factor < 1:
            raise ValueError("prefetch_factor must be at least 1 when num_workers is enabled")
        loader_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=(persistent_workers and not deterministic))
    val_kwargs = dict(loader_kwargs)
    val_kwargs["shuffle"] = False
    val_kwargs["generator"] = make_generator(seed + 1)
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, **val_kwargs),
        label_cols,
    )


def build_split_loader(
    csv_path: str | Path,
    image_root: str | Path = ".",
    split: str = "val",
    image_col: str = "image",
    label_cols: Sequence[str] | None = None,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    persistent_workers: bool = True,
    preprocessing: dict | None = None,
) -> tuple[DataLoader, list[str]]:
    """Load one persisted split after validating split and patient isolation."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train, val, test")
    frame = pd.read_csv(csv_path)
    if "split" not in frame.columns:
        raise ValueError("Named split loading requires a persisted split column")
    _validate_persisted_splits(frame)
    if label_cols is None or not label_cols:
        if all(column in frame.columns for column in LABEL_COLUMNS):
            label_cols = list(LABEL_COLUMNS)
        else:
            metadata_columns = {image_col, "image_id", "path", "split", "patient_id"}
            label_cols = [column for column in frame.columns if column not in metadata_columns]
    label_cols = list(label_cols)
    subset = frame.loc[frame["split"].eq(split)].reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"Persisted split '{split}' contains no rows")
    spec = evaluation_preprocessing_spec(image_size)
    if preprocessing is not None and preprocessing != spec:
        raise ValueError("Evaluation preprocessing does not match the frozen specification")
    normalize = transforms.Normalize(
        mean=spec["normalize_mean"], std=spec["normalize_std"]
    )
    transform = transforms.Compose(
        [transforms.Resize(tuple(spec["resize"])), transforms.ToTensor(), normalize]
    )
    dataset = MultiLabelImageDataset(subset, image_root, image_col, label_cols, transform)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers:
        if prefetch_factor is None or prefetch_factor < 1:
            raise ValueError("prefetch_factor must be at least 1 when num_workers is enabled")
        loader_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    return DataLoader(dataset, shuffle=False, **loader_kwargs), label_cols


def build_evaluation_loader(
    csv_path: str | Path,
    image_root: str | Path = ".",
    split: str = "val",
    image_col: str = "image",
    label_cols: Sequence[str] | None = None,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 0,
    prefetch_factor: int | None = 2,
    persistent_workers: bool = True,
    preprocessing: dict | None = None,
) -> tuple[DataLoader, list[str]]:
    """Load a persisted validation or test split with deterministic preprocessing."""
    if split not in {"val", "test"}:
        raise ValueError("evaluation split must be val or test")
    return build_split_loader(
        csv_path=csv_path,
        image_root=image_root,
        split=split,
        image_col=image_col,
        label_cols=label_cols,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        preprocessing=preprocessing,
    )


def evaluation_preprocessing_spec(image_size: int) -> dict:
    """Return the canonical deterministic validation/test preprocessing spec."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    return {
        "resize": [image_size, image_size],
        "center_crop": None,
        "to_tensor": True,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "random_augmentation": False,
    }


def _validate_persisted_splits(frame: pd.DataFrame) -> None:
    if not bool(frame["split"].isin(("train", "val", "test")).all()):
        raise ValueError("CSV split column contains an invalid split")
    if "patient_id" not in frame.columns:
        return
    patients_by_split = {
        split: set(frame.loc[frame["split"].eq(split), "patient_id"])
        for split in ("train", "val", "test")
    }
    for index, first_split in enumerate(("train", "val", "test")):
        for second_split in ("train", "val", "test")[index + 1 :]:
            if patients_by_split[first_split].intersection(patients_by_split[second_split]):
                raise ValueError(
                    f"Patient leakage between {first_split} and {second_split} splits"
                )
