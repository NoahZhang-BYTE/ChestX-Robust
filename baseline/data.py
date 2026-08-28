from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    seed: int = 42,
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
        train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
        val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
        if train_frame.empty or val_frame.empty:
            raise ValueError("CSV split column must contain non-empty train and val rows")
    else:
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
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1")
    loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers:
        loader_kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        label_cols,
    )
