from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from baseline import engine
from baseline.asl import AsymmetricLoss
from baseline.engine import build_training_criterion, fit, load_training_checkpoint


class _RandomDataset(Dataset):
    """Small dataset whose augmentation exercises all persisted RNG streams."""

    def __init__(self) -> None:
        self.x = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
        self.y = (self.x[:, :2] > 1.0).to(dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        x = self.x[index].clone()
        if random.random() < 0.5:
            x = -x
        x += float(np.random.normal(0.0, 0.01))
        x += torch.rand_like(x) * 0.01
        return x, self.y[index]


class _DropoutModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.25), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _loader(seed: int) -> DataLoader:
    return DataLoader(
        _RandomDataset(),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _config(output_dir: Path) -> dict:
    return {
        "output_dir": str(output_dir),
        "training": {
            "epochs": 8,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "threshold": 0.5,
            "amp": False,
            "loss": "bce",
            "scheduler": {
                "name": "reduce_on_plateau",
                "monitor": "macro_auprc",
                "patience": 0,
                "factor": 0.5,
            },
            "early_stopping": {
                "monitor": "macro_auprc",
                "patience": 3,
                "min_delta": 1.0,
                "min_epoch": 1,
            },
            "checkpoint_monitor": "macro_auroc",
        },
    }


def _constant_evaluate(*_args, **_kwargs):
    # Constant validation metrics make the nested early-stop rule deterministic.
    return 0.4, {"macro_auprc": 0.5, "macro_auroc": 0.7, "macro_f1": 0.2}


def _run(output_dir: Path, *, resume_checkpoint=None, max_epochs_this_call=None):
    torch.set_num_threads(1)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model = _DropoutModel()
    loader = _loader(991)
    return fit(
        model,
        loader,
        _loader(992),
        _config(output_dir),
        torch.device("cpu"),
        criterion=nn.BCEWithLogitsLoss(),
        resume_checkpoint=resume_checkpoint,
        max_epochs_this_call=max_epochs_this_call,
    )


def test_unified_fit_resume_is_cpu_exact_with_random_augmentation(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(engine, "evaluate", _constant_evaluate)

    uninterrupted_history = _run(tmp_path / "full")
    full_checkpoint = load_training_checkpoint(tmp_path / "full" / "last.pt")

    _run(tmp_path / "split", max_epochs_this_call=2)
    split_checkpoint = load_training_checkpoint(tmp_path / "split" / "last.pt")
    resumed_history = _run(
        tmp_path / "split",
        resume_checkpoint=split_checkpoint,
        max_epochs_this_call=None,
    )
    resumed_checkpoint = load_training_checkpoint(tmp_path / "split" / "last.pt")

    assert [row["epoch"] for row in uninterrupted_history] == [1, 2, 3, 4]
    assert resumed_history == uninterrupted_history
    assert resumed_checkpoint["epoch"] == full_checkpoint["epoch"] == 4
    assert resumed_checkpoint["history"] == full_checkpoint["history"]
    assert resumed_checkpoint["training_state"] == full_checkpoint["training_state"]
    assert resumed_checkpoint["scheduler_state_dict"] == full_checkpoint["scheduler_state_dict"]
    for key, value in full_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(resumed_checkpoint["model_state_dict"][key], value, rtol=0, atol=0)
    assert resumed_checkpoint["optimizer_state_dict"]["param_groups"] == full_checkpoint["optimizer_state_dict"]["param_groups"]
    for key, state in full_checkpoint["optimizer_state_dict"]["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(resumed_checkpoint["optimizer_state_dict"]["state"][key][field], value, rtol=0, atol=0)


def test_load_training_checkpoint_accepts_new_alias_only_keys(tmp_path: Path):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "alias-only.pt"
    torch.save(
        {
            "epoch": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": [],
        },
        path,
    )
    loaded = load_training_checkpoint(path)
    assert loaded["epoch"] == 2
    assert set(loaded["model_state_dict"]) == set(model.state_dict())


@pytest.mark.parametrize("loss_config", ["asl", {"name": "asymmetric", "gamma_neg": 2, "gamma_pos": 1, "clip": 0.05}])
def test_build_training_criterion_accepts_asl_string_and_mapping(loss_config):
    class _Loader:
        dataset = type("_Dataset", (), {"frame": None, "label_cols": []})()

    criterion, metadata = build_training_criterion(
        _Loader(), {"training": {"loss": loss_config}}, torch.device("cpu")
    )
    assert isinstance(criterion, AsymmetricLoss)
    assert metadata["uses_pos_weight"] is False
