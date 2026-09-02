from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from baseline.engine import (
    fit,
    load_training_checkpoint,
    save_checkpoint,
    save_history,
)
from baseline.resources import CommitStats, check_commit_headroom
from smoke_b2_resources import collect_resource_snapshot


def test_checkpoint_is_atomic_and_contains_resume_state(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint_path = tmp_path / "last.pt"
    history = [{"epoch": 1, "train_loss": 0.4, "val_loss": 0.3}]

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        config={"model": {"name": "resnet18"}},
        metrics=history[-1],
        scaler=scaler,
        best_metric=0.7,
        history=history,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert not checkpoint_path.with_name("last.pt.tmp").exists()
    assert {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "best_metric",
        "history",
        "config",
    } <= checkpoint.keys()
    assert checkpoint["epoch"] == 1
    assert checkpoint["history"] == history
    assert checkpoint["best_metric"] == 0.7
    assert checkpoint["model_state"] == checkpoint["model_state_dict"]
    assert checkpoint["optimizer_state"] == checkpoint["optimizer_state_dict"]


def test_history_is_atomically_replaced_each_epoch(tmp_path: Path):
    history_path = tmp_path / "history.csv"

    save_history(history_path, [{"epoch": 1, "train_loss": 0.4}])
    save_history(history_path, [{"epoch": 1, "train_loss": 0.4}, {"epoch": 2, "train_loss": 0.3}])

    assert history_path.read_text(encoding="utf-8").splitlines() == [
        "epoch,train_loss",
        "1,0.4",
        "2,0.3",
    ]
    assert not history_path.with_name("history.csv.tmp").exists()


def test_checkpoint_loader_accepts_legacy_keys_and_rejects_missing_file(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "epoch": 3,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {},
            "metrics": {"macro_auroc": 0.6},
        },
        legacy_path,
    )

    loaded = load_training_checkpoint(legacy_path)

    assert loaded["epoch"] == 3
    torch.testing.assert_close(loaded["model_state_dict"], model.state_dict())
    assert loaded["optimizer_state_dict"].keys() == optimizer.state_dict().keys()
    assert loaded["history"] == []
    with pytest.raises(FileNotFoundError, match="Resume checkpoint does not exist"):
        load_training_checkpoint(tmp_path / "missing.pt")


def test_commit_guard_blocks_low_headroom_and_allows_safe_headroom():
    blocked = check_commit_headroom(
        CommitStats(committed_bytes=98, limit_bytes=100), minimum_headroom_percent=20
    )
    allowed = check_commit_headroom(
        CommitStats(committed_bytes=70, limit_bytes=100), minimum_headroom_percent=20
    )

    assert blocked.passed is False
    assert blocked.headroom_percent == 2.0
    assert allowed.passed is True
    assert allowed.headroom_percent == 30.0


def test_resource_snapshot_has_process_system_commit_and_gpu_fields():
    snapshot = collect_resource_snapshot(torch.device("cpu"))

    assert {"process", "system_ram", "system_commit", "gpu"} <= snapshot.keys()
    assert "rss_bytes" in snapshot["process"]
    assert "allocated_bytes" in snapshot["gpu"]


def test_fit_resumes_from_checkpoint_epoch_plus_one(tmp_path: Path):
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    targets = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    config = {
        "output_dir": str(tmp_path),
        "training": {
            "epochs": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "threshold": 0.5,
            "amp": False,
        },
    }
    model = torch.nn.Linear(2, 2)
    criterion = torch.nn.BCEWithLogitsLoss()
    first_history = fit(model, loader, loader, config, torch.device("cpu"), criterion=criterion)
    checkpoint = load_training_checkpoint(tmp_path / "last.pt")

    config["training"]["epochs"] = 2
    resumed_model = torch.nn.Linear(2, 2)
    resumed_model.load_state_dict(checkpoint["model_state_dict"])
    resumed_history = fit(
        resumed_model,
        loader,
        loader,
        config,
        torch.device("cpu"),
        criterion=criterion,
        start_epoch=checkpoint["epoch"] + 1,
        history=checkpoint["history"],
        best_metric=checkpoint["best_metric"],
        optimizer_state_dict=checkpoint["optimizer_state_dict"],
        scaler_state_dict=checkpoint["scaler_state_dict"],
    )

    assert [row["epoch"] for row in first_history] == [1]
    assert [row["epoch"] for row in resumed_history] == [1, 2]
