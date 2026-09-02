import json
from pathlib import Path

import pytest

from workflow_state import (
    DEFAULT_STAGES,
    WorkflowState,
    atomic_write_json,
    load_state,
    update_stage,
)


def test_atomic_json_write_replaces_target_and_cleans_tmp(tmp_path: Path):
    path = tmp_path / "workflow_state.json"

    atomic_write_json(path, {"current_stage": "B2_training"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"current_stage": "B2_training"}
    assert not path.with_name("workflow_state.json.tmp").exists()


def test_new_state_has_all_stages_and_running_stage_can_resume(tmp_path: Path):
    path = tmp_path / "workflow_state.json"
    state = WorkflowState.new()
    atomic_write_json(path, state.to_dict())

    loaded = load_state(path)
    assert set(DEFAULT_STAGES) <= loaded.stages.keys()
    assert loaded.stages["B2_training"] == "pending"
    update_stage(path, "B2_training", "running")
    assert load_state(path).stages["B2_training"] == "running"


def test_completed_stage_cannot_be_reopened_or_skipped(tmp_path: Path):
    path = tmp_path / "workflow_state.json"
    atomic_write_json(path, WorkflowState.new().to_dict())
    update_stage(path, "B2_training", "completed")

    with pytest.raises(ValueError, match="completed"):
        update_stage(path, "B2_training", "running")


def test_unknown_stage_and_status_are_rejected(tmp_path: Path):
    path = tmp_path / "workflow_state.json"
    atomic_write_json(path, WorkflowState.new().to_dict())
    with pytest.raises(ValueError, match="Unknown stage"):
        update_stage(path, "not-a-stage", "running")
    with pytest.raises(ValueError, match="Unknown status"):
        update_stage(path, "B2_training", "started")
