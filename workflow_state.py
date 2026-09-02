"""Restart-safe stage state and experiment registry helpers."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STAGES = (
    "B2_training",
    "B2_analysis",
    "B2_threshold_tuning",
    "B2_test",
    "hard_label_analysis",
    "cooccurrence_analysis",
    "B3_training",
    "B3_analysis",
    "B3_threshold_tuning",
    "B3_test",
    "B2_vs_B3",
    "B2_vs_B3_validation",
    "B4_training",
    "B4_analysis",
    "final_test",
    "B5_confirmation",
)
VALID_STATUSES = {"pending", "running", "completed", "blocked"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: str | Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class WorkflowState:
    current_stage: str
    stages: dict[str, str] = field(default_factory=dict)
    last_update: str = ""
    recovery_count: dict[str, int] = field(default_factory=dict)
    night_mode: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls) -> "WorkflowState":
        return cls(
            current_stage=DEFAULT_STAGES[0],
            stages={stage: "pending" for stage in DEFAULT_STAGES},
            last_update=utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            **self.stages,
            "stages": dict(self.stages),
            "last_update": self.last_update,
            "recovery_count": dict(self.recovery_count),
            "night_mode": dict(self.night_mode),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WorkflowState":
        if not isinstance(raw, dict):
            raise ValueError("workflow state must be a JSON object")
        nested = raw.get("stages")
        stages = dict(nested) if isinstance(nested, dict) else {
            key: value for key, value in raw.items() if key in DEFAULT_STAGES
        }
        # Night-mode stages were added after the original B2/B3 state file.
        # Older state files remain valid and receive pending values atomically.
        for stage in DEFAULT_STAGES:
            stages.setdefault(stage, "pending")
        invalid = {stage: status for stage, status in stages.items() if status not in VALID_STATUSES}
        if invalid:
            raise ValueError(f"workflow state has invalid statuses: {invalid}")
        current_stage = str(raw.get("current_stage", DEFAULT_STAGES[0]))
        if current_stage not in stages:
            raise ValueError(f"Unknown current stage: {current_stage}")
        recovery_count = raw.get("recovery_count", {})
        if not isinstance(recovery_count, dict) or any(
            not isinstance(value, int) or value < 0 for value in recovery_count.values()
        ):
            raise ValueError("workflow recovery_count must be a mapping of non-negative integers")
        night_mode = raw.get("night_mode", {})
        if not isinstance(night_mode, dict):
            raise ValueError("workflow night_mode must be a mapping")
        return cls(
            current_stage,
            stages,
            str(raw.get("last_update", "")),
            dict(recovery_count),
            dict(night_mode),
        )


def load_state(path: str | Path) -> WorkflowState:
    path = Path(path)
    if not path.is_file():
        return WorkflowState.new()
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read workflow state: {path}") from error
    return WorkflowState.from_dict(raw)


def save_state(path: str | Path, state: WorkflowState) -> None:
    atomic_write_json(path, state.to_dict())


def update_stage(path: str | Path, stage: str, status: str) -> WorkflowState:
    if stage not in DEFAULT_STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    if status not in VALID_STATUSES:
        raise ValueError(f"Unknown status: {status}")
    state = load_state(path)
    current = state.stages[stage]
    if current == "completed" and status != "completed":
        raise ValueError(f"Stage {stage} is completed and cannot be reopened")
    if current == "blocked" and status == "pending":
        raise ValueError(f"Stage {stage} is blocked and cannot be reset automatically")
    state.stages[stage] = status
    if status in {"running", "pending"}:
        state.current_stage = stage
    elif status == "completed":
        following = [candidate for candidate in DEFAULT_STAGES if state.stages[candidate] != "completed"]
        if following:
            state.current_stage = following[0]
    state.last_update = utc_now()
    save_state(path, state)
    return state


def update_night_metadata(
    path: str | Path,
    *,
    recovery_count: dict[str, int] | None = None,
    night_mode: dict[str, Any] | None = None,
) -> WorkflowState:
    """Persist unattended-workflow metadata through the same atomic state file."""
    state = load_state(path)
    if recovery_count is not None:
        if any(not isinstance(value, int) or value < 0 for value in recovery_count.values()):
            raise ValueError("recovery_count values must be non-negative integers")
        state.recovery_count = dict(recovery_count)
    if night_mode is not None:
        state.night_mode = dict(night_mode)
    state.last_update = utc_now()
    save_state(path, state)
    return state


REGISTRY_FIELDS = (
    "experiment", "backbone", "resolution", "loss", "batch", "lr", "scheduler",
    "epochs_completed", "best_val_macro_auroc", "best_val_macro_auprc",
    "test_macro_auroc", "test_macro_auprc", "test_macro_f1_tuned", "status", "notes",
)


def ensure_registry(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        return
    atomic_write_csv(path, REGISTRY_FIELDS, [])


def append_registry_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    existing.append({field: row.get(field, "") for field in REGISTRY_FIELDS})
    atomic_write_csv(path, REGISTRY_FIELDS, existing)
