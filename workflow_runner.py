"""Command-line controller for the restart-safe experiment workflow.

The runner never starts formal training implicitly. It observes the current B2
process, validates stage artifacts, applies safety gates, and only runs an
explicitly requested post-processing command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from baseline.engine import load_training_checkpoint
from baseline.resources import check_commit_headroom, get_system_commit
from workflow_state import (
    DEFAULT_STAGES,
    REGISTRY_FIELDS,
    WorkflowState,
    atomic_write_json,
    ensure_registry,
    load_state,
    update_stage,
)


STATE_PATH = Path("workflow_state.json")
REGISTRY_PATH = Path("outputs/experiment_registry.csv")
B2_OUTPUT = Path("outputs/B2_densenet121_sqrt_posweight")


def detect_b2_process() -> list[int]:
    """Return Python PIDs whose command line identifies the B2 formal run."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        pids = []
        for process in psutil.process_iter(("pid", "name", "cmdline")):
            try:
                cmdline = " ".join(process.info.get("cmdline") or []).lower()
            except (psutil.Error, OSError):
                continue
            if "formal_train.py" in cmdline and "formal_b2_densenet121_sqrt_posweight" in cmdline:
                pids.append(int(process.info["pid"]))
        return pids
    if os.name != "nt":
        return []
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match '(?i)formal_train\\.py' "
        "-and $_.CommandLine -match '(?i)formal_b2_densenet121_sqrt_posweight' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def b2_completion_gate(output_dir: Path = B2_OUTPUT) -> dict[str, Any]:
    history_path = output_dir / "history.csv"
    checkpoint_path = output_dir / "last.pt"
    if not history_path.is_file() or not checkpoint_path.is_file():
        return {"passed": False, "reason": "history.csv or last.pt is missing"}
    try:
        import pandas as pd

        history = pd.read_csv(history_path)
        checkpoint = load_training_checkpoint(checkpoint_path)
    except Exception as error:
        return {"passed": False, "reason": f"artifact read failed: {error}"}
    metrics = ["train_loss", "val_loss", "macro_auroc", "macro_auprc", "macro_f1"]
    finite = all(column in history and np.isfinite(history[column].to_numpy(dtype=float)).all() for column in metrics)
    passed = bool(not history.empty and int(history["epoch"].max()) == 10 and int(checkpoint["epoch"]) == 10 and finite)
    return {
        "passed": passed,
        "history_epoch": int(history["epoch"].max()) if not history.empty else None,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "finite_metrics": finite,
        "reason": None if passed else "requires epoch 10, readable checkpoint, and finite history",
    }


def safety_gate() -> dict[str, Any]:
    try:
        result = check_commit_headroom(get_system_commit(), 20.0)
    except OSError as error:
        return {"passed": False, "reason": str(error), "whea": "unknown"}
    return {
        "passed": result.passed,
        "committed_bytes": result.stats.committed_bytes,
        "limit_bytes": result.stats.limit_bytes,
        "headroom_bytes": result.stats.headroom_bytes,
        "headroom_percent": result.headroom_percent,
        "whea": "not checked by this portable gate",
    }


def recent_whea_events(since_iso: str | None = None) -> dict[str, Any]:
    """Read recent WHEA-Logger 18/19 records when Windows Event Log is available."""
    if os.name != "nt":
        return {"available": False, "events": [], "reason": "not Windows"}
    filter_clause = ""
    if since_iso:
        filter_clause = f" | Where-Object {{ $_.TimeCreated -ge [datetime]::Parse('{since_iso}') }}"
    script = (
        "$events = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; Id=18,19} "
        f"-ErrorAction SilentlyContinue{filter_clause}; "
        "$events | Select-Object Id,TimeCreated,LevelDisplayName,Message | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "events": [], "reason": str(error)}
    if not result.stdout.strip():
        return {"available": True, "events": []}
    try:
        events = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"available": False, "events": [], "reason": "invalid PowerShell JSON"}
    if isinstance(events, dict):
        events = [events]
    return {"available": True, "events": events if isinstance(events, list) else []}


def initialize() -> WorkflowState:
    state = load_state(STATE_PATH)
    atomic_write_json(STATE_PATH, state.to_dict())
    ensure_registry(REGISTRY_PATH)
    return state


def run_explicit_post_b2() -> None:
    gate = b2_completion_gate()
    if not gate["passed"]:
        raise RuntimeError(f"[BLOCKED] B2 completion gate failed: {gate['reason']}")
    commands = [
        [sys.executable, "analyze_b2_training.py"],
        [sys.executable, "tune_b2_thresholds.py"],
        [sys.executable, "evaluate_b2_frozen_test.py"],
        [sys.executable, "analyze_hard_labels.py"],
        [sys.executable, "analyze_cooccurrence.py"],
    ]
    for command in commands:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and advance the experiment workflow.")
    parser.add_argument("command", choices=("init", "status", "b2-gate", "safety-gate", "post-b2"))
    args = parser.parse_args()
    if args.command == "init":
        state = initialize()
        print(json.dumps(state.to_dict(), ensure_ascii=True, indent=2))
    elif args.command == "status":
        state = load_state(STATE_PATH)
        print(json.dumps({"state": state.to_dict(), "b2_pids": detect_b2_process(), "b2_gate": b2_completion_gate(), "safety_gate": safety_gate()}, ensure_ascii=True, indent=2))
    elif args.command == "b2-gate":
        gate = b2_completion_gate()
        if gate["passed"]:
            update_stage(STATE_PATH, "B2_training", "completed")
        print(json.dumps(gate, ensure_ascii=True, indent=2))
    elif args.command == "safety-gate":
        gate = safety_gate()
        print(json.dumps(gate, ensure_ascii=True, indent=2))
        if not gate["passed"]:
            raise SystemExit(2)
    else:
        run_explicit_post_b2()


if __name__ == "__main__":
    main()
