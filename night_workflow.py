"""Pure validation-only decision rules shared by the unattended workflow."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

from workflow_state import WorkflowState, load_state, update_night_metadata


def _atomic_append_log(path: Path, line: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Night Workflow Log\n\n"
    temporary = path.with_name(f"{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(existing)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_night_mode(
    state_path: str | Path,
    log_path: str | Path,
    *,
    started_at: str | None = None,
    whea_baseline: Mapping[str, Any] | None = None,
) -> WorkflowState:
    """Initialize once, preserving existing recovery counters and state stages."""
    state_path, log_path = Path(state_path), Path(log_path)
    state = load_state(state_path)
    if state.night_mode.get("initialized"):
        return state
    timestamp = started_at or datetime.now(timezone.utc).isoformat()
    counts = {"B3_training": 0, "B4_training": 0, **state.recovery_count}
    night_mode = {
        **state.night_mode,
        "initialized": True,
        "started_at": timestamp,
        "whea_baseline": dict(whea_baseline or {"available": False, "latest_record_id": None}),
        "last_observed_epoch": {"B3_training": 0, "B4_training": 0},
        "test_forbidden": True,
        "b5_forbidden": True,
    }
    state = update_night_metadata(state_path, recovery_count=counts, night_mode=night_mode)
    _atomic_append_log(log_path, f"- {timestamp} | night_mode | night_mode_initialized | PASS\n")
    return state


def record_observed_epoch(
    state_path: str | Path,
    log_path: str | Path,
    stage: str,
    epoch: int,
    *,
    timestamp: str | None = None,
) -> bool:
    """Persist and log a newly completed epoch once, never a heartbeat duplicate."""
    if epoch < 1:
        raise ValueError("epoch must be positive")
    state_path, log_path = Path(state_path), Path(log_path)
    state = load_state(state_path)
    night_mode = dict(state.night_mode)
    observed = dict(night_mode.get("last_observed_epoch", {}))
    if int(observed.get(stage, 0)) >= epoch:
        return False
    observed[stage] = epoch
    night_mode["last_observed_epoch"] = observed
    update_night_metadata(state_path, recovery_count=state.recovery_count, night_mode=night_mode)
    event_time = timestamp or datetime.now(timezone.utc).isoformat()
    _atomic_append_log(log_path, f"- {event_time} | {stage} | epoch_{epoch}_completed | observed from history.csv\n")
    return True


def record_recovery(
    state_path: str | Path,
    log_path: str | Path,
    stage: str,
    *,
    timestamp: str | None = None,
    maximum_recoveries: int = 2,
) -> WorkflowState:
    """Authorize at most two explicit resumes for one formal training stage."""
    if stage not in {"B3_training", "B4_training"}:
        raise ValueError(f"Recovery is not authorized for stage: {stage}")
    state_path, log_path = Path(state_path), Path(log_path)
    state = load_state(state_path)
    counts = dict(state.recovery_count)
    count = int(counts.get(stage, 0))
    if count >= maximum_recoveries:
        raise RuntimeError(f"{stage} has already reached the recovery limit ({maximum_recoveries})")
    counts[stage] = count + 1
    state = update_night_metadata(state_path, recovery_count=counts, night_mode=state.night_mode)
    event_time = timestamp or datetime.now(timezone.utc).isoformat()
    _atomic_append_log(log_path, f"- {event_time} | {stage} | recovery_{counts[stage]}_authorized | resume from validated last.pt\n")
    return state


def should_early_stop(epoch: int, stale_epochs: int, patience: int, min_epoch: int = 5) -> bool:
    """Apply B4's validation-AUPRC early-stop rule without a warm-up violation."""
    if epoch < 1 or stale_epochs < 0 or patience < 1 or min_epoch < 1:
        raise ValueError("epoch, stale_epochs, patience, and min_epoch must be positive")
    return epoch >= min_epoch and stale_epochs >= patience


def _metric(metrics: Mapping[str, Any], key: str) -> float:
    value = float(metrics[key])
    if value != value:  # NaN without importing another numerical dependency.
        raise ValueError(f"{key} must be finite")
    return value


def _label_metric(metrics: Mapping[str, Mapping[str, Any]], label: str, key: str) -> float | None:
    row = metrics.get(label)
    if row is None or key not in row:
        return None
    return _metric(row, key)


def choose_b4_winner(
    b2_metrics: Mapping[str, Any],
    b3_metrics: Mapping[str, Any],
    b2_per_label: Mapping[str, Mapping[str, Any]],
    b3_per_label: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the user's fixed validation-only B2/B3 selection protocol."""
    b2_auprc, b3_auprc = _metric(b2_metrics, "macro_AUPRC"), _metric(b3_metrics, "macro_AUPRC")
    b2_auroc, b3_auroc = _metric(b2_metrics, "macro_AUROC"), _metric(b3_metrics, "macro_AUROC")
    b2_f1, b3_f1 = _metric(b2_metrics, "macro_F1_tuned"), _metric(b3_metrics, "macro_F1_tuned")
    delta_auprc, delta_auroc = b3_auprc - b2_auprc, b3_auroc - b2_auroc
    pneumonia_b2 = _label_metric(b2_per_label, "Pneumonia", "auprc")
    pneumonia_b3 = _label_metric(b3_per_label, "Pneumonia", "auprc")
    pneumonia_relative = None if pneumonia_b2 in (None, 0.0) or pneumonia_b3 is None else (pneumonia_b3 - pneumonia_b2) / pneumonia_b2
    rare_benefit = bool(pneumonia_relative is not None and pneumonia_relative >= 0.20 and delta_auprc >= 0.0)

    if delta_auprc >= 0.003 - 1e-12 and delta_auroc >= -0.003 - 1e-12:
        winner, rule = "B3_ASL", "macro_auprc_gain"
    elif delta_auprc <= -0.003 + 1e-12 and delta_auroc <= 0.0:
        winner, rule = "B2_sqrt_pos_weight", "macro_auprc_decline"
    elif abs(delta_auprc) < 0.003 + 1e-12:
        if rare_benefit and delta_auroc >= -0.003 - 1e-12:
            winner, rule = "B3_ASL", "near_tie_rare_label_benefit"
        else:
            winner, rule = "B2_sqrt_pos_weight", "near_tie_stability"
    else:
        winner, rule = "B2_sqrt_pos_weight", "conservative_validation_fallback"
    return {
        "winner": winner,
        "rule": rule,
        "near_tie": abs(delta_auprc) < 0.003 + 1e-12,
        "B2": {"macro_AUPRC": b2_auprc, "macro_AUROC": b2_auroc, "macro_F1_tuned": b2_f1},
        "B3": {"macro_AUPRC": b3_auprc, "macro_AUROC": b3_auroc, "macro_F1_tuned": b3_f1},
        "delta_B3_minus_B2": {"macro_AUPRC": delta_auprc, "macro_AUROC": delta_auroc, "macro_F1_tuned": b3_f1 - b2_f1},
        "Pneumonia_AUPRC": {"B2": pneumonia_b2, "B3": pneumonia_b3, "relative_change": pneumonia_relative},
        "ASL_RARE_LABEL_BENEFIT": rare_benefit,
        "test_used_for_selection": False,
    }


def build_b4_config(winner: str, winner_config: Mapping[str, Any]) -> dict[str, Any]:
    """Derive B4 only by adding the approved schedule and early-stop controls."""
    if winner not in {"B2_sqrt_pos_weight", "B3_ASL"}:
        raise ValueError(f"Unsupported B4 winner: {winner}")
    config = deepcopy(dict(winner_config))
    loss = deepcopy(dict(config["training"]["loss"]))
    suffix = "sqrt_posweight" if winner == "B2_sqrt_pos_weight" else "asl"
    config["output_dir"] = f"outputs/B4_densenet121_{suffix}_scheduler"
    config["training"].update(
        {
            "epochs": 20,
            "scheduler": "reduce_on_plateau",
            "scheduler_monitor": "macro_auprc",
            "scheduler_factor": 0.5,
            "scheduler_patience": 2,
            "min_learning_rate": 1e-6,
            "early_stopping_patience": 4,
            "early_stopping_min_delta": 1e-4,
            "early_stopping_min_epoch": 5,
            "loss": loss,
        }
    )
    return config
