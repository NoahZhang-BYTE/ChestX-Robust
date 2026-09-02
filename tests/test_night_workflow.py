from __future__ import annotations

import json

import pandas as pd

from analyze_validation_training import select_validation_checkpoints
from night_workflow import build_b4_config, choose_b4_winner, initialize_night_mode, record_recovery, should_early_stop
from workflow_state import WorkflowState, atomic_write_json
from tune_validation_thresholds import fit_threshold_grid


def _metrics(auprc: float, auroc: float, f1: float = 0.33) -> dict:
    return {"macro_AUPRC": auprc, "macro_AUROC": auroc, "macro_F1_tuned": f1}


def test_b3_wins_when_auprc_gain_meets_threshold_without_excess_auc_loss():
    decision = choose_b4_winner(
        _metrics(0.268, 0.837),
        _metrics(0.271, 0.834),
        {"Pneumonia": {"auprc": 0.050}, "Infiltration": {"auroc": 0.71, "auprc": 0.35}, "Hernia": {"auprc": 0.33}},
        {"Pneumonia": {"auprc": 0.052}, "Infiltration": {"auroc": 0.71, "auprc": 0.35}, "Hernia": {"auprc": 0.33}},
    )

    assert decision["winner"] == "B3_ASL"
    assert decision["rule"] == "macro_auprc_gain"
    assert decision["test_used_for_selection"] is False


def test_b2_wins_when_b3_auprc_declines_materially_without_auc_advantage():
    decision = choose_b4_winner(
        _metrics(0.268, 0.837),
        _metrics(0.264, 0.836),
        {"Pneumonia": {"auprc": 0.050}},
        {"Pneumonia": {"auprc": 0.060}},
    )

    assert decision["winner"] == "B2_sqrt_pos_weight"
    assert decision["rule"] == "macro_auprc_decline"


def test_near_tie_can_select_asl_only_for_predefined_rare_label_benefit():
    decision = choose_b4_winner(
        _metrics(0.268, 0.837, 0.330),
        _metrics(0.269, 0.836, 0.331),
        {"Pneumonia": {"auprc": 0.050}, "Infiltration": {"auroc": 0.71, "auprc": 0.35}, "Hernia": {"auprc": 0.33}},
        {"Pneumonia": {"auprc": 0.061}, "Infiltration": {"auroc": 0.71, "auprc": 0.35}, "Hernia": {"auprc": 0.34}},
    )

    assert decision["winner"] == "B3_ASL"
    assert decision["rule"] == "near_tie_rare_label_benefit"
    assert decision["ASL_RARE_LABEL_BENEFIT"] is True


def test_early_stopping_waits_until_epoch_five():
    assert should_early_stop(epoch=4, stale_epochs=4, patience=4, min_epoch=5) is False
    assert should_early_stop(epoch=5, stale_epochs=4, patience=4, min_epoch=5) is True


def test_workflow_state_migrates_night_stage_and_recovery_fields():
    legacy = {
        "current_stage": "B3_training",
        "stages": {stage: "pending" for stage in WorkflowState.new().stages if stage not in {"B2_vs_B3_validation", "final_test"}},
    }

    state = WorkflowState.from_dict(legacy)

    assert state.stages["B2_vs_B3_validation"] == "pending"
    assert state.stages["final_test"] == "pending"
    assert state.recovery_count == {}


def test_b4_config_preserves_b3_asl_and_changes_only_schedule_controls():
    b3 = {
        "seed": 42,
        "device": "auto",
        "output_dir": "outputs/B3_densenet121_asl",
        "data": {"image_size": 224, "batch_size": 32, "num_workers": 0},
        "model": {"name": "densenet121", "pretrained": True},
        "training": {
            "epochs": 10, "learning_rate": 3e-4, "weight_decay": 1e-4,
            "threshold": 0.5, "amp": True, "scheduler": "none",
            "loss": {"name": "asymmetric", "gamma_neg": 4, "gamma_pos": 1, "clip": 0.05},
        },
    }

    b4 = build_b4_config("B3_ASL", b3)

    assert b4["data"] == b3["data"]
    assert b4["model"] == b3["model"]
    assert b4["training"]["loss"] == b3["training"]["loss"]
    assert b4["training"]["epochs"] == 20
    assert b4["training"]["scheduler"] == "reduce_on_plateau"
    assert b4["training"]["scheduler_monitor"] == "macro_auprc"
    assert b4["training"]["early_stopping_min_epoch"] == 5
    assert b4["output_dir"].endswith("B4_densenet121_asl_scheduler")


def test_night_initialization_persists_recovery_and_whea_baseline_atomically(tmp_path):
    state_path = tmp_path / "workflow_state.json"
    log_path = tmp_path / "NIGHT_WORKFLOW_LOG.md"
    atomic_write_json(state_path, WorkflowState.new().to_dict())

    state = initialize_night_mode(
        state_path,
        log_path,
        started_at="2026-09-01T04:00:00+00:00",
        whea_baseline={"available": True, "latest_record_id": None},
    )

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert state.recovery_count == {"B3_training": 0, "B4_training": 0}
    assert raw["night_mode"]["started_at"] == "2026-09-01T04:00:00+00:00"
    assert raw["night_mode"]["whea_baseline"]["latest_record_id"] is None
    assert "night_mode_initialized" in log_path.read_text(encoding="utf-8")


def test_recovery_record_increments_only_authorized_stage_and_preserves_atomic_state(tmp_path):
    state_path = tmp_path / "workflow_state.json"
    log_path = tmp_path / "NIGHT_WORKFLOW_LOG.md"
    atomic_write_json(state_path, WorkflowState.new().to_dict())
    initialize_night_mode(state_path, log_path, started_at="2026-09-01T04:00:00+00:00")

    state = record_recovery(state_path, log_path, "B3_training", timestamp="2026-09-01T05:00:00+00:00")

    assert state.recovery_count["B3_training"] == 1
    assert state.recovery_count["B4_training"] == 0
    assert "recovery_1_authorized" in log_path.read_text(encoding="utf-8")


def test_validation_checkpoint_selection_prioritizes_auprc_then_records_auroc_candidate():
    history = pd.DataFrame(
        [
            {"epoch": 1, "macro_auprc": 0.21, "macro_auroc": 0.82},
            {"epoch": 2, "macro_auprc": 0.24, "macro_auroc": 0.81},
            {"epoch": 3, "macro_auprc": 0.23, "macro_auroc": 0.83},
        ]
    )
    inventory = [
        {"filename": "best_macro_auprc.pt", "epoch": 2, "macro_auprc": 0.24, "macro_auroc": 0.81, "loadable": True},
        {"filename": "best_macro_auroc.pt", "epoch": 3, "macro_auprc": 0.23, "macro_auroc": 0.83, "loadable": True},
    ]

    selection = select_validation_checkpoints(history, inventory)

    assert selection["selected_filename"] == "best_macro_auprc.pt"
    assert selection["selected_epoch"] == 2
    assert selection["secondary_filename"] == "best_macro_auroc.pt"
    assert selection["test_used_for_selection"] is False


def test_grid_threshold_tuning_uses_only_given_validation_arrays_and_records_fixed_metrics():
    targets = __import__("numpy").array([[1, 0], [1, 0], [0, 1], [0, 1]])
    probabilities = __import__("numpy").array([[0.6, 0.2], [0.4, 0.3], [0.6, 0.7], [0.4, 0.6]])

    result = fit_threshold_grid(targets, probabilities, ["first", "second"])

    assert result["fit_split"] == "val"
    assert result["tie_break"] == "threshold closest to 0.5"
    assert len(result["thresholds"]) == 2
    assert {"fixed_precision", "fixed_recall", "fixed_f1", "tuned_f1", "auroc", "auprc"} <= result["per_label"][0].keys()
