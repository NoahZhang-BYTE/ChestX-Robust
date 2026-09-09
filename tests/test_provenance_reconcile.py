import json
import hashlib
from pathlib import Path

from provenance_reconcile import build_reconciliation, apply_reconciliation


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_reconciliation_distinguishes_b3_pending_from_final_ready(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    b4 = artifacts / "b4"
    b5 = artifacts / "b5"
    ensemble = artifacts / "ensemble"
    for directory in (b4, b5):
        _write_json(directory / "evaluation_protocol.json", {
            "status": "frozen",
            "test_used_for_tuning": False,
            "checkpoint": {"epoch": 7, "sha256": "a" * 64},
            "labels_csv": {"sha256": "b" * 64},
            "split_counts": {"train": 1, "val": 1, "test": 1},
        })
        _write_json(directory / "test" / "run_metadata.json", {
            "status": "completed",
            "test_used_for_tuning": False,
        })
        test_dir = directory / "test"
        for name in ("test_per_class_metrics.csv", "test_probabilities.npy", "test_summary.json", "test_targets.npy"):
            (test_dir / name).write_bytes(name.encode("ascii"))
        files = {name: hashlib.sha256((test_dir / name).read_bytes()).hexdigest() for name in (
            "run_metadata.json", "test_per_class_metrics.csv", "test_probabilities.npy", "test_summary.json", "test_targets.npy"
        )}
        _write_json(test_dir / "manifest.json", {"status": "complete", "files": files})
    _write_json(ensemble / "ensemble_protocol.json", {
        "status": "complete",
        "test_used_for_selection_or_tuning": False,
        "selected_weights": {"b4": 0.4, "b5": 0.6},
    })
    _write_json(ensemble / "final_analysis" / "analysis_summary.json", {
        "test_used_for_selection": False,
        "n_test": 1,
    })

    result = build_reconciliation(
        repo_root=tmp_path,
        b4_eval_dir=b4,
        b5_eval_dir=b5,
        ensemble_dir=ensemble,
        b3_test_available=False,
    )

    assert result["status"] == "final_candidate_ready"
    assert result["stages"]["B3_test"] == "pending"
    assert result["stages"]["B4_analysis"] == "completed"
    assert result["stages"]["B5_confirmation"] == "completed"
    assert result["stages"]["final_test"] == "completed"
    assert result["execution_policy"]["allow_implicit_training"] is False


def test_apply_reconciliation_writes_manifest_state_and_registry_atomically(tmp_path: Path):
    manifest = {"status": "final_candidate_ready", "stages": {"B3_test": "pending"}}
    state_path = tmp_path / "workflow_state.json"
    registry_path = tmp_path / "outputs" / "experiment_registry.csv"
    manifest_path = tmp_path / "artifacts" / "canonical_run_manifest.json"

    apply_reconciliation(
        manifest,
        state_path=state_path,
        registry_path=registry_path,
        manifest_path=manifest_path,
        registry_rows=[{"experiment": "B4", "status": "completed"}],
    )

    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "final_candidate_ready"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "final_candidate_ready"
    assert "B4" in registry_path.read_text(encoding="utf-8")
    assert not state_path.with_suffix(".json.tmp").exists()
