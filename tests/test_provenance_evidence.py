import json
from pathlib import Path

import pytest

from baseline.provenance import build_manifest, safe_relpath


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_missing_artifacts_are_pending_and_manifest_has_evidence(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "labels.csv").write_text("image_id,split,patient_id,Atelectasis\na,train,1,0\n", encoding="utf-8")
    m = build_manifest(tmp_path)
    assert m["schema_version"] >= 2
    assert "audit_context" in m
    assert "B0" in {r["run"] for r in m["runs"]}
    assert m["stages"]["B3_test"]["status"] in {"pending", "not_available"}
    assert "data/labels.csv" in m["evidence"]
    assert m["evidence"]["data/labels.csv"]["size_bytes"] > 0


def test_hash_mismatch_is_blocking(tmp_path):
    d = tmp_path / "artifacts" / "B4_eval" / "test"
    d.mkdir(parents=True)
    (d / "test_targets.npy").write_bytes(b"targets")
    write_json(d / "run_metadata.json", {"status": "completed", "test_used_for_tuning": False})
    write_json(d / "test_summary.json", {"sample_count": 1})
    write_json(d / "test_per_class_metrics.csv", {"rows": []})
    (d / "test_probabilities.npy").write_bytes(b"probs")
    write_json(d / "manifest.json", {"status": "complete", "files": {"test_targets.npy": "0"}})
    m = build_manifest(tmp_path)
    assert any("hash" in x.lower() for x in m["blocking_errors"])
    b4 = next(x for x in m["runs"] if x["run"] == "B4")
    assert b4["status"] in {"blocked", "not_available", "pending"}


def test_manifest_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        safe_relpath(tmp_path, tmp_path / ".." / "outside.txt")


def test_b3_without_test_bundle_is_not_complete(tmp_path):
    out = tmp_path / "outputs" / "B3_densenet121_asl"
    out.mkdir(parents=True)
    (out / "history.csv").write_text("epoch,macro_auroc\n1,0.8\n", encoding="utf-8")
    m = build_manifest(tmp_path)
    b3 = next(x for x in m["runs"] if x["run"] == "B3")
    assert b3["test"]["status"] in {"pending", "not_available"}
    assert m["stages"]["B3_test"]["status"] in {"pending", "not_available"}
