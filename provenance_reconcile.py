"""CLI and ledger projection for the evidence-driven provenance manifest."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from baseline.provenance import build_manifest, sha256_file
from workflow_state import REGISTRY_FIELDS, load_state


def build_reconciliation(repo_root: str | Path = ".", b4_eval_dir: str | Path | None = None, b5_eval_dir: str | Path | None = None, ensemble_dir: str | Path | None = None, b3_test_available: bool = False, **_: Any) -> dict[str, Any]:
    """Compatibility name; the baseline evidence manifest is authoritative.

    Explicit legacy bundle paths are accepted for old callers and are reflected
    in a compatibility status only when their manifests are complete.
    """
    m = build_manifest(repo_root)
    if b4_eval_dir is not None or b5_eval_dir is not None or ensemble_dir is not None:
        # Do not invent metrics; expose only an evidence status for synthetic/legacy callers.
        from baseline.provenance import _bundle
        root = Path(repo_root).resolve(); errs = list(m.get("blocking_errors", [])); fs=set()
        bundles={}
        for key, value in (("B4",b4_eval_dir),("B5",b5_eval_dir)):
            if value is None: continue
            bundles[key] = _bundle(root, (root / value).resolve(), fs, errs)
        ep = (root / ensemble_dir / "ensemble_protocol.json").resolve() if ensemble_dir else None
        proto = json.loads(ep.read_text(encoding="utf-8")) if ep and ep.is_file() else None
        if proto and proto.get("status") == "complete" and proto.get("test_used_for_selection_or_tuning") is False:
            m["status"] = "final_candidate_ready"
        m["bundles"] = bundles
        m["stages"]["B3_test"] = {"status": "complete" if b3_test_available else "pending"}
        m["stages"]["final_test"] = {"status": "complete" if m["status"] == "final_candidate_ready" else "pending"}
        m["stages"]["B5_confirmation"] = {"status": "completed" if m["status"] == "final_candidate_ready" else "pending"}
        m["execution_policy"] = {"allow_implicit_training":False,"allow_stage_start_from_stale_current_stage":False,"manual_approval_required_for_training":True}
    return m


def _rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows=[]
    for run in manifest.get("runs", []):
        if run.get("run") == "ensemble": name="B4B5_ensemble"
        else: name=run.get("run", "")
        tr=run.get("training", {}); cp=tr.get("checkpoint", {}); met=cp.get("metrics", {}) if isinstance(cp,dict) else {}
        test=run.get("test", {}); metrics={}
        if isinstance(test,dict): metrics=test.get("metrics",{}) if isinstance(test.get("metrics"),dict) else {}
        rows.append({"experiment":name,"backbone":(cp.get("config",{}).get("model",{}) or {}).get("name", "") if isinstance(cp,dict) else "","resolution":224,"loss":"","batch":"","lr":"","scheduler":"","epochs_completed":tr.get("history",{}).get("max_epoch", ""),"best_val_macro_auroc":met.get("macro_auroc", ""),"best_val_macro_auprc":met.get("macro_auprc", ""),"test_macro_auroc":metrics.get("macro_auroc", ""),"test_macro_auprc":metrics.get("macro_auprc", ""),"test_macro_f1_tuned":metrics.get("macro_f1_tuned", ""),"status":run.get("status", ""),"notes":"evidence projection; unknown fields remain blank"})
    return rows


def _state(manifest: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    stages = {k:(v.get("status") if isinstance(v,dict) else v) for k,v in manifest.get("stages",{}).items()}
    # workflow_state accepts completed/pending/blocked; retain not_available as pending projection.
    stages = {k:("pending" if v == "not_available" else v) for k,v in stages.items()}
    return {"schema_version":2,"status":manifest.get("status"),"current_stage": next((k for k,v in stages.items() if v != "completed"), "final_test"), **stages,"stages":stages,"last_update":manifest.get("reconciled_at_utc"),"execution_policy":manifest.get("execution_policy",{}),"final_candidate":manifest.get("final_candidate",{}),"reconciliation_manifest":"artifacts/canonical_run_manifest.json","reconciliation_generation":manifest.get("generation_id"),"recovery_count":(previous or {}).get("recovery_count",{}),"night_mode":(previous or {}).get("night_mode",{})}


def check_projection_drift(manifest: dict[str, Any], state_path: str | Path, registry_path: str | Path) -> list[str]:
    drift=[]
    try: current=json.loads(Path(state_path).read_text(encoding="utf-8")); expected=_state(manifest, current)
    except Exception as e: return [f"state unreadable: {e}"]
    for key in ("status","stages","final_candidate"):
        if current.get(key) != expected.get(key): drift.append(f"state projection drift: {key}")
    try:
        with Path(registry_path).open(encoding="utf-8", newline="") as f: actual=list(csv.DictReader(f))
        expected_rows=_rows(manifest)
        if actual != expected_rows: drift.append("registry projection drift")
    except Exception as e: drift.append(f"registry unreadable: {e}")
    return drift


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd,"wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def apply_reconciliation(manifest: dict[str, Any], *, state_path: str | Path, registry_path: str | Path, manifest_path: str | Path, registry_rows: list[dict[str, Any]] | None = None) -> None:
    """Write an immutable generation and atomically replace canonical views."""
    root = Path(manifest_path).resolve().parent.parent
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = dict(manifest); manifest["generation_id"] = generation
    payload=json.dumps(manifest,ensure_ascii=True,indent=2,sort_keys=True).encode()+b"\n"
    gen_path=root / "artifacts" / "ledger_generations" / f"{generation}.json"
    _atomic_bytes(gen_path,payload)
    previous=None
    try: previous=json.loads(Path(state_path).read_text(encoding="utf-8"))
    except Exception: pass
    state=_state(manifest, previous)
    rows=registry_rows if registry_rows is not None else _rows(manifest)
    state_bytes=json.dumps(state,ensure_ascii=True,indent=2,sort_keys=True).encode()+b"\n"
    out=Path(registry_path); sio=[]
    import io
    s=io.StringIO(); w=csv.DictWriter(s,fieldnames=REGISTRY_FIELDS); w.writeheader(); w.writerows(rows); reg_bytes=s.getvalue().encode()
    _atomic_bytes(Path(manifest_path),payload); _atomic_bytes(Path(state_path),state_bytes); _atomic_bytes(out,reg_bytes)


def main() -> None:
    p=argparse.ArgumentParser(description="Read-only provenance audit; --write updates projections atomically.")
    p.add_argument("--root",default="."); p.add_argument("--write",action="store_true"); p.add_argument("--manifest",default="artifacts/canonical_run_manifest.json")
    a=p.parse_args(); m=build_manifest(a.root)
    if a.write: apply_reconciliation(m,state_path=Path(a.root)/"workflow_state.json",registry_path=Path(a.root)/"outputs/experiment_registry.csv",manifest_path=Path(a.root)/a.manifest)
    print(json.dumps(m,ensure_ascii=False,indent=2,sort_keys=True))
if __name__ == "__main__": main()
