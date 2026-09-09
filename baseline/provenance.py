"""Evidence-driven, read-only provenance reconciliation.

This module only reads metadata, manifests, CSV/JSON/NPY files and selected
checkpoints on CPU.  It never starts training or evaluation workloads.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

SCHEMA_VERSION = 2
LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
MANDATORY_TEST_FILES = ("run_metadata.json", "test_per_class_metrics.csv", "test_probabilities.npy", "test_summary.json", "test_targets.npy")


def _root(path: str | Path) -> Path:
    return Path(path).resolve()


def safe_relpath(root: Path, path: Path) -> str:
    root, path = _root(root), Path(path).resolve()
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {path}") from exc
    return rel.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _git(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"sha": None, "dirty": None}
    for args, key in [((["git", "rev-parse", "HEAD"], "sha")), ((["git", "status", "--porcelain"], "dirty"))]:
        try:
            r = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=10, check=True)
            out[key] = (r.stdout.strip() != "") if key == "dirty" else (r.stdout.strip() or None)
        except (OSError, subprocess.SubprocessError):
            pass
    return out


def _evidence(root: Path, files: set[Path], errors: list[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for p in sorted(files):
        try:
            rel = safe_relpath(root, p)
        except ValueError as e:
            errors.append(str(e)); continue
        if not p.is_file():
            continue
        try:
            result[rel] = {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        except OSError as e:
            errors.append(f"cannot fingerprint {rel}: {e}")
    return result


def _find(root: Path, *parts: str) -> Path:
    p = root.joinpath(*parts)
    return p.resolve()


def _checkpoint(path: Path, root: Path, errors: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"path": None, "status": "missing"}
    if not path.is_file(): return out
    out.update(path=str(path), sha256=sha256_file(path))
    if torch is None:
        out["status"] = "unreadable_no_torch"; return out
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
        out["status"] = "read"
        out["epoch"] = c.get("epoch") if isinstance(c, dict) else None
        cfg = c.get("config") if isinstance(c, dict) else None
        met = c.get("metrics") if isinstance(c, dict) else None
        if isinstance(cfg, dict):
            out["config"] = {"model": cfg.get("model"), "data": cfg.get("data"), "training": cfg.get("training"), "seed": cfg.get("seed")}
        if isinstance(met, dict):
            out["metrics"] = {k: met.get(k) for k in ("macro_auroc", "macro_auprc", "macro_f1")}
    except Exception as e:
        out["status"] = "unreadable"; errors.append(f"checkpoint unreadable {safe_relpath(root, path)}: {e}")
    return out


def _history(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    out = {"status": "missing", "path": str(path) if path.exists() else None}
    if not path.is_file(): return out
    try:
        with path.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
        epochs = [int(float(r["epoch"])) for r in rows if r.get("epoch") not in (None, "")]
        out.update(status="complete" if rows else "empty", epochs=epochs, rows=len(rows), max_epoch=max(epochs) if epochs else None)
        ep = checkpoint.get("epoch")
        if ep is not None: out["checkpoint_epoch_present"] = int(ep) in epochs
    except Exception as e: out.update(status="invalid", error=str(e))
    return out


def _bundle(root: Path, directory: Path, evidence_files: set[Path], errors: list[str]) -> dict[str, Any]:
    labels = root / "data" / "labels.csv"
    rec: dict[str, Any] = {"path": str(directory), "status": "not_available", "test": {"status": "not_available"}}
    protocol_path, manifest_path = directory / "evaluation_protocol.json", directory / "test" / "manifest.json"
    metadata_path = directory / "test" / "run_metadata.json"
    for p in (protocol_path, manifest_path, metadata_path):
        if p.exists(): evidence_files.add(p)
    protocol, manifest, metadata = _json(protocol_path), _json(manifest_path), _json(metadata_path)
    if protocol is not None:
        rec["protocol"] = {"status": protocol.get("status"), "sha256": sha256_file(protocol_path), "test_used_for_tuning": protocol.get("test_used_for_tuning")}
        evidence_files.add(protocol_path)
    if manifest is None:
        return rec
    rec["test"]["manifest_status"] = manifest.get("status")
    declared = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    for name in MANDATORY_TEST_FILES:
        p = directory / "test" / name; evidence_files.add(p)
        if not p.is_file(): errors.append(f"missing mandatory test artifact: {safe_relpath(root,p)}")
    bad = []
    # Verify manifest declarations and protocol links against current files.
    for name, expected in declared.items():
        p = directory / "test" / name; evidence_files.add(p)
        if not p.is_file(): bad.append(f"missing {name}")
        elif expected and sha256_file(p) != expected: bad.append(f"hash mismatch {name}")
    if protocol:
        cp_info = protocol.get("checkpoint") if isinstance(protocol.get("checkpoint"), dict) else {}
        cp_path = cp_info.get("path") if isinstance(cp_info, dict) else None
        cp_sha = cp_info.get("sha256") if isinstance(cp_info, dict) else None
        if cp_path:
            cp = Path(cp_path)
            if not cp.is_absolute(): cp = (root / cp).resolve()
            try:
                safe_relpath(root, cp)
                if not cp.is_file(): bad.append("protocol checkpoint missing")
                elif cp_sha and sha256_file(cp) != cp_sha: bad.append("checkpoint hash mismatch")
                else: evidence_files.add(cp)
            except ValueError: bad.append("protocol checkpoint escapes root")
        labels_info = protocol.get("labels_csv") if isinstance(protocol.get("labels_csv"), dict) else {}
        labels_sha = labels_info.get("sha256") if isinstance(labels_info, dict) else None
        labels_path = labels_info.get("path") if isinstance(labels_info, dict) else None
        if labels_sha and labels.is_file() and sha256_file(labels) != labels_sha: bad.append("labels hash mismatch")
        threshold = protocol.get("threshold") if isinstance(protocol.get("threshold"), dict) else {}
        threshold_sha = threshold.get("sha256") if isinstance(threshold, dict) else None
        threshold_art = threshold.get("artifact") if isinstance(threshold, dict) else None
        if threshold_sha and threshold_art:
            tp = directory / "test" / str(threshold_art)
            if not tp.is_file(): tp = directory / str(threshold_art)
            if not tp.is_file() or sha256_file(tp) != threshold_sha: bad.append("threshold hash mismatch")
    if bad:
        errors.extend([f"{directory.name} test manifest: {x}" for x in bad]); rec["test"]["status"] = "blocked"; return rec
    if manifest.get("status") != "complete" or any(not (directory / "test" / x).is_file() for x in MANDATORY_TEST_FILES):
        rec["test"]["status"] = "pending"; return rec
    rec["test"]["status"] = "complete"
    if metadata: rec["test"]["test_used_for_tuning"] = metadata.get("test_used_for_tuning")
    if np is not None:
        try:
            y = np.load(directory / "test" / "test_targets.npy", mmap_mode="r"); p = np.load(directory / "test" / "test_probabilities.npy", mmap_mode="r")
            rec["test"].update(sample_count=int(y.shape[0]), targets_shape=list(y.shape), probabilities_shape=list(p.shape), probabilities_dtype=str(p.dtype))
            if y.shape != p.shape: errors.append(f"test shape mismatch in {directory.name}: {y.shape} vs {p.shape}")
        except Exception as e: rec["test"]["array_error"] = str(e); errors.append(f"test arrays unreadable {directory.name}: {e}")
    return rec



def _ensemble_source_check(root: Path, ens: Path, proto: dict[str, Any] | None, errors: list[str], files: set[Path]) -> dict[str, Any]:
    """Verify ensemble arrays are reproducible from cached B4/B5 test probabilities."""
    out: dict[str, Any] = {"status": "unknown", "sources": []}
    if not proto or np is None:
        return out
    weights = proto.get("selected_weights") or {}
    try:
        w4, w5 = float(weights["b4"]), float(weights["b5"])
    except Exception:
        errors.append("ensemble selected_weights missing or invalid"); return {"status":"blocked"}
    candidates = {
        "B4": root / "artifacts/B4_densenet121_sqrt_posweight_scheduler_eval_20260904_025154",
        "B5": root / "artifacts/B5_densenet121_asl_eval_20260904_1200",
    }
    arrays = {}
    declared_sources = proto.get("sources") if isinstance(proto.get("sources"), dict) else {}
    for name, d in candidates.items():
        declared_hash = declared_sources.get(f"{name.lower()}_protocol_sha256")
        protocol_path = d / "evaluation_protocol.json"
        if declared_hash and protocol_path.is_file() and sha256_file(protocol_path) != declared_hash:
            errors.append(f"ensemble source {name} protocol hash mismatch")
        elif protocol_path.is_file():
            files.add(protocol_path)

        pp = d / "test" / "test_probabilities.npy"; yp = d / "test" / "test_targets.npy"
        if not pp.is_file() or not yp.is_file():
            errors.append(f"ensemble source {name} test arrays missing"); continue
        files.update({pp, yp})
        try:
            arrays[name] = (np.load(pp, mmap_mode="r"), np.load(yp, mmap_mode="r"))
            out["sources"].append({"run":name,"probabilities":safe_relpath(root,pp),"targets":safe_relpath(root,yp),"probabilities_sha256":sha256_file(pp),"targets_sha256":sha256_file(yp),"dtype":str(arrays[name][0].dtype)})
        except Exception as e:
            errors.append(f"ensemble source {name} arrays unreadable: {e}")
    ep, et = ens / "test_probabilities.npy", ens / "test_targets.npy"
    if len(arrays) != 2 or not ep.is_file() or not et.is_file(): return out
    files.update({ep, et})
    try:
        b4, y4 = arrays["B4"]; b5, y5 = arrays["B5"]; actual = np.load(ep, mmap_mode="r"); yt = np.load(et, mmap_mode="r")
        if y4.shape != y5.shape or not np.array_equal(y4, y5):
            errors.append("ensemble source B4/B5 test targets differ")
        if yt.shape != y4.shape or not np.array_equal(yt, y4):
            errors.append("ensemble targets differ from source targets")
        expected64 = np.asarray(b4, dtype=np.float64) * w4 + np.asarray(b5, dtype=np.float64) * w5
        expected32 = (np.asarray(b4, dtype=np.float32) * np.float32(w4) + np.asarray(b5, dtype=np.float32) * np.float32(w5)).astype(np.float32)
        a = np.asarray(actual)
        d64 = float(np.max(np.abs(a.astype(np.float64) - expected64)))
        d32 = float(np.max(np.abs(a.astype(np.float64) - expected32.astype(np.float64))))
        out.update(status="verified" if d32 == 0.0 or d64 < 1e-6 else "mismatch", weights={"b4":w4,"b5":w5}, shape=list(a.shape), actual_dtype=str(a.dtype), max_abs_diff_float64=d64, max_abs_diff_float32=d32, dtype_discrepancy=abs(d64-d32))
        if out["status"] == "mismatch": errors.append(f"ensemble weighted probabilities mismatch (max abs {min(d32,d64):.3g})")
    except Exception as e:
        errors.append(f"ensemble source verification failed: {e}")
    return out


def _legacy_test(root: Path, run: str, files: set[Path], errors: list[str]) -> dict[str, Any]:
    """Recognize historical B2 output layout without upgrading its provenance."""
    if run != "B2": return {"status": "not_available"}
    d = root / "outputs/B2_densenet121_sqrt_posweight/test"
    yp, yt, met = d / "y_prob_test.npy", d / "y_true_test.npy", d / "test_tuned_metrics.json"
    if not yp.is_file() or not yt.is_file(): return {"status": "not_available", "layout": "legacy_outputs"}
    files.update({yp, yt});
    out = {"status": "complete", "layout": "legacy_outputs", "probabilities": safe_relpath(root, yp), "targets": safe_relpath(root, yt), "probabilities_sha256": sha256_file(yp), "targets_sha256": sha256_file(yt)}
    if met.is_file(): files.add(met); out["metrics_sha256"] = sha256_file(met)
    if np is not None:
        try:
            a,b=np.load(yp,mmap_mode="r"),np.load(yt,mmap_mode="r"); out.update(shape=list(a.shape), targets_shape=list(b.shape), probabilities_dtype=str(a.dtype), sample_count=int(b.shape[0]))
            if a.shape != b.shape: out["status"]="blocked"; errors.append("B2 legacy test target/probability shape mismatch")
        except Exception as e: out["status"]="blocked"; errors.append(f"B2 legacy test arrays unreadable: {e}")
    return out

def build_manifest(root: str | Path = ".") -> dict[str, Any]:
    root = _root(root); errors: list[str] = []; warnings: list[str] = []; files: set[Path] = set()
    labels = root / "data" / "labels.csv"
    if labels.is_file(): files.add(labels)
    split_counts: dict[str, int] = {}; patient_split: dict[str, set[str]] = {}
    if labels.is_file():
        try:
            with labels.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    s = row.get("split", ""); split_counts[s] = split_counts.get(s, 0) + 1
                    pid = row.get("patient_id");
                    if pid is not None: patient_split.setdefault(pid, set()).add(s)
        except Exception as e: errors.append(f"labels.csv unreadable: {e}")
    if any(len(v) > 1 for v in patient_split.values()): errors.append("patient split leakage detected in labels.csv")
    run_specs = {
      "B0": ("outputs/B0_resnet18", "best.pt", "outputs/B0_resnet18/history.csv", None),
      "B1": ("outputs/B1_densenet121", "best.pt", "outputs/B1_densenet121/history.csv", "artifacts/B1_densenet121_eval"),
      "B2": ("outputs/B2_densenet121_sqrt_posweight", "best.pt", "outputs/B2_densenet121_sqrt_posweight/history.csv", "artifacts/B2_densenet121_sqrt_posweight_eval"),
      "B3": ("outputs/B3_densenet121_asl", "best_macro_auprc.pt", "outputs/B3_densenet121_asl/history.csv", None),
      "B4": ("outputs/B4_densenet121_sqrt_posweight_scheduler", "best_macro_auroc.pt", "outputs/B4_densenet121_sqrt_posweight_scheduler/history.csv", "artifacts/B4_densenet121_sqrt_posweight_scheduler_eval_20260904_025154"),
      "B5": ("outputs/B5_densenet121_asl_scheduler_20260904_0326", "best_macro_auprc.pt", "outputs/B5_densenet121_asl_scheduler_20260904_0326/history.csv", "artifacts/B5_densenet121_asl_eval_20260904_1200"),
    }
    runs=[]
    for name, (od, ck, hp, bd) in run_specs.items():
        outdir=root / od; cp=outdir / ck; hist=root / hp
        if cp.exists(): files.add(cp)
        cp_rec = _checkpoint(cp, root, errors)
        hist_rec = _history(hist, cp_rec)
        if hist_rec.get("checkpoint_epoch_present") is False:
            errors.append(f"{name} checkpoint epoch absent from history.csv")
        if hist.exists(): files.add(hist)
        config = outdir / "config_snapshot.yaml"
        if config.exists(): files.add(config)
        if bd:
            candidate = root / bd
            if not candidate.exists():
                pats = {"B1": "artifacts/B1*", "B2": "artifacts/B2*eval*", "B4": "artifacts/B4*eval*", "B5": "artifacts/B5*eval*"}
                matches = sorted(root.glob(pats.get(name, "__none__")))
                candidate = matches[0] if matches else candidate
            if candidate.exists():
                bundle = _bundle(root, candidate, files, errors)
            else:
                bundle = {"path": None, "test": _legacy_test(root, name, files, errors)}
        else:
            legacy = _legacy_test(root, name, files, errors)
            bundle = {"path": None, "test": legacy}
        train_status = "complete" if cp_rec.get("status") == "read" and hist_rec.get("status") == "complete" else ("pending" if hist.exists() else "not_available")
        runs.append({"run": name, "status": "blocked" if cp_rec.get("status") == "unreadable" else train_status, "training": {"status": train_status, "checkpoint": cp_rec, "history": hist_rec}, "test": bundle.get("test", {"status":"not_available"}), "protocol": bundle.get("protocol")})
    ens = root / "artifacts/B4B5_ensemble_20260904_1300"; ep=ens / "ensemble_protocol.json"; er=ens / "ensemble_results.json"; em=ens / "test_manifest.json"; et=ens / "test_probabilities.npy"
    for p in (ep,er,em,et,ens/"test_targets.npy",ens/"thresholds.json"):
        if p.exists(): files.add(p)
    ensemble: dict[str, Any] = {"run":"ensemble", "status":"not_available", "test":{"status":"not_available"}}
    proto,res = _json(ep),_json(er)
    if proto:
        ensemble["protocol"]={"status":proto.get("status"),"sha256":sha256_file(ep),"selected_weights":proto.get("selected_weights"),"test_used_for_selection_or_tuning":proto.get("test_used_for_selection_or_tuning")}
    if em.is_file():
        declared = _json(em) or {}
        if declared.get("status") != "complete":
            errors.append("ensemble test manifest is not complete")
        for name, expected in (declared.get("files") or {}).items():
            p = ens / name; files.add(p)
            if not p.is_file(): errors.append(f"ensemble test manifest missing {name}")
            elif expected and sha256_file(p) != expected: errors.append(f"ensemble test manifest hash mismatch {name}")
    if et.is_file() and (ens/"test_targets.npy").is_file():
        try:
            p=np.load(et,mmap_mode="r"); y=np.load(ens/"test_targets.npy",mmap_mode="r"); ensemble["test"]={"status":"complete" if p.shape==y.shape else "blocked","shape":list(p.shape),"dtype":str(p.dtype),"sample_count":int(y.shape[0])}
            if p.shape!=y.shape: errors.append("ensemble test target/probability shape mismatch")
        except Exception as e: errors.append(f"ensemble arrays unreadable: {e}")
    ensemble["source_verification"] = _ensemble_source_check(root, ens, proto, errors, files)
    # The ensemble keeps its test manifest at the bundle root (legacy layout).
    # The manifest is required to declare the arrays and derived test tables.
    if not em.is_file():
        ensemble["test"]["status"] = "pending"
        warnings.append("ensemble test manifest is missing")
    ensemble["status"] = "complete" if ensemble["test"].get("status")=="complete" and em.is_file() and proto and proto.get("test_used_for_selection_or_tuning") is False and not any("ensemble test manifest" in e for e in errors) else ("blocked" if errors else "pending")
    final = {}
    if res and isinstance(res.get("selected"),dict):
        sel=res["selected"]; final={"experiment":"B4B5_ensemble","selected_weights":proto.get("selected_weights") if proto else None,"test_used_for_selection_or_tuning":proto.get("test_used_for_selection_or_tuning") if proto else None,"metrics":sel.get("test",{}),"status":"ready" if ensemble["status"]=="complete" else "pending"}
    stage_statuses = {"B0_training":runs[0]["training"]["status"],"B1_training":runs[1]["training"]["status"],"B2_training":runs[2]["training"]["status"],"B2_test":runs[2]["test"]["status"],"B3_training":runs[3]["training"]["status"],"B3_test":runs[3]["test"]["status"],"B4_training":runs[4]["training"]["status"],"B4_test":runs[4]["test"]["status"],"B5_training":runs[5]["training"]["status"],"B5_test":runs[5]["test"]["status"],"final_test":ensemble["test"]["status"]}
    stage = {k: {"status": v} for k, v in stage_statuses.items()}
    status = "reconciliation_blocked" if errors else ("final_candidate_ready" if final.get("status")=="ready" else "pending")
    return {"schema_version":SCHEMA_VERSION,"reconciled_at_utc":datetime.now(timezone.utc).isoformat(),"audit_context":{"git":_git(root),"python":sys.version,"platform":platform.platform(),"read_only":True,"training_started":False,"inference_started":False},"evidence":_evidence(root,files,errors),"data":{"labels_csv":safe_relpath(root,labels) if labels.is_file() else None,"labels_sha256":sha256_file(labels) if labels.is_file() else None,"split_counts":split_counts,"patient_split_integrity":"pass" if patient_split and not any(len(v)>1 for v in patient_split.values()) else "unknown"},"runs":runs+[ensemble],"stages":stage,"status":status,"final_candidate":final,"blocking_errors":sorted(set(errors)),"warnings":warnings+["Historical training git/commands/hardware clearance are not proven by current artifacts.","B2_vs_B3 is validation-only evidence; historical B5_confirmation is not approval proof."],"execution_policy":{"allow_implicit_training":False,"allow_stage_start_from_stale_current_stage":False,"manual_approval_required_for_training":True}}
