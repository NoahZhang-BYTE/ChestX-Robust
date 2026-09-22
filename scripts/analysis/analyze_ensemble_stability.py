"""Validation-only B4/B5 ensemble stability and complementarity analysis.

This analysis never reads Test inputs. It compares fixed-threshold global and
per-label weights, shrinkage toward a global weight, and paired patient
bootstrap gains. Optional family member lists average seed/TTA probabilities
before the same analysis; a single member is kept as a single-member baseline
and is not presented as independent diversity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from baseline.labels import LABEL_COLUMNS


GLOBAL_WEIGHT = 0.4
DEFAULT_SHRINKAGE = (0.25, 0.5, 0.75)


def _load_array(path: Path, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != len(LABEL_COLUMNS) or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape [N, {len(LABEL_COLUMNS)}], got {array.shape}")
    return array


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_probabilities(name: str, probabilities: np.ndarray, shape: tuple[int, int]) -> None:
    if probabilities.shape != shape or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"Invalid {name}: expected {shape}, got {probabilities.shape}")


def _load_family(paths: Iterable[Path], targets: np.ndarray, family: str) -> tuple[np.ndarray, list[str]]:
    members = list(paths)
    if not members:
        raise ValueError(f"{family} requires at least one probability member")
    arrays = []
    for path in members:
        array = _load_array(path, f"{family} probabilities")
        _validate_probabilities(str(path), array, targets.shape)
        arrays.append(array.astype(np.float64))
    return np.mean(np.stack(arrays, axis=0), axis=0), [str(path.resolve()) for path in members]


def _thresholds(path: Path) -> np.ndarray:
    artifact = _load_json(path)
    if artifact.get("source") not in (None, "validation"):
        raise ValueError("Threshold artifact must be validation-sourced")
    if artifact.get("label_cols", list(LABEL_COLUMNS)) != list(LABEL_COLUMNS):
        raise ValueError("Threshold label order differs from canonical label order")
    values = np.asarray(artifact.get("thresholds"), dtype=np.float64)
    if values.shape != (len(LABEL_COLUMNS),) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Threshold artifact has invalid values")
    return values


def _validation_patient_ids(labels_csv: Path, targets: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(labels_csv)
    if "split" not in frame or "patient_id" not in frame:
        raise ValueError("labels CSV must contain split and patient_id columns for patient bootstrap")
    validation = frame.loc[frame["split"].eq("val")].reset_index(drop=True)
    if len(validation) != len(targets):
        raise ValueError(f"Validation metadata rows {len(validation)} do not match predictions {len(targets)}")
    metadata_targets = validation.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.int64)
    if not np.array_equal(metadata_targets, targets.astype(np.int64)):
        raise ValueError("Validation target ordering does not match labels CSV validation ordering")
    patient_ids = validation["patient_id"].to_numpy()
    if pd.isna(patient_ids).any():
        raise ValueError("Validation patient_id contains missing values")
    return patient_ids


def _groups(patient_ids: np.ndarray) -> list[np.ndarray]:
    unique = pd.unique(patient_ids)
    return [np.flatnonzero(patient_ids == patient) for patient in unique]


def _metrics(targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    if weights is None:
        weights = np.ones(len(targets), dtype=np.float64)
    valid_auroc: list[float] = []
    valid_auprc: list[float] = []
    for column in range(targets.shape[1]):
        target = targets[:, column]
        probability = probabilities[:, column]
        if np.unique(target).size < 2:
            continue
        valid_auroc.append(float(roc_auc_score(target, probability, sample_weight=weights)))
        valid_auprc.append(float(average_precision_score(target, probability, sample_weight=weights)))
    predictions = probabilities >= thresholds[None, :]
    weighted_f1 = np.mean([
        f1_score(targets[:, column], predictions[:, column], sample_weight=weights, zero_division=0)
        for column in range(targets.shape[1])
    ])
    return {
        "macro_auroc": float(np.mean(valid_auroc)) if valid_auroc else float("nan"),
        "macro_auprc": float(np.mean(valid_auprc)) if valid_auprc else float("nan"),
        "macro_f1": float(weighted_f1),
        "micro_auprc": float(average_precision_score(targets.ravel(), probabilities.ravel(), sample_weight=np.repeat(weights, targets.shape[1]))),
        "micro_f1": float(f1_score(targets, predictions, average="micro", sample_weight=weights, zero_division=0)),
    }


def _candidate_probabilities(b4: np.ndarray, b5: np.ndarray, per_class: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    global_half = np.full(len(LABEL_COLUMNS), 0.5, dtype=np.float64)
    global_current = np.full(len(LABEL_COLUMNS), GLOBAL_WEIGHT, dtype=np.float64)
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "global_0.5": (global_half * b4 + (1.0 - global_half) * b5, global_half),
        "global_0.4": (global_current * b4 + (1.0 - global_current) * b5, global_current),
        "per_class": (per_class * b4 + (1.0 - per_class) * b5, per_class),
    }
    for shrinkage in DEFAULT_SHRINKAGE:
        weights = GLOBAL_WEIGHT + shrinkage * (per_class - GLOBAL_WEIGHT)
        candidates[f"per_class_shrinkage_{shrinkage:.2f}"] = (weights * b4 + (1.0 - weights) * b5, weights)
    return candidates


def _complementarity(targets: np.ndarray, b4: np.ndarray, b5: np.ndarray, candidate: np.ndarray, reference: np.ndarray, thresholds: np.ndarray) -> list[dict[str, Any]]:
    b4_pred = b4 >= thresholds[None, :]
    b5_pred = b5 >= thresholds[None, :]
    candidate_pred = candidate >= thresholds[None, :]
    reference_pred = reference >= thresholds[None, :]
    rows = []
    for column, label in enumerate(LABEL_COLUMNS):
        target = targets[:, column].astype(bool)
        b4_correct = b4_pred[:, column] == target
        b5_correct = b5_pred[:, column] == target
        candidate_correct = candidate_pred[:, column] == target
        reference_correct = reference_pred[:, column] == target
        rows.append({
            "label": label,
            "validation_count": int(len(target)),
            "b4_correct": int(b4_correct.sum()),
            "b5_correct": int(b5_correct.sum()),
            "both_wrong": int((~b4_correct & ~b5_correct).sum()),
            "b4_wrong_b5_correct": int((~b4_correct & b5_correct).sum()),
            "b4_correct_b5_wrong": int((b4_correct & ~b5_correct).sum()),
            "both_correct": int((b4_correct & b5_correct).sum()),
            "b4_b5_disagreement_rate": float(np.mean(b4_pred[:, column] != b5_pred[:, column])),
            "reference_correct": int(reference_correct.sum()),
            "candidate_correct": int(candidate_correct.sum()),
            "candidate_correct_when_both_wrong": int((candidate_correct & ~b4_correct & ~b5_correct).sum()),
            "candidate_delta_correct_vs_reference": int(candidate_correct.sum() - reference_correct.sum()),
        })
    return rows


def _bootstrap(
    targets: np.ndarray,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    thresholds: np.ndarray,
    groups: list[np.ndarray],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = list(candidates)
    metrics = ("macro_auroc", "macro_auprc", "macro_f1")
    baseline_name = "global_0.4"
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for draw in range(draws):
        selected = rng.integers(0, len(groups), size=len(groups))
        frequency = np.bincount(np.concatenate([groups[index] for index in selected]), minlength=len(targets)).astype(np.float64)
        draw_metrics = {name: _metrics(targets, probabilities, thresholds, frequency) for name, (probabilities, _weights) in candidates.items()}
        for name in names:
            item = {"draw": draw, "candidate": name}
            item.update(draw_metrics[name])
            for metric in metrics:
                item[f"delta_vs_{baseline_name}_{metric}"] = draw_metrics[name][metric] - draw_metrics[baseline_name][metric]
            raw.append(item)
    raw_frame = pd.DataFrame(raw)
    for name in names:
        subset = raw_frame.loc[raw_frame["candidate"].eq(name)]
        for metric in metrics:
            values = subset[metric].to_numpy(dtype=float)
            delta = subset[f"delta_vs_{baseline_name}_{metric}"].to_numpy(dtype=float)
            rows.append({
                "candidate": name,
                "metric": metric,
                "bootstrap_mean": float(np.nanmean(values)),
                "bootstrap_std": float(np.nanstd(values, ddof=1)),
                "ci_lower_95": float(np.nanpercentile(values, 2.5)),
                "ci_upper_95": float(np.nanpercentile(values, 97.5)),
                "delta_mean_vs_global_0.4": float(np.nanmean(delta)),
                "delta_std_vs_global_0.4": float(np.nanstd(delta, ddof=1)),
                "delta_ci_lower_95_vs_global_0.4": float(np.nanpercentile(delta, 2.5)),
                "delta_ci_upper_95_vs_global_0.4": float(np.nanpercentile(delta, 97.5)),
                "probability_delta_positive": float(np.mean(delta > 0)),
            })
    return pd.DataFrame(rows), raw_frame


def run_analysis(
    b4_probs: Path,
    b5_probs: Path,
    val_targets: Path,
    labels_csv: Path,
    thresholds: Path,
    per_class_weights_path: Path,
    output_dir: Path,
    b4_members: list[Path] | None = None,
    b5_members: list[Path] | None = None,
    bootstrap_draws: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    targets = _load_array(val_targets, "validation targets")
    b4, b4_sources = _load_family(b4_members or [b4_probs], targets, "B4 family")
    b5, b5_sources = _load_family(b5_members or [b5_probs], targets, "B5 family")
    _validate_probabilities("B4 family", b4, targets.shape)
    _validate_probabilities("B5 family", b5, targets.shape)
    threshold_values = _thresholds(thresholds)
    weight_frame = pd.read_csv(per_class_weights_path)
    if list(weight_frame["label"]) != list(LABEL_COLUMNS):
        raise ValueError("Per-class weight label order differs from canonical order")
    per_class_weights = weight_frame["b4_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(per_class_weights).all() or ((per_class_weights < 0) | (per_class_weights > 1)).any():
        raise ValueError("Per-class B4 weights must be finite values in [0, 1]")
    patient_ids = _validation_patient_ids(labels_csv, targets)
    candidates = _candidate_probabilities(b4, b5, per_class_weights)
    metric_rows = []
    for name, (probabilities, weights) in candidates.items():
        row = {"candidate": name, "b4_weight_mean": float(np.mean(weights)), **_metrics(targets, probabilities, threshold_values)}
        metric_rows.append(row)
    bootstrap_summary, bootstrap_raw = _bootstrap(targets, candidates, threshold_values, _groups(patient_ids), bootstrap_draws, seed)
    complementarity = _complementarity(targets, b4, b5, candidates["per_class"][0], candidates["global_0.4"][0], threshold_values)
    pd.DataFrame(metric_rows).to_csv(output_dir / "validation_candidate_metrics.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "patient_bootstrap_summary.csv", index=False)
    bootstrap_raw.to_csv(output_dir / "patient_bootstrap_draws.csv", index=False)
    pd.DataFrame(complementarity).to_csv(output_dir / "complementarity_by_label.csv", index=False)
    (output_dir / "family_members.json").write_text(json.dumps({"B4": b4_sources, "B5": b5_sources, "interpretation": "family probabilities are arithmetic means; one source per family is a single-member baseline, not an independent diversity result"}, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "status": "complete",
        "analysis": "validation_only_ensemble_stability",
        "selection_or_tuning_split": "val",
        "test_read": False,
        "thresholds_fixed": True,
        "threshold_policy": "one fixed validation threshold array reused by every weight candidate; thresholds are never refit per candidate",
        "complementarity_note": "a convex B4/B5 blend with a single fixed threshold cannot flip a row where both models agree, so candidate_correct_when_both_wrong is structurally zero and is reported only as a diagnostic",
        "threshold_source": str(thresholds.resolve()),
        "weight_source": str(per_class_weights_path.resolve()),
        "labels_csv_sha256": _sha256(labels_csv),
        "bootstrap": {"unit": "patient", "draws": bootstrap_draws, "seed": seed, "patient_count": len(_groups(patient_ids))},
        "families": {"B4": b4_sources, "B5": b5_sources},
        "candidate_names": list(candidates),
    }
    (output_dir / "analysis_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir), "candidate_metrics": metric_rows, "bootstrap_summary": bootstrap_summary.to_dict(orient="records"), "complementarity": complementarity}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b4-probs", type=Path, required=True)
    parser.add_argument("--b5-probs", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--per-class-weights", type=Path, required=True, dest="per_class_weights_path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b4-member", type=Path, action="append", dest="b4_members")
    parser.add_argument("--b5-member", type=Path, action="append", dest="b5_members")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("--bootstrap-draws must be positive")
    print(json.dumps(run_analysis(**vars(args)), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
