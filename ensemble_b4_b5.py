"""Validation-selected probability ensemble for frozen B4 and B5 outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from baseline.evaluation import (
    compute_auc_metrics,
    compute_f1_metrics,
    compute_per_class_metrics,
    find_per_class_thresholds,
)
from baseline.labels import LABEL_COLUMNS


WEIGHTS = (0.7, 0.6, 0.5, 0.4, 0.3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _validate_bundle(directory: Path) -> dict[str, Any]:
    protocol = _read_json(directory / "evaluation_protocol.json")
    if protocol.get("status") != "frozen":
        raise ValueError(f"Protocol is not frozen: {directory}")
    if protocol.get("test_used_for_tuning") is not False:
        raise ValueError(f"Protocol used test for tuning: {directory}")
    if protocol.get("label_cols") != list(LABEL_COLUMNS):
        raise ValueError(f"Unexpected label order: {directory}")
    if protocol.get("split_counts") != {"train": 89789, "val": 11348, "test": 10983}:
        raise ValueError(f"Unexpected split counts: {directory}")
    return protocol


def _metrics(targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    auc = compute_auc_metrics(targets, probabilities, LABEL_COLUMNS)
    fixed = compute_f1_metrics(targets, probabilities, 0.5)
    tuned = compute_f1_metrics(targets, probabilities, thresholds)
    return {
        "macro_auroc": auc["macro_auroc"],
        "micro_auroc": auc["micro_auroc"],
        "macro_auprc": auc["macro_auprc"],
        "micro_auprc": auc["micro_auprc"],
        "macro_f1_at_0.5": fixed["macro_f1"],
        "micro_f1_at_0.5": fixed["micro_f1"],
        "sample_f1_at_0.5": fixed["sample_f1"],
        "macro_f1_tuned": tuned["macro_f1"],
        "micro_f1_tuned": tuned["micro_f1"],
        "sample_f1_tuned": tuned["sample_f1"],
        "per_class_auc": auc["per_class"],
    }


def _write_per_class_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(b4_dir: Path, b5_dir: Path, output_dir: Path) -> dict[str, Any]:
    b4_dir = b4_dir.resolve()
    b5_dir = b5_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    b4_protocol = _validate_bundle(b4_dir)
    b5_protocol = _validate_bundle(b5_dir)
    if b4_protocol["label_cols"] != b5_protocol["label_cols"]:
        raise ValueError("B4/B5 label order differs")
    if b4_protocol["split_counts"] != b5_protocol["split_counts"]:
        raise ValueError("B4/B5 split counts differ")

    b4_val_targets = np.load(b4_dir / "validation_targets.npy")
    b5_val_targets = np.load(b5_dir / "validation_targets.npy")
    b4_test_targets = np.load(b4_dir / "test" / "test_targets.npy")
    b5_test_targets = np.load(b5_dir / "test" / "test_targets.npy")
    if not np.array_equal(b4_val_targets, b5_val_targets):
        raise ValueError("B4/B5 validation targets or ordering differ")
    if not np.array_equal(b4_test_targets, b5_test_targets):
        raise ValueError("B4/B5 test targets or ordering differ")

    b4_val = np.load(b4_dir / "validation_probabilities.npy").astype(np.float64)
    b5_val = np.load(b5_dir / "validation_probabilities.npy").astype(np.float64)
    b4_test = np.load(b4_dir / "test" / "test_probabilities.npy").astype(np.float64)
    b5_test = np.load(b5_dir / "test" / "test_probabilities.npy").astype(np.float64)
    for name, array, expected in (
        ("b4_val", b4_val, b4_val_targets.shape),
        ("b5_val", b5_val, b4_val_targets.shape),
        ("b4_test", b4_test, b4_test_targets.shape),
        ("b5_test", b5_test, b4_test_targets.shape),
    ):
        if array.shape != expected or not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
            raise ValueError(f"Invalid {name} probability array: {array.shape}")

    rows: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    for b4_weight in WEIGHTS:
        b5_weight = 1.0 - b4_weight
        val_prob = b4_weight * b4_val + b5_weight * b5_val
        thresholds = find_per_class_thresholds(b4_val_targets, val_prob)
        metrics = _metrics(b4_val_targets, val_prob, thresholds)
        key = f"{b4_weight:.1f}_{b5_weight:.1f}"
        candidate = {
            "b4_weight": b4_weight,
            "b5_weight": b5_weight,
            "thresholds": thresholds,
            "metrics": metrics,
        }
        candidates[key] = candidate
        rows.append({
            "b4_weight": b4_weight,
            "b5_weight": b5_weight,
            **{k: v for k, v in metrics.items() if k != "per_class_auc"},
        })

    selected_key = max(candidates, key=lambda key: (candidates[key]["metrics"]["macro_auroc"], candidates[key]["metrics"]["macro_auprc"]))
    selected = candidates[selected_key]
    selected_val_prob = selected["b4_weight"] * b4_val + selected["b5_weight"] * b5_val
    selected_test_prob = selected["b4_weight"] * b4_test + selected["b5_weight"] * b5_test
    thresholds = np.asarray(selected["thresholds"], dtype=np.float64)
    val_metrics = _metrics(b4_val_targets, selected_val_prob, thresholds)
    test_metrics = _metrics(b4_test_targets, selected_test_prob, thresholds)
    val_rows = compute_per_class_metrics(b4_val_targets, selected_val_prob, LABEL_COLUMNS, thresholds)
    test_rows = compute_per_class_metrics(b4_test_targets, selected_test_prob, LABEL_COLUMNS, thresholds)

    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "validation_targets.npy", b4_val_targets.astype(np.uint8))
    np.save(output_dir / "validation_probabilities.npy", selected_val_prob.astype(np.float32))
    np.save(output_dir / "test_targets.npy", b4_test_targets.astype(np.uint8))
    np.save(output_dir / "test_probabilities.npy", selected_test_prob.astype(np.float32))
    _write_per_class_csv(output_dir / "validation_per_class_metrics.csv", val_rows)
    _write_per_class_csv(output_dir / "test_per_class_metrics.csv", test_rows)
    _write_json(output_dir / "ensemble_results.json", {
        "selection_rule": "highest_validation_macro_auroc_then_macro_auprc",
        "threshold_source": "validation_only",
        "test_used_for_selection_or_tuning": False,
        "candidates": rows,
        "selected": {
            "key": selected_key,
            "b4_weight": selected["b4_weight"],
            "b5_weight": selected["b5_weight"],
            "validation": val_metrics,
            "test": test_metrics,
        },
    })
    _write_json(output_dir / "thresholds.json", {
        "source": "validation",
        "fit_split": "val",
        "strategy": "per_class_max_f1",
        "label_cols": list(LABEL_COLUMNS),
        "thresholds": thresholds.tolist(),
        "thresholds_by_label": dict(zip(LABEL_COLUMNS, thresholds.tolist(), strict=True)),
    })
    _write_json(output_dir / "ensemble_protocol.json", {
        "schema_version": 1,
        "status": "complete",
        "method": "weighted_average_of_frozen_sigmoid_probabilities",
        "weights_tested": [{"b4": w, "b5": 1.0 - w} for w in WEIGHTS],
        "selection_rule": "validation_macro_auroc_maximize",
        "selected_weights": {"b4": selected["b4_weight"], "b5": selected["b5_weight"]},
        "threshold_source": "validation",
        "test_used_for_selection_or_tuning": False,
        "split_counts": b4_protocol["split_counts"],
        "label_cols": list(LABEL_COLUMNS),
        "sources": {
            "b4_eval_dir": str(b4_dir),
            "b5_eval_dir": str(b5_dir),
            "b4_protocol_sha256": _sha256(b4_dir / "evaluation_protocol.json"),
            "b5_protocol_sha256": _sha256(b5_dir / "evaluation_protocol.json"),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return {
        "output_dir": str(output_dir),
        "selected_key": selected_key,
        "selected": selected,
        "validation": val_metrics,
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b4-dir", required=True, type=Path)
    parser.add_argument("--b5-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.b4_dir, args.b5_dir, args.output_dir)
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
