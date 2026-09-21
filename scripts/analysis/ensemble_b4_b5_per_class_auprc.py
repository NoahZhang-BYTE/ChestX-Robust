"""Select one B4/B5 probability weight per label using validation AUPRC.

Test probabilities are read only after all validation weights and thresholds
have been frozen. Existing ensemble artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from baseline.evaluation import compute_auc_metrics, compute_f1_metrics, compute_per_class_metrics, find_per_class_thresholds
from baseline.labels import LABEL_COLUMNS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_protocol(directory: Path) -> dict[str, Any]:
    protocol = _json(directory / "evaluation_protocol.json")
    if protocol.get("status") != "frozen" or protocol.get("test_used_for_tuning") is not False:
        raise ValueError(f"Input bundle is not a frozen, test-independent evaluation: {directory}")
    if protocol.get("label_cols") != list(LABEL_COLUMNS):
        raise ValueError(f"Unexpected label order: {directory}")
    if protocol.get("split_counts") != {"train": 89789, "val": 11348, "test": 10983}:
        raise ValueError(f"Unexpected split counts: {directory}")
    return protocol


def _load_inputs(b4_dir: Path, b5_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b4_protocol = _validate_protocol(b4_dir)
    b5_protocol = _validate_protocol(b5_dir)
    if b4_protocol["split_counts"] != b5_protocol["split_counts"]:
        raise ValueError("B4/B5 split counts differ")
    val_targets = np.load(b4_dir / "validation_targets.npy")
    b5_val_targets = np.load(b5_dir / "validation_targets.npy")
    test_targets = np.load(b4_dir / "test" / "test_targets.npy")
    b5_test_targets = np.load(b5_dir / "test" / "test_targets.npy")
    if not np.array_equal(val_targets, b5_val_targets) or not np.array_equal(test_targets, b5_test_targets):
        raise ValueError("B4/B5 target arrays or ordering differ")
    arrays = (
        np.load(b4_dir / "validation_probabilities.npy").astype(np.float64),
        np.load(b5_dir / "validation_probabilities.npy").astype(np.float64),
        np.load(b4_dir / "test" / "test_probabilities.npy").astype(np.float64),
        np.load(b5_dir / "test" / "test_probabilities.npy").astype(np.float64),
    )
    for name, array, expected in zip(("b4_val", "b5_val", "b4_test", "b5_test"), arrays, (val_targets.shape, val_targets.shape, test_targets.shape, test_targets.shape), strict=True):
        if array.shape != expected or not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
            raise ValueError(f"Invalid {name} probability array: {array.shape}")
    return b4_protocol, val_targets, arrays[0], arrays[1], test_targets, arrays[2], arrays[3]


def _best_weight(target: np.ndarray, b4: np.ndarray, b5: np.ndarray) -> tuple[float, float]:
    """Maximize AP with a deterministic coarse-to-fine search on [0, 1]."""
    best_weight = 0.0
    best_score = -np.inf
    lower, upper = 0.0, 1.0
    for _ in range(3):
        weights = np.linspace(lower, upper, 1001)
        scores = np.asarray([average_precision_score(target, weight * b4 + (1.0 - weight) * b5) for weight in weights])
        index = int(np.argmax(scores))
        best_weight = float(weights[index])
        best_score = float(scores[index])
        if index == 0:
            lower, upper = float(weights[0]), float(weights[1])
        elif index == len(weights) - 1:
            lower, upper = float(weights[-2]), float(weights[-1])
        else:
            lower, upper = float(weights[index - 1]), float(weights[index + 1])
    return best_weight, best_score


def _metrics(targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    auc = compute_auc_metrics(targets, probabilities, LABEL_COLUMNS)
    fixed = compute_f1_metrics(targets, probabilities, 0.5)
    tuned = compute_f1_metrics(targets, probabilities, thresholds)
    return {
        "macro_auroc": auc["macro_auroc"], "micro_auroc": auc["micro_auroc"],
        "macro_auprc": auc["macro_auprc"], "micro_auprc": auc["micro_auprc"],
        "macro_f1_at_0.5": fixed["macro_f1"], "micro_f1_at_0.5": fixed["micro_f1"],
        "sample_f1_at_0.5": fixed["sample_f1"], "macro_f1_tuned": tuned["macro_f1"],
        "micro_f1_tuned": tuned["micro_f1"], "sample_f1_tuned": tuned["sample_f1"],
        "per_class_auc": auc["per_class"],
    }


def run(b4_dir: Path, b5_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    b4_dir, b5_dir, output_dir = b4_dir.resolve(), b5_dir.resolve(), output_dir.resolve()
    b4_protocol, val_targets, b4_val, b5_val, test_targets, b4_test, b5_test = _load_inputs(b4_dir, b5_dir)

    weights: list[float] = []
    weight_rows: list[dict[str, Any]] = []
    for index, label in enumerate(LABEL_COLUMNS):
        weight, score = _best_weight(val_targets[:, index], b4_val[:, index], b5_val[:, index])
        weights.append(weight)
        b4_score = average_precision_score(val_targets[:, index], b4_val[:, index])
        b5_score = average_precision_score(val_targets[:, index], b5_val[:, index])
        weight_rows.append({"label": label, "b4_weight": weight, "b5_weight": 1.0 - weight, "validation_auprc": score, "b4_validation_auprc": b4_score, "b5_validation_auprc": b5_score})

    weights_array = np.asarray(weights, dtype=np.float64)
    val_probabilities = weights_array * b4_val + (1.0 - weights_array) * b5_val
    thresholds = np.asarray(find_per_class_thresholds(val_targets, val_probabilities), dtype=np.float64)
    test_probabilities = weights_array * b4_test + (1.0 - weights_array) * b5_test
    val_metrics = _metrics(val_targets, val_probabilities, thresholds)
    test_metrics = _metrics(test_targets, test_probabilities, thresholds)

    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "validation_targets.npy", val_targets.astype(np.uint8))
    np.save(output_dir / "validation_probabilities.npy", val_probabilities.astype(np.float32))
    np.save(output_dir / "test_targets.npy", test_targets.astype(np.uint8))
    np.save(output_dir / "test_probabilities.npy", test_probabilities.astype(np.float32))
    _write_csv(output_dir / "per_class_weights.csv", weight_rows)
    _write_csv(output_dir / "validation_per_class_metrics.csv", compute_per_class_metrics(val_targets, val_probabilities, LABEL_COLUMNS, thresholds))
    _write_csv(output_dir / "test_per_class_metrics.csv", compute_per_class_metrics(test_targets, test_probabilities, LABEL_COLUMNS, thresholds))
    _write_json(output_dir / "thresholds.json", {"source": "validation", "fit_split": "val", "strategy": "per_class_max_f1", "label_cols": list(LABEL_COLUMNS), "thresholds": thresholds.tolist(), "thresholds_by_label": dict(zip(LABEL_COLUMNS, thresholds.tolist(), strict=True))})
    _write_json(output_dir / "ensemble_results.json", {"selection_rule": "per_class_validation_AUPRC_maximize", "threshold_source": "validation_only", "test_used_for_selection_or_tuning": False, "weights_by_label": dict(zip(LABEL_COLUMNS, weights_array.tolist(), strict=True)), "validation": val_metrics, "test": test_metrics})
    _write_json(output_dir / "ensemble_protocol.json", {"schema_version": 1, "status": "complete", "method": "per_class_weighted_average_of_frozen_sigmoid_probabilities", "selection_rule": "independent_per_class_validation_AUPRC_maximize", "selected_weights": {label: {"b4": float(weight), "b5": float(1.0 - weight)} for label, weight in zip(LABEL_COLUMNS, weights_array, strict=True)}, "threshold_source": "validation", "test_used_for_selection_or_tuning": False, "split_counts": b4_protocol["split_counts"], "label_cols": list(LABEL_COLUMNS), "sources": {"b4_eval_dir": str(b4_dir), "b5_eval_dir": str(b5_dir), "b4_protocol_sha256": _sha256(b4_dir / "evaluation_protocol.json"), "b5_protocol_sha256": _sha256(b5_dir / "evaluation_protocol.json")}, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return {"output_dir": str(output_dir), "weights": dict(zip(LABEL_COLUMNS, weights_array.tolist(), strict=True)), "validation": val_metrics, "test": test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b4-dir", required=True, type=Path)
    parser.add_argument("--b5-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.b4_dir, args.b5_dir, args.output_dir), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
