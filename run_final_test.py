"""Run the one-time independent test evaluation from frozen val thresholds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision

from baseline.data import build_evaluation_loader, evaluation_preprocessing_spec
from baseline.evaluation import (
    collect_predictions,
    compute_auc_metrics,
    compute_f1_metrics,
    compute_per_class_metrics,
)
from baseline.labels import LABEL_COLUMNS
from baseline.models import build_model
from baseline.nih import validate_labels_csv


DEFAULT_FROZEN_DIR = Path("artifacts/nih_resnet18_imagenet_pretrained_best")


def load_and_validate_frozen_bundle(
    frozen_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load frozen validation artifacts and reject any provenance mismatch."""
    frozen_dir = Path(frozen_dir).resolve()
    protocol_path = frozen_dir / "evaluation_protocol.json"
    protocol = _read_json(protocol_path, "evaluation protocol")
    if protocol.get("status") != "frozen":
        raise ValueError("Evaluation protocol status must be frozen")
    if protocol.get("test_used_for_tuning") is not False:
        raise ValueError("Evaluation protocol must declare test_used_for_tuning=false")
    threshold_spec = protocol.get("threshold")
    if not isinstance(threshold_spec, dict):
        raise ValueError("Evaluation protocol is missing threshold metadata")
    if threshold_spec.get("source") != "validation":
        raise ValueError("Frozen threshold source must be validation")
    if threshold_spec.get("strategy") != "per_class_max_f1":
        raise ValueError("Frozen threshold strategy is not per_class_max_f1")
    artifact_name = threshold_spec.get("artifact")
    if not isinstance(artifact_name, str) or not artifact_name:
        raise ValueError("Evaluation protocol has an invalid threshold artifact path")
    threshold_path = (frozen_dir / artifact_name).resolve()
    try:
        threshold_path.relative_to(frozen_dir)
    except ValueError as error:
        raise ValueError("Threshold artifact must stay inside frozen_dir") from error
    if sha256_file(threshold_path) != threshold_spec.get("sha256"):
        raise ValueError("Frozen threshold artifact SHA256 does not match protocol")
    artifact = _read_json(threshold_path, "threshold artifact")
    if artifact.get("source") != "validation" or artifact.get("fit_split") != "val":
        raise ValueError("Threshold artifact must be fitted only on val")
    if artifact.get("strategy") != "per_class_max_f1":
        raise ValueError("Threshold artifact has an unexpected strategy")

    labels = protocol.get("label_cols")
    if not isinstance(labels, list) or artifact.get("label_cols") != labels:
        raise ValueError("Threshold artifact label order does not match protocol")
    if protocol.get("num_classes") != len(labels):
        raise ValueError("Protocol num_classes does not match label count")
    values = np.asarray(artifact.get("thresholds"), dtype=np.float64)
    if values.shape != (len(labels),):
        raise ValueError("Threshold artifact must contain one threshold per label")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Frozen thresholds must be finite values between 0 and 1")
    mapping = artifact.get("thresholds_by_label")
    if not isinstance(mapping, dict) or set(mapping) != set(labels):
        raise ValueError("Threshold mapping does not match protocol labels")
    mapped_values = np.asarray([mapping[label] for label in labels], dtype=np.float64)
    if not np.array_equal(values, mapped_values):
        raise ValueError("Ordered thresholds disagree with thresholds_by_label")
    protocol_mapping = threshold_spec.get("values_by_label")
    if protocol_mapping is not None and protocol_mapping != mapping:
        raise ValueError("Protocol thresholds disagree with threshold artifact")

    checkpoint_spec = protocol.get("checkpoint")
    labels_spec = protocol.get("labels_csv")
    if not isinstance(checkpoint_spec, dict) or not isinstance(labels_spec, dict):
        raise ValueError("Protocol is missing checkpoint or labels provenance")
    checkpoint_path = Path(checkpoint_spec.get("path", ""))
    labels_path = Path(labels_spec.get("path", ""))
    if sha256_file(checkpoint_path) != checkpoint_spec.get("sha256"):
        raise ValueError("Checkpoint SHA256 does not match frozen protocol")
    if sha256_file(labels_path) != labels_spec.get("sha256"):
        raise ValueError("labels.csv SHA256 does not match frozen protocol")
    if artifact.get("checkpoint_sha256") != checkpoint_spec.get("sha256"):
        raise ValueError("Threshold artifact checkpoint hash does not match protocol")
    if artifact.get("labels_csv_sha256") != labels_spec.get("sha256"):
        raise ValueError("Threshold artifact labels hash does not match protocol")
    return protocol, artifact


def run_final_test(
    frozen_dir: str | Path = DEFAULT_FROZEN_DIR,
    output_dir: str | Path | None = None,
    device_value: str = "auto",
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Consume frozen validation artifacts and evaluate the test split once."""
    frozen_dir = Path(frozen_dir).resolve()
    expected_output_dir = (frozen_dir / "test").resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else expected_output_dir
    if output_dir != expected_output_dir:
        raise ValueError("Formal test evaluation requires the fixed test output directory")
    if output_dir == frozen_dir:
        raise ValueError("Test output directory must differ from frozen_dir")
    if output_dir.exists():
        raise FileExistsError(f"Test output directory already exists: {output_dir}")
    protocol, threshold_artifact = load_and_validate_frozen_bundle(frozen_dir)
    labels = list(protocol["label_cols"])
    thresholds = np.asarray(threshold_artifact["thresholds"], dtype=np.float64)
    checkpoint_path = Path(protocol["checkpoint"]["path"])
    labels_csv = Path(protocol["labels_csv"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state", "epoch", "config", "metrics"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint is missing required keys: {sorted(required)}")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a mapping")
    if sha256_json(config) != protocol.get("config", {}).get("sha256"):
        raise ValueError("Embedded checkpoint config SHA256 does not match protocol")
    checkpoint_labels = list(config.get("data", {}).get("label_cols") or [])
    if checkpoint_labels != labels or labels != list(LABEL_COLUMNS):
        raise ValueError("Checkpoint, protocol, and canonical label order must match")
    if int(checkpoint["epoch"]) != int(protocol["checkpoint"]["epoch"]):
        raise ValueError("Checkpoint epoch does not match frozen protocol")

    data_config = bind_frozen_data_paths(config["data"], protocol)
    image_root = Path(data_config["image_root"]).resolve()
    frame = validate_labels_csv(labels_csv, image_root)
    actual_counts = {
        split: int(frame["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    if actual_counts != protocol.get("split_counts"):
        raise ValueError("Current split counts do not match frozen protocol")
    loader_config = _loader_config(
        data_config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader, loader_labels = build_evaluation_loader(
        split="test",
        preprocessing=protocol["preprocessing"],
        **loader_config,
    )
    if loader_labels != labels:
        raise ValueError("Test loader label order does not match frozen protocol")
    image_size = int(data_config.get("image_size", 224))
    if protocol.get("preprocessing") != evaluation_preprocessing_spec(image_size):
        raise ValueError("Test preprocessing does not match frozen protocol")

    model_config = config["model"]
    model = build_model(model_config["name"], len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = resolve_device(device_value)
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    amp_enabled = bool(config.get("training", {}).get("amp", False)) and device.type == "cuda"
    targets, probabilities = collect_predictions(
        model, test_loader, device, amp_enabled=amp_enabled
    )
    if targets.shape != (actual_counts["test"], len(labels)):
        raise ValueError(f"Test predictions have unexpected shape {targets.shape}")

    auc_metrics = compute_auc_metrics(targets, probabilities, labels)
    f1_at_0_5 = compute_f1_metrics(targets, probabilities, 0.5)
    f1_tuned = compute_f1_metrics(targets, probabilities, thresholds)
    tuned_rows = compute_per_class_metrics(
        targets, probabilities, labels, thresholds
    )
    reference_rows = compute_per_class_metrics(targets, probabilities, labels, 0.5)
    reference_by_label = {str(row["label"]): row for row in reference_rows}
    csv_rows: list[dict[str, Any]] = []
    for row in tuned_rows:
        reference = reference_by_label[str(row["label"])]
        csv_rows.append(
            {
                "class": row["label"],
                "threshold": row["threshold"],
                "auroc": row["auroc"],
                "auprc": row["auprc"],
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "sensitivity": row["recall"],
                "specificity": row["specificity"],
                "positive_count": row["positive_count"],
                "total_count": row["total_count"],
                "f1_at_0_5": reference["f1"],
                "precision_at_0_5": reference["precision"],
                "recall_at_0_5": reference["recall"],
                "specificity_at_0_5": reference["specificity"],
            }
        )
    evaluated_at = utc_now()
    summary = {
        "schema_version": 1,
        "split": "test",
        "sample_count": int(targets.shape[0]),
        "num_classes": int(targets.shape[1]),
        "macro_auroc": auc_metrics["macro_auroc"],
        "micro_auroc": auc_metrics["micro_auroc"],
        "macro_auprc": auc_metrics["macro_auprc"],
        "micro_auprc": auc_metrics["micro_auprc"],
        "macro_f1_at_0.5": f1_at_0_5["macro_f1"],
        "micro_f1_at_0.5": f1_at_0_5["micro_f1"],
        "sample_f1_at_0.5": f1_at_0_5["sample_f1"],
        "macro_f1_tuned": f1_tuned["macro_f1"],
        "micro_f1_tuned": f1_tuned["micro_f1"],
        "sample_f1_tuned": f1_tuned["sample_f1"],
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "labels_csv_sha256": protocol["labels_csv"]["sha256"],
        "thresholds_sha256": protocol["threshold"]["sha256"],
        "evaluation_protocol_sha256": sha256_file(
            frozen_dir / "evaluation_protocol.json"
        ),
        "evaluation_timestamp_utc": evaluated_at,
        "test_used_for_tuning": False,
    }

    stage = _make_stage(output_dir)
    try:
        np.save(stage / "test_targets.npy", targets.astype(np.uint8))
        np.save(
            stage / "test_probabilities.npy", probabilities.astype(np.float32)
        )
        _write_csv(stage / "test_per_class_metrics.csv", csv_rows)
        write_json(stage / "test_summary.json", summary)
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in stage.iterdir()
            if path.is_file()
        }
        metadata = {
            "schema_version": 1,
            "stage": "independent_test_evaluation",
            "evaluation_timestamp_utc": evaluated_at,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "checkpoint_sha256": protocol["checkpoint"]["sha256"],
            "labels_csv_sha256": protocol["labels_csv"]["sha256"],
            "thresholds_sha256": protocol["threshold"]["sha256"],
            "evaluation_protocol_sha256": summary[
                "evaluation_protocol_sha256"
            ],
            "artifact_sha256": artifact_hashes,
        }
        write_json(stage / "run_metadata.json", metadata)
        manifest_files = {
            path.name: sha256_file(path)
            for path in stage.iterdir()
            if path.is_file()
        }
        write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "files": manifest_files,
            },
        )
        stage.replace(output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir),
        "summary": summary,
        "per_class": csv_rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independent test evaluation from a frozen val protocol."
    )
    parser.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_final_test(
        frozen_dir=args.frozen_dir,
        output_dir=None,
        device_value=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(json_ready(result), ensure_ascii=True, indent=2))


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _loader_config(
    data_config: dict[str, Any],
    *,
    batch_size: int | None,
    num_workers: int | None,
) -> dict[str, Any]:
    names = {
        "csv_path",
        "image_root",
        "image_col",
        "label_cols",
        "image_size",
        "batch_size",
        "num_workers",
        "prefetch_factor",
        "persistent_workers",
    }
    result = {key: value for key, value in data_config.items() if key in names}
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        result["batch_size"] = batch_size
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        result["num_workers"] = num_workers
    return result


def bind_frozen_data_paths(
    data_config: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Replace checkpoint-relative paths with the paths validated by the protocol."""
    labels_spec = protocol.get("labels_csv") or {}
    image_root = protocol.get("image_root")
    if not isinstance(labels_spec, dict) or not isinstance(labels_spec.get("path"), str):
        raise ValueError("Protocol is missing an absolute labels_csv path")
    if not isinstance(image_root, str):
        raise ValueError("Protocol is missing an absolute image_root path")
    bound = dict(data_config)
    bound["csv_path"] = labels_spec["path"]
    bound["image_root"] = image_root
    return bound


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required provenance file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            json_ready(value),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Per-class metrics must contain at least one row")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_ready(rows))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_stage(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )


if __name__ == "__main__":
    main()
