# One-Day F1 Experiment and External Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe NIH F1 training/evaluation loop, compare ordinary BCE with training-set-derived `pos_weight`, and verify CheXpert-small readiness on the U drive without deleting or replacing NIH data.

**Architecture:** Keep the existing NIH data and ImageNet-initialized ResNet18 as the reference. Add pure metric/threshold utilities, a named persisted-split loader, a separate evaluation path, and a small configurable loss factory. Training selects `best.pt` by validation macro AUPRC, thresholds are fitted once on validation for that frozen checkpoint, and only then is a final NIH test batch generated. CheXpert work is limited to a read-only staging preflight; no external labels are converted and no external model is trained in this plan.

**Tech Stack:** Python 3.12, PyTorch 2.11 + CUDA when available, torchvision, pandas, NumPy, Pillow, scikit-learn, PyYAML, pytest, PowerShell 7.6.

## Global Constraints

- Do not delete, move, archive, overwrite, or rename `data/labels.csv`, `data/raw/images`, or any existing checkpoint during this plan.
- Keep the historical reference at `checkpoints/imagenet_pretrained/best.pt`; it was selected by fixed-threshold macro AUROC and is not an F1-selected model.
- Use the current NIH persisted `train`/`val`/`test` split; never use an image-level fallback when a patient identifier exists without a persisted split.
- Fit per-label thresholds only on `val`; test and external data may only consume a frozen threshold artifact.
- The same `val` split selects the AUPRC checkpoint, fits thresholds, and ranks
  candidates; this is an exploratory, potentially overfit calibration loop.
  Always report fixed-0.5 and tuned values separately, and never describe tuned
  validation F1 as an unbiased generalization estimate. An independent
  calibration split is a follow-up design, not a hidden day-one step.
- Every real `test` evaluation must be authorized by
  `docs/experiments/nih-f1-selection.json`; `run_final_test.py` is the only
  planned command that generates real NIH test reports.
- The final-test output directory is write-once: if it already exists,
  `run_final_test.py` must exit before loading the test split and must not alter
  any existing result.
- Select between BCE and `pos_weight` using validation-only tuned macro F1, with validation macro AUPRC as the tie-breaker. Do not inspect real NIH test output before writing the selection record.
- Each new run uses ResNet18 with ImageNet initialization, seed `42`, batch size `64`, learning rate `0.0003`, weight decay `0.0001`, AMP when CUDA is available, at most `10` epochs or `2` wall-clock hours, `patience: 2`, and `min_delta: 0.0005`.
- The two new runs set `deterministic: true`: seed Python/NumPy/PyTorch, set
  `torch.backends.cudnn.benchmark = False`, set
  `torch.backends.cudnn.deterministic = True`, and call
  `torch.use_deterministic_algorithms(True)`. If a documented CUDA operator
  exception forces a relaxation, record the exact warning and setting in the
  run metadata; never silently leave `benchmark=True` for one arm only.
- Checkpoints, reports, and generated artifacts go under `D:/ChestXRobustRuns` or the repository; never write checkpoints to the removable drive.
- Training output directories are write-once too: `fit`/`train.py` must reject
  an existing non-empty directory unless an explicit, out-of-scope `--resume`
  mode is added in a separately approved change. Check this in Python, not only
  in the surrounding PowerShell script.
- CheXpert is only a preflight target at `E:/ChestXRobustData/CheXpert-v1.0-small`; no download is assumed to be authorized, and no external robustness claim may be made from a preflight.
- Existing NIH tests must remain green. New tests use tiny synthetic PNG/JPG fixtures and must not require the NIH image corpus or an internet connection.

## Frozen Input Inventory

These values were measured read-only on `2026-08-30` before implementation or
new training and are frozen by the commit containing this plan. The plan itself
must be committed before any implementation or training; the protocol records
that commit hash.

| Input | SHA-256 / inventory | Bytes |
| --- | --- | ---: |
| `data/labels.csv` | `c0b1784dce4215b12396d2e610d70469f91b0c068d6096081fbfee184df93318` | 10,076,512 |
| `checkpoints/imagenet_pretrained/best.pt` | `be8ef6905f358488ed6d5dd2da8879c656084ee5d184795587a5ccef3584b972` | 134,322,153 |
| `data/raw/images` | 112,120 files | 45,057,440,698 |

The historical checkpoint predates training-time provenance fields, so this
inventory binds its exact pre-experiment bytes and the current NIH labels but
cannot retroactively prove the labels-file hash used during its original
training. State that limitation in the result record. Every new candidate must
embed its source config-file and labels-CSV hashes at training start.

## Research Decision Contract

This is an understanding-stage experiment: maximize evidence that distinguishes
the leading explanations, not the number of techniques tried.

| Hypothesis or alternative | Decisive one-day observation | Decision |
| --- | --- | --- |
| H1: fixed `0.5` thresholds explain much of the low F1 | The historical checkpoint's frozen validation thresholds materially raise validation macro F1, then retain useful test F1 without changing AUROC/AUPRC | Keep threshold fitting as part of evaluation; do not call it a training improvement |
| H2: training imbalance limits tail-label recall | The matched `pos_weight` run beats BCE on tuned validation macro F1, with macro AUPRC as the tie-breaker | Select `pos_weight` for the single final comparison |
| A1: re-training alone is sufficient | BCE beats both the historical reference and `pos_weight` under the same frozen-threshold protocol | Select BCE and attribute the result only to this controlled rerun |
| A2: weighting trades too much precision for recall | `pos_weight` loses the predeclared validation ranking or produces degenerate/non-finite evidence | Reject `pos_weight`; do not rescue it with test-driven tuning |
| A3: the day cannot answer robustness | CheXpert is only preflighted and no frozen model is evaluated externally | Report robustness as unmeasured regardless of NIH F1 |

Pre-registered comparison tolerances are fixed: treat macro-F1 or macro-AUPRC
differences with absolute value `<= 1e-6` as ties; resolve macro-F1 ties by
macro-AUPRC, then resolve exact AUPRC ties in favor of BCE. Call a threshold or
loss change a *meaningful F1 improvement* only when tuned validation macro-F1
increases by at least `0.01` absolute versus the named control. Smaller wins are
reported as numerical/tie movement, not as a robustness or training claim.

Fast-fail gates are binding: stop before GPU training if split/threshold tests
fail; stop candidate comparison if the historical threshold scan cannot be
reproduced from the declared checkpoint/config; mark a timed-out or non-finite
candidate ineligible rather than extrapolating its trajectory. Both candidate
arms are required for the comparison, so an ineligible arm stops selection and
the final NIH test rather than promoting the remaining arm. A negative or
inconclusive comparison is a valid result and triggers a later, separately
approved ASL or external-evaluation experiment instead of post-hoc rule changes.

## PowerShell Execution Gate

Every `powershell` block implicitly starts with this PowerShell 7.6 prologue:

~~~powershell
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
~~~

Thus a non-zero Python, pytest, or git exit stops that step and all dependent
steps. The only intentional non-zero commands are the explicitly labeled
red-phase tests; run their following implementation step in a fresh shell after
confirming the expected failure. Never commit or advance a gate after an
unexpected native-command failure.

## External Dataset Triage

The following public sources were compared for the later robustness phase. The
one-day choice is based on access and ontology risk, not on a claim that any
source is clinically interchangeable with NIH:

| Dataset | Best use after this day | Access / operational constraint | Day-one status |
| --- | --- | --- | --- |
| CheXpert-v1.0-small (Stanford AIMI) | First staged cross-domain seven-concept experiment; CSV/JPG layout is easy to preflight | Stanford data-use terms; `-1` uncertainty and `Pleural Effusion` naming require an explicit policy/mapping | Read-only `E:` preflight only |
| BRAX (PhysioNet) | Preferred external holdout from a different country/hospital domain | Credentialed PhysioNet access and preparation; do not assume availability | Queue after CheXpert contract is frozen |
| VinDr-CXR (PhysioNet) | Fallback external holdout with useful lesion labels | Credentialed access; native DICOM and `Nodule/Mass` ontology need separate handling | Queue only if BRAX is blocked |
| MIMIC-CXR-JPG (PhysioNet) | Larger pretraining or sensitivity analysis | Credentialed access and report-derived labels; not a drop-in NIH test set | Defer beyond the one-day budget |

Official entry points to recheck at execution time are [Stanford
CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays), [BRAX on
PhysioNet](https://physionet.org/content/brax/1.0.0/), [VinDr-CXR on
PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/), and [MIMIC-CXR on
PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.1.0/). Access status,
terms, and published size metadata must be rechecked rather than inferred from
an old local listing.

## File Map

| File | Responsibility |
| --- | --- |
| `baseline/metrics.py` | Threshold normalization/fitting and aggregate/per-label metrics. |
| `baseline/data.py` | Named persisted split loading and training-label counts while preserving the existing training API. |
| `baseline/evaluation.py` | Checkpoint/config validation, prediction collection, and split evaluation shared by both CLIs. |
| `baseline/losses.py` | Ordinary BCE and finite capped `pos_weight` BCE factory. |
| `baseline/engine.py` | Configured loss, validation AUPRC checkpoint selection, early stopping, wall-clock guard, and run history. |
| `train.py` | Pass resolved labels and configured training behavior into `fit`. |
| `evaluate.py` | Evaluate one frozen checkpoint on validation with scalar or frozen vector thresholds. |
| `tune_thresholds.py` | Fit a threshold artifact from validation predictions only. |
| `select_candidate.py` | Rank completed candidates from validation JSON only and write the pre-test selection record. |
| `run_final_test.py` | Validate the frozen selection record and generate the only authorized NIH test batch. |
| `baseline/chexpert_preflight.py` | Read-only CheXpert layout, capacity, header, path, and sampled-image checks. |
| `preflight_chexpert.py` | CLI wrapper for the preflight function. |
| `configs/nih_f1_bce.yaml` | Ordinary BCE experiment. |
| `configs/nih_f1_posweight.yaml` | Training-count-derived `pos_weight` experiment. |
| `configs/chexpert_preflight.yaml` | U-drive preflight contract and seven-label future comparison list. |
| `tests/test_metrics_thresholds.py` | Threshold and metric edge cases. |
| `tests/test_split_loader.py` | Named split isolation and patient-disjointness guards. |
| `tests/test_evaluation.py` | Split evaluation and label-order/test-leakage guards. |
| `tests/test_losses.py` | Loss factory and weight calculation. |
| `tests/test_engine.py` | Validation selection, early stopping, and checkpoint metadata. |
| `tests/test_chexpert_preflight.py` | External staging checks without network access. |
| `docs/experiments/2026-08-30-nih-f1-run.md` | Pre-registered protocol with append-only measured result sections. |
| `docs/experiments/2026-08-30-input-inventory.json` | Frozen NIH/reference hashes and the explicit legacy-provenance limitation. |

---

### Task 0: Commit the frozen plan before implementation

**Files:**
- Add: `docs/superpowers/plans/2026-08-30-one-day-f1-experiment.md`

The plan is a decision boundary, so commit it before executing Task 1 or
running any implementation, smoke, or training command. Preserve the separate
untracked `docs/superpowers/plans/2026-08-29-external-dataset-migration.md` and
do not stage it.

- [ ] **Step 1: Verify and commit only this plan**

~~~powershell
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
git diff --check -- docs/superpowers/plans/2026-08-30-one-day-f1-experiment.md
git add docs/superpowers/plans/2026-08-30-one-day-f1-experiment.md
git commit -m "docs: freeze one-day F1 implementation plan"
git rev-parse HEAD
~~~

Expected: one new commit containing only this plan. Record its printed hash as
`plan_commit` in the Task 5 protocol before the protocol/config freeze commit.

---

### Task 1: Add threshold-safe metrics and validation-only threshold fitting

**Files:**
- Modify: `baseline/metrics.py:1-36`
- Create: `tests/test_metrics_thresholds.py`
- Modify: `tests/test_baseline.py:129-137`

**Interfaces:**
- `normalize_thresholds(threshold: float | Sequence[float], num_labels: int) -> np.ndarray`
- `apply_thresholds(probabilities: np.ndarray, thresholds: float | Sequence[float]) -> np.ndarray`
- `fit_per_class_thresholds(targets: np.ndarray, probabilities: np.ndarray, label_cols: Sequence[str], fit_split: str = "val", objective: str = "binary_f1") -> dict[str, Any]`
- `load_threshold_artifact(path: str | Path, label_cols: Sequence[str]) -> dict[str, Any]`
- `multilabel_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float | Sequence[float] = 0.5, label_cols: Sequence[str] | None = None) -> dict[str, Any]`

Keep the five existing aggregate keys and add `macro_auprc`, `micro_auprc`, and
`per_label`. Each per-label entry has `label`, positive/negative counts,
prevalence, precision, recall, F1, AUROC, AUPRC, and `degenerate`. One-class labels emit `None` for
their score fields and are excluded from macro means. Preserve Python
`float("nan")` for an all-degenerate aggregate AUROC; CLI JSON converts
non-finite values to `null`.

- [ ] **Step 1: Write failing tests**

~~~python
def test_scalar_and_vector_thresholds_are_equivalent():
    targets = np.array([[1, 0], [0, 1], [1, 1]])
    probabilities = np.array([[0.9, 0.2], [0.4, 0.8], [0.6, 0.7]])
    assert multilabel_metrics(targets, probabilities, 0.5)["macro_f1"] == \
           multilabel_metrics(targets, probabilities, [0.5, 0.5])["macro_f1"]


def test_threshold_vector_keeps_label_order():
    assert apply_thresholds(np.array([[0.30, 0.30]]), [0.20, 0.40]).tolist() == [[1, 0]]


def test_fit_requires_val_and_marks_single_class():
    with pytest.raises(ValueError, match="fit_split must be 'val'"):
        fit_per_class_thresholds(np.array([[1], [0]]), np.array([[.8], [.2]]), ["x"], "test")
    artifact = fit_per_class_thresholds(
        np.zeros((3, 1), dtype=int), np.full((3, 1), .5), ["x"]
    )
    assert artifact["thresholds"] == [1.0]
    assert artifact["available"] == [False]
    assert artifact["reasons"] == ["no_positive"]


def test_threshold_tie_uses_highest_value():
    targets = np.array([[1], [0], [0], [1]])
    probabilities = np.array([[.9], [.8], [.7], [.6]])
    artifact = fit_per_class_thresholds(targets, probabilities, ["x"])
    assert artifact["thresholds"] == [0.9]


def test_per_label_metrics_include_prevalence_and_null_degenerate_scores():
    targets = np.array([[1, 0], [0, 0]])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.3]])
    metrics = multilabel_metrics(targets, probabilities, 0.5, ["a", "b"])
    assert metrics["per_label"][0]["prevalence"] == 0.5
    assert metrics["per_label"][0]["positive_count"] == 1
    assert metrics["per_label"][1]["degenerate"] is True
    for key in ("precision", "recall", "f1", "auroc", "auprc"):
        assert metrics["per_label"][1][key] is None
~~~

Run: `& ".venv/Scripts/python.exe" -m pytest tests/test_metrics_thresholds.py -q`
Expected: FAIL because these APIs do not exist.

- [ ] **Step 2: Implement the metric contract**

Use `np.asarray` shape checks and the following threshold normalization:

~~~python
def normalize_thresholds(threshold, num_labels):
    values = np.asarray(
        [threshold] * num_labels if np.isscalar(threshold) else threshold,
        dtype=np.float32,
    )
    if values.shape != (num_labels,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("threshold must be a scalar or one finite value in [0, 1] per label")
    return values


def apply_thresholds(probabilities, thresholds):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array")
    return (probabilities >= normalize_thresholds(thresholds, probabilities.shape[1])[None, :]).astype(np.int64)
~~~

For each non-degenerate label, sort probabilities descending, group equal
probabilities, and compute cumulative TP/FP at each group boundary. Evaluate
the unique observed thresholds plus `0.0` and `1.0` using `>=`, calculate
`2 * TP / (2 * TP + FP + FN)`, and resolve exact ties to the highest threshold.
This is `O(n log n)` per label and must not call `f1_score` once per unique
probability. Require
`fit_split == "val"` and `objective == "binary_f1"`. Store `schema_version: 1`,
label order, thresholds, availability, and reasons. Reject mismatched labels or
invalid values when loading an artifact. Return the validated artifact mapping
(not only its vector) so evaluation can enforce `fit_split == "val"`. Add safe
AUROC and average-precision
helpers; macro means skip degenerate labels and micro metrics flatten all labels.
Update the existing all-degenerate baseline test to assert non-crashing
`micro_f1` plus `np.isnan(macro_f1)` and `np.isnan(macro_auroc)`, matching the
approved macro-exclusion contract.

- [ ] **Step 3: Run focused and regression tests**

~~~powershell
& ".venv/Scripts/python.exe" -m pytest tests/test_metrics_thresholds.py tests/test_baseline.py -q
~~~

Expected: all new tests and existing metric tests pass.

- [ ] **Step 4: Commit**

~~~powershell
git add baseline/metrics.py tests/test_metrics_thresholds.py tests/test_baseline.py
git commit -m "feat: add validation-only threshold metrics"
~~~

---

### Task 2: Expose a named persisted split loader without changing NIH behavior

**Files:**
- Modify: `baseline/data.py:60-126`
- Create: `tests/test_split_loader.py`

**Interfaces:**
- `validate_persisted_patient_disjointness(frame: pd.DataFrame) -> None`
- `build_split_loader(csv_path, image_root, split, image_col, label_cols, image_size=224, batch_size=16, num_workers=0, prefetch_factor=2, persistent_workers=True) -> tuple[DataLoader, list[str]]`
- `training_label_counts(loader: DataLoader, label_cols: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write failing split tests**

Create four valid 16x16 PNGs with distinct patient IDs and persisted
`train`/`val`/`test` values. Assert the test loader contains only the test row.
Create a second CSV with `patient_id` but no `split` and assert both the named
loader and `build_dataloaders` raise
`ValueError("named split requires a persisted split when patient_id is present")`.
Create a legacy CSV with neither `patient_id` nor `split` and assert
`build_dataloaders` still uses its seeded image-level fallback.
Add a third CSV where one `patient_id` occurs in both `train` and `test` and
assert that both `build_dataloaders` and `build_split_loader` raise
`ValueError("patient leakage across persisted splits")` before constructing a
loader.
Add a fourth CSV with a missing or `development` split value and assert both
entry points reject it with `ValueError("split values must be train, val, or test")`.

Run: `& ".venv/Scripts/python.exe" -m pytest tests/test_split_loader.py -q`
Expected: FAIL because the named loader is absent.

- [ ] **Step 2: Implement the named loader**

Refactor loader construction into private
`_make_loader(frame, image_root, image_col, label_cols, image_size, batch_size,
num_workers, prefetch_factor, persistent_workers, training)` and
leave the existing `build_dataloaders` signature untouched. Preserve its random
fallback only when both `split` and `patient_id` are absent. If `patient_id` is
present without `split`, reject the input before `_split_frame`; never perform an
image-level fallback on patient-addressable data.
Add this shared guard and call it from both loader entry points whenever
`split` and `patient_id` are present:

~~~python
def validate_persisted_patient_disjointness(frame):
    if "split" not in frame.columns or "patient_id" not in frame.columns:
        return
    if frame["patient_id"].isna().any():
        raise ValueError("patient_id contains missing values")
    leaking = frame.groupby("patient_id")["split"].nunique()
    if (leaking > 1).any():
        raise ValueError("patient leakage across persisted splits")


def validate_split_values(frame):
    if frame["split"].isna().any() or not set(frame["split"].astype(str)).issubset({"train", "val", "test"}):
        raise ValueError("split values must be train, val, or test")
~~~

Then add, calling `validate_split_values(frame)` immediately after the patient
guard and before selecting rows:

~~~python
def build_split_loader(csv_path, image_root, split, image_col="path", label_cols=None,
                       image_size=224, batch_size=16, num_workers=0,
                       prefetch_factor=2, persistent_workers=True):
    frame = pd.read_csv(csv_path)
    validate_persisted_patient_disjointness(frame)
    if "split" not in frame.columns:
        if "patient_id" in frame.columns:
            raise ValueError("named split requires a persisted split when patient_id is present")
        raise ValueError("CSV has no persisted split")
    validate_split_values(frame)
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    selected = frame.loc[frame["split"] == split].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"persisted split '{split}' is empty")
    labels = _resolve_label_cols(frame, image_col, label_cols)
    return _make_loader(selected, image_root, image_col, labels, image_size,
                        batch_size, num_workers, prefetch_factor,
                        persistent_workers, training=False), labels


def training_label_counts(loader, label_cols):
    frame = loader.dataset.frame
    positives = torch.tensor(frame[list(label_cols)].sum(0).to_numpy(), dtype=torch.float32)
    negatives = torch.full_like(positives, len(frame)) - positives
    if (positives <= 0).any():
        missing = [label_cols[i] for i, value in enumerate(positives.tolist()) if value <= 0]
        raise ValueError(f"training labels have no positives: {missing}")
    return positives, negatives
~~~

Use the existing validation transform for `training=False`, preserve ImageNet
normalization, worker checks, and explicit label lists. In
`build_dataloaders`, call `validate_persisted_patient_disjointness(frame)`
immediately after `pd.read_csv`; then reject `patient_id` without `split` before
the existing `_split_frame` branch.

- [ ] **Step 3: Run regression tests**

~~~powershell
& ".venv/Scripts/python.exe" -m pytest tests/test_split_loader.py tests/test_baseline.py tests/test_nih.py -q
~~~

Expected: PASS with no change to NIH adapter behavior.

- [ ] **Step 4: Commit**

~~~powershell
git add baseline/data.py tests/test_split_loader.py
git commit -m "feat: load persisted evaluation splits"
~~~

---

### Task 3: Add frozen-checkpoint evaluation and a programmatic test gate

**Files:**
- Create: `baseline/evaluation.py`
- Create: `evaluate.py`
- Create: `tune_thresholds.py`
- Create: `select_candidate.py`
- Create: `run_final_test.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- `sha256_file(path: str | Path) -> str`
- `validate_selection_record(record: Mapping[str, Any]) -> dict[str, Any]`
- `load_checkpoint_bundle(checkpoint_path: str | Path, expected_label_cols: Sequence[str], device: torch.device) -> tuple[nn.Module, dict[str, Any]]`
- `collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool = False) -> tuple[np.ndarray, np.ndarray]`
- `run_final_test_batch(selection_record: Mapping[str, Any], output_dir: str | Path) -> Path`
- `evaluate_checkpoint(config_path: str | Path, checkpoint_path: str | Path, split: str, threshold_artifact: str | Path | None = None, scalar_threshold: float | None = None, selection_record: Mapping[str, Any] | None = None, role: str | None = None, legacy_inventory: str | Path | None = None) -> dict[str, Any]`
- `evaluate.py --config PATH --checkpoint PATH --split val (--thresholds PATH | --threshold FLOAT) --output PATH [--legacy-inventory PATH]`
- `tune_thresholds.py --config PATH --checkpoint PATH --split val --output PATH [--legacy-inventory PATH]`
- `select_candidate.py --reference-report PATH --bce-report PATH --posweight-report PATH --output PATH`
- `run_final_test.py --selection-record PATH --output-dir PATH`

- [ ] **Step 1: Write failing tests for resolved labels and the test gate**

Use a tiny persisted-split CSV and CPU `resnet18` checkpoint. Define the test
helpers completely so the examples below run without repository data:

~~~python
import copy
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from baseline.evaluation import evaluate_checkpoint, sanitize_for_json, sha256_file
from baseline.models import build_model


def tiny_bundle(tmp_path, config_labels, checkpoint_labels):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index, split in enumerate(("train", "train", "val", "val", "test", "test")):
        name = f"{split}-{index}.png"
        Image.new("RGB", (16, 16), color=(index * 40, 0, 0)).save(image_root / name)
        rows.append({
            "path": name, "split": split, "patient_id": index + 1,
            "a": index % 2, "b": (index + 1) % 2,
        })
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    config = {
        "device": "cpu",
        "data": {
            "csv_path": str(csv_path), "image_root": str(image_root),
            "image_col": "path", "label_cols": list(config_labels),
            "image_size": 16, "batch_size": 1, "num_workers": 0,
        },
        "model": {"name": "resnet18", "pretrained": False},
        "training": {"loss": {"name": "bce"}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    saved_config = copy.deepcopy(config)
    saved_config["data"]["label_cols"] = list(checkpoint_labels)
    model = build_model("resnet18", len(checkpoint_labels), pretrained=False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": {}, "epoch": 1,
        "config": saved_config, "metrics": {},
        "label_cols": list(checkpoint_labels),
        "run_metadata": {
            "loss_name": "bce", "positive_counts": None,
            "negative_counts": None, "pos_weight": None,
            "selection_metric": "macro_auroc", "patience": None,
            "min_delta": 0.0, "max_hours": None,
            "source_config": str(config_path),
            "source_config_sha256": sha256_file(config_path),
            "labels_csv": str(csv_path),
            "labels_csv_sha256": sha256_file(csv_path),
            "deterministic": False, "cudnn_benchmark": False,
            "cudnn_deterministic": False, "deterministic_algorithms": False,
            "elapsed_seconds": 0.0, "stop_reason": "completed",
        },
    }, checkpoint_path)
    return config_path, checkpoint_path


def write_validation_evidence(name, config_path, checkpoint_path, macro_f1, macro_auprc):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    labels_csv = Path(config["data"]["csv_path"])
    threshold_path = config_path.parent / f"{name}-thresholds.json"
    threshold = {
        "schema_version": 1, "fit_split": "val", "objective": "binary_f1",
        "label_cols": ["a", "b"], "thresholds": [0.5, 0.5],
        "available": [True, True], "reasons": [None, None],
        "config": str(config_path), "checkpoint": str(checkpoint_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "labels_csv_sha256": sha256_file(labels_csv),
        "training_provenance": "checkpoint_bound",
    }
    threshold_path.write_text(json.dumps(threshold), encoding="utf-8")
    report_path = config_path.parent / f"{name}-validation.json"
    report = {
        "schema_version": 1, "evaluation_split": "val",
        "config": str(config_path), "checkpoint": str(checkpoint_path),
        "label_cols": ["a", "b"],
        "provenance": {
            "config_sha256": sha256_file(config_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "labels_csv_sha256": sha256_file(labels_csv),
            "thresholds_sha256": sha256_file(threshold_path),
            "training_provenance": "checkpoint_bound",
        },
        "threshold": {
            "mode": "artifact", "value": None, "artifact": str(threshold_path),
            "fit_split": "val",
        },
        "metrics": {
            "macro_auroc": 0.5, "micro_auroc": 0.5,
            "macro_auprc": macro_auprc, "micro_auprc": macro_auprc,
            "macro_f1": macro_f1, "micro_f1": macro_f1,
            "sample_f1": macro_f1,
            "per_label": [
                {
                    "label": label, "positive_count": 1, "negative_count": 1,
                    "prevalence": 0.5, "precision": macro_f1,
                    "recall": macro_f1, "f1": macro_f1, "auroc": 0.5,
                    "auprc": macro_auprc, "degenerate": False,
                }
                for label in ("a", "b")
            ],
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return {
        "name": name, "validation_report": str(report_path),
        "config": str(config_path), "checkpoint": str(checkpoint_path),
        "thresholds": str(threshold_path), "label_cols": ["a", "b"],
        "metrics": {"macro_f1": macro_f1, "macro_auprc": macro_auprc},
        "sha256": {
            "config": sha256_file(config_path),
            "checkpoint": sha256_file(checkpoint_path),
            "thresholds": sha256_file(threshold_path),
            "labels_csv": sha256_file(labels_csv),
            "validation_report": sha256_file(report_path),
        },
    }


def frozen_selection_record(config_path, authorized_checkpoint):
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pos_config = copy.deepcopy(base_config)
    pos_config["training"] = {"loss": {"name": "bce_pos_weight", "cap": 20.0}}
    pos_config_path = config_path.parent / "posweight-config.yaml"
    pos_config_path.write_text(yaml.safe_dump(pos_config), encoding="utf-8")
    pos_checkpoint = torch.load(authorized_checkpoint, map_location="cpu", weights_only=False)
    pos_checkpoint["config"] = copy.deepcopy(pos_config)
    pos_checkpoint["run_metadata"].update({
        "source_config": str(pos_config_path),
        "source_config_sha256": sha256_file(pos_config_path),
    })
    pos_checkpoint_path = config_path.parent / "posweight.pt"
    torch.save(pos_checkpoint, pos_checkpoint_path)

    bce = write_validation_evidence("bce", config_path, authorized_checkpoint, 0.4, 0.3)
    posweight = write_validation_evidence(
        "posweight", pos_config_path, pos_checkpoint_path, 0.3, 0.3
    )
    reference = write_validation_evidence(
        "reference", config_path, authorized_checkpoint, 0.9, 0.8
    )
    inventory_path = config_path.parent / "legacy-inventory.json"
    fixture_images = list((config_path.parent / "images").rglob("*.png"))
    inventory_path.write_text(json.dumps({
        "schema_version": 1,
        "measured_on": "fixture",
        "labels_csv": {"path": str(Path(base_config["data"]["csv_path"])),
                        "sha256": sha256_file(Path(base_config["data"]["csv_path"])),
                        "bytes": Path(base_config["data"]["csv_path"]).stat().st_size},
        "reference_checkpoint": {"path": str(authorized_checkpoint),
                                  "sha256": sha256_file(authorized_checkpoint),
                                  "bytes": Path(authorized_checkpoint).stat().st_size,
                                  "training_provenance": "legacy_unavailable"},
        "nih_images": {"path": str(config_path.parent / "images"),
                        "file_count": len(fixture_images),
                        "bytes": sum(path.stat().st_size for path in fixture_images)},
    }), encoding="utf-8")
    selected = {
        key: bce[key]
        for key in ("name", "config", "checkpoint", "thresholds", "sha256")
    }
    reference.pop("name")
    reference.pop("label_cols")
    reference.pop("metrics")
    reference["legacy_inventory"] = str(inventory_path)
    reference["sha256"]["legacy_inventory"] = sha256_file(inventory_path)
    return {
        "schema_version": 1, "selection_split": "val",
        "selection_metric": "tuned_macro_f1", "tie_breaker": "macro_auprc",
        "reference": reference, "selected": selected,
        "candidates": [bce, posweight],
    }
~~~

Cover these exact cases:

~~~python
def test_empty_config_labels_resolve_from_csv_before_checkpoint_comparison(tmp_path):
    config_path, checkpoint = tiny_bundle(tmp_path, config_labels=[], checkpoint_labels=["a", "b"])
    report = evaluate_checkpoint(config_path, checkpoint, "val", scalar_threshold=0.5)
    assert report["label_cols"] == ["a", "b"]


def test_checkpoint_label_order_mismatch_is_rejected(tmp_path):
    config_path, checkpoint = tiny_bundle(tmp_path, config_labels=["a", "b"], checkpoint_labels=["b", "a"])
    with pytest.raises(ValueError, match="label order"):
        evaluate_checkpoint(config_path, checkpoint, "val", scalar_threshold=0.5)


def test_real_test_requires_matching_selection_record_before_loader(tmp_path, monkeypatch):
    config_path, checkpoint = tiny_bundle(tmp_path, config_labels=["a", "b"], checkpoint_labels=["a", "b"])
    loader_spy = Mock(side_effect=AssertionError("test loader must remain unopened"))
    monkeypatch.setattr("baseline.evaluation.build_split_loader", loader_spy)
    with pytest.raises(ValueError, match="selection record is required"):
        evaluate_checkpoint(config_path, checkpoint, "test", scalar_threshold=0.5)
    loader_spy.assert_not_called()
    different = tmp_path / "different.pt"
    shutil.copy2(checkpoint, different)
    record = frozen_selection_record(config_path, authorized_checkpoint=different)
    with pytest.raises(ValueError, match="does not authorize checkpoint"):
        evaluate_checkpoint(
            config_path, checkpoint, "test", scalar_threshold=0.5,
            selection_record=record, role="selected",
        )
    loader_spy.assert_not_called()


def test_json_sanitizer_handles_nested_non_finite_values():
    value = {"a": float("nan"), "b": [float("inf"), {"c": -float("inf")}], "d": 1.0}
    assert sanitize_for_json(value) == {
        "a": None, "b": [None, {"c": None}], "d": 1.0,
    }


@pytest.mark.parametrize("mutation", ["checkpoint", "config", "labels", "thresholds"])
def test_same_path_provenance_mutation_is_rejected_before_test_loader(tmp_path, monkeypatch, mutation):
    config_path, checkpoint = tiny_bundle(tmp_path, config_labels=["a", "b"], checkpoint_labels=["a", "b"])
    record = frozen_selection_record(config_path, authorized_checkpoint=checkpoint)
    selected = record["selected"]
    paths = {
        "checkpoint": Path(selected["checkpoint"]),
        "config": Path(selected["config"]),
        "labels": Path(yaml.safe_load(config_path.read_text(encoding="utf-8"))["data"]["csv_path"]),
        "thresholds": Path(selected["thresholds"]),
    }
    path = paths[mutation]
    with path.open("ab") as handle:
        handle.write(b"mutation")
    loader_spy = Mock(side_effect=AssertionError("test loader must remain unopened"))
    monkeypatch.setattr("baseline.evaluation.build_split_loader", loader_spy)
    with pytest.raises(ValueError, match="hash"):
        evaluate_checkpoint(
            config_path, checkpoint, "test", scalar_threshold=0.5,
            selection_record=record, role="selected",
        )
    loader_spy.assert_not_called()


def test_selection_rejects_empty_candidates_and_never_selects_reference(tmp_path):
    config_path, checkpoint = tiny_bundle(tmp_path, config_labels=["a", "b"], checkpoint_labels=["a", "b"])
    record = frozen_selection_record(config_path, authorized_checkpoint=checkpoint)
    with pytest.raises(ValueError, match="candidates"):
        validate_selection_record({**record, "candidates": []})
    ranked = validate_selection_record(record)
    assert ranked["selected"]["name"] in {"bce", "posweight"}
    assert ranked["selected"]["name"] != "reference"


def test_final_test_refuses_existing_output_directory_before_prediction(tmp_path, monkeypatch):
    config_path, checkpoint = tiny_bundle(tmp_path, config_labels=["a", "b"], checkpoint_labels=["a", "b"])
    output_dir = tmp_path / "final"
    output_dir.mkdir()
    marker = output_dir / "marker.txt"
    marker.write_bytes(b"keep")
    monkeypatch.setattr("baseline.evaluation.collect_predictions", Mock(side_effect=AssertionError("must not infer")))
    with pytest.raises(ValueError, match="output directory already exists"):
        run_final_test_batch(
            frozen_selection_record(config_path, authorized_checkpoint=checkpoint),
            output_dir,
        )
    assert marker.read_bytes() == b"keep"
~~~

Also test that an artifact with wrong label order or `fit_split: test` is
rejected. Use a schema-valid candidate list in `frozen_selection_record`; the
selection-schema tests below separately reject an empty candidate list.

Run: `& ".venv/Scripts/python.exe" -m pytest tests/test_evaluation.py -q`
Expected: FAIL because the evaluation module and selection-record gate do not exist.

- [ ] **Step 2: Implement label resolution, checkpoint loading, and prediction collection**

For `val`, `evaluate_checkpoint` calls `build_split_loader`, which resolves
`data.label_cols: []` from the CSV into a concrete ordered list, then calls
`load_checkpoint_bundle(checkpoint_path, expected_label_cols=resolved_labels,
device=device)`. For `test`,
the order is stricter: load the config; validate selection schema, role, exact
paths, and every config/checkpoint/threshold/labels/report hash; load checkpoint
metadata and compare its saved labels to the authorized record; only then call
`build_split_loader`. Compare the loader-resolved labels again before inference.
Missing or invalid authorization must never construct a test dataset/loader.
Resolve saved labels from top-level `checkpoint["label_cols"]`, falling back to
`checkpoint["config"]["data"]["label_cols"]`, and require exact equality. This
makes the existing empty-list NIH configs valid while still rejecting ontology
or order drift.

Before calling `build_split_loader(split=split, **data_config)`, copy the data
mapping and remove the training-only `val_split` key. Do not mutate the loaded
config. Add a test that uses the complete `data` block from `configs/baseline.yaml`
(with fixture paths substituted) so this path cannot fail on an unexpected
`val_split` keyword.

Load with `torch.load(path, map_location="cpu", weights_only=False)`, build the
saved model name with the resolved output count and `pretrained=False` (evaluation
must never download weights), load `model_state`, set eval
mode, and move it to the requested device. `collect_predictions` uses
`torch.no_grad()`, the engine's channels-last CUDA transfer, and returns CPU
targets plus sigmoid probabilities without fitting or applying thresholds.
Require the runtime config and checkpoint config to match on `model.name`,
`data.image_col`, `data.image_size`, and resolved label order. Batch size,
worker count, and output directory are operational fields and may differ.

- [ ] **Step 3: Implement a complete validation report schema**

Both scalar and artifact reports use this exact envelope:

~~~json
{
  "schema_version": 1,
  "evaluation_split": "val",
  "config": "configs/nih_f1_bce.yaml",
  "checkpoint": "D:/ChestXRobustRuns/nih_f1_bce/best.pt",
  "provenance": {
    "config_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "checkpoint_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "labels_csv_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "thresholds_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "training_provenance": "checkpoint_bound"
  },
  "label_cols": ["Atelectasis"],
  "threshold": {
    "mode": "artifact",
    "value": null,
    "artifact": "docs/experiments/nih-f1-bce-thresholds.json",
    "fit_split": "val"
  },
  "metrics": {
    "macro_f1": 0.0,
    "macro_auprc": 0.0,
    "micro_f1": 0.0,
    "per_label": []
  }
}
~~~

Numeric zeros and all-zero hashes are schema examples, not measurements.
Artifact reports require
`fit_split: val` and exact label order; scalar reports use
`{"mode": "scalar", "value": 0.5, "artifact": null, "fit_split": null}`.
`evaluate.py` requires exactly one threshold option and accepts only `--split
val`; it exposes no public test mode and refuses an existing output path.
`tune_thresholds.py` accepts only `--split val`, refuses an existing output path,
and writes the schema-versioned threshold artifact. Both CLIs write UTF-8 JSON,
recursively convert non-finite floats to `null`, and exit 2 for
schema/split/authorization/output-exists errors. The same write-once rule
applies to `select_candidate.py` and every validation artifact.

The threshold artifact adds `config`, `checkpoint`, `config_sha256`,
`checkpoint_sha256`, and `labels_csv_sha256` to the metric-core fields from Task
1. Artifact evaluation recomputes those hashes and refuses any mismatch before
prediction collection. Validation reports carry the same hashes and the
threshold artifact hash when applicable. Add tests that mutate checkpoint,
config, labels CSV, and threshold contents in place while preserving each path;
every mutation must be rejected.

For a new checkpoint, require `run_metadata.source_config_sha256` and
`run_metadata.labels_csv_sha256` to match the runtime files and emit
`training_provenance: checkpoint_bound`. The historical checkpoint lacks those
fields; allow it on validation as `training_provenance: legacy_unavailable` only
when `--legacy-inventory docs/experiments/2026-08-30-input-inventory.json` is
provided and its checkpoint and labels hashes equal both the files and the
frozen inventory above. Candidate
reports with legacy provenance are ineligible; only the historical reference may
carry that explicit limitation. Tests cover missing/mismatched candidate
provenance and the one exact legacy-reference exception.

- [ ] **Step 4: Implement and test validation-only selection**

`select_candidate.py` accepts only artifact-mode `val` reports with equal label
orders. It rejects fixed-threshold reports, `test` fields, inconsistent
checkpoint/threshold paths, and non-finite selection metrics. Rank only BCE and
`pos_weight` by `metrics.macro_f1` and then `metrics.macro_auprc`; the historical
reference is a control and never enters the candidate ranking. Add tests proving
the macro-AUPRC tie-break and every rejection above. Refuse an existing output
path so a frozen selection cannot be silently rewritten.

Expose the same checks as pure `validate_selection_record(record)` for the
final-test gate and unit tests. It returns a deep-copied normalized record only
after verifying the exact two names (`bce`, `posweight`), non-empty evidence,
hash/path consistency, finite ranking metrics, and a `selected` object identical
to the top-ranked candidate.

The selection record contains no test metrics and uses this structure:

~~~json
{
  "schema_version": 1,
  "selection_split": "val",
  "selection_metric": "tuned_macro_f1",
  "tie_breaker": "macro_auprc",
  "reference": {
    "validation_report": "docs/experiments/nih-f1-reference-validation-tuned.json",
    "config": "configs/baseline.yaml",
    "checkpoint": "checkpoints/imagenet_pretrained/best.pt",
    "thresholds": "docs/experiments/nih-f1-reference-thresholds.json",
    "legacy_inventory": "docs/experiments/2026-08-30-input-inventory.json",
    "sha256": {
      "config": "0000000000000000000000000000000000000000000000000000000000000000",
      "checkpoint": "0000000000000000000000000000000000000000000000000000000000000000",
      "thresholds": "0000000000000000000000000000000000000000000000000000000000000000",
      "labels_csv": "0000000000000000000000000000000000000000000000000000000000000000",
      "validation_report": "0000000000000000000000000000000000000000000000000000000000000000",
      "legacy_inventory": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "selected": {
    "name": "bce",
    "config": "configs/nih_f1_bce.yaml",
    "checkpoint": "D:/ChestXRobustRuns/nih_f1_bce/best.pt",
    "thresholds": "docs/experiments/nih-f1-bce-thresholds.json",
    "sha256": {
      "config": "0000000000000000000000000000000000000000000000000000000000000000",
      "checkpoint": "0000000000000000000000000000000000000000000000000000000000000000",
      "thresholds": "0000000000000000000000000000000000000000000000000000000000000000",
      "labels_csv": "0000000000000000000000000000000000000000000000000000000000000000",
      "validation_report": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "candidates": [
    {
      "name": "bce",
      "validation_report": "docs/experiments/nih-f1-bce-validation-tuned.json",
      "config": "configs/nih_f1_bce.yaml",
      "checkpoint": "D:/ChestXRobustRuns/nih_f1_bce/best.pt",
      "thresholds": "docs/experiments/nih-f1-bce-thresholds.json",
      "label_cols": ["Atelectasis"],
      "metrics": {"macro_f1": 0.31, "macro_auprc": 0.24},
      "sha256": {
        "config": "0000000000000000000000000000000000000000000000000000000000000000",
        "checkpoint": "0000000000000000000000000000000000000000000000000000000000000000",
        "thresholds": "0000000000000000000000000000000000000000000000000000000000000000",
        "labels_csv": "0000000000000000000000000000000000000000000000000000000000000000",
        "validation_report": "0000000000000000000000000000000000000000000000000000000000000000"
      }
    },
    {
      "name": "posweight",
      "validation_report": "docs/experiments/nih-f1-posweight-validation-tuned.json",
      "config": "configs/nih_f1_posweight.yaml",
      "checkpoint": "D:/ChestXRobustRuns/nih_f1_posweight/best.pt",
      "thresholds": "docs/experiments/nih-f1-posweight-thresholds.json",
      "label_cols": ["Atelectasis"],
      "metrics": {"macro_f1": 0.30, "macro_auprc": 0.25},
      "sha256": {
        "config": "0000000000000000000000000000000000000000000000000000000000000000",
        "checkpoint": "0000000000000000000000000000000000000000000000000000000000000000",
        "thresholds": "0000000000000000000000000000000000000000000000000000000000000000",
        "labels_csv": "0000000000000000000000000000000000000000000000000000000000000000",
        "validation_report": "0000000000000000000000000000000000000000000000000000000000000000"
      }
    }
  ]
}
~~~

The example scores and all-zero hashes above define field types only; execution
always copies measured validation values and computed SHA-256 digests.

The two candidate objects are mandatory and preserve the exact report path,
config, checkpoint, threshold artifact, content hashes, label order, and the two ranking values
read from that report. Every identity field in `selected` must match the
highest-ranked candidate; metrics and label order remain in the candidate
evidence object.
Tests reject an empty/missing candidate list, duplicate or unknown names,
different evidence paths or scores, and a `selected.name` that is not rank one.
Also prove that a higher-scoring historical reference never becomes `selected`.

- [ ] **Step 5: Implement the sole final-test entry point**

For `split == "test"`, `evaluate_checkpoint` requires a parsed selection record
and `role`. Validate schema version, `selection_split == "val"`, absence of any
test-result field, and exact config/checkpoint/threshold authorization for the
requested role. Recompute the recorded config, checkpoint, threshold, labels
CSV, validation-report, and any legacy-inventory hashes before opening the test loader; a same-path
content change is unauthorized. A scalar 0.5 report is authorized for either role; an artifact
report must use that role's recorded threshold path.

`run_final_test.py` is the only planned real-test command. It validates the
record once. Before constructing any test loader, it requires that the final
output directory is absent. It then runs reference and selected checkpoints at
fixed 0.5 and their recorded artifacts, finishes all inference, writes and
validates these four JSON files plus `manifest.json` inside a unique sibling
staging directory:
`nih-f1-reference-test-fixed.json`, `nih-f1-reference-test-tuned.json`,
`nih-f1-selected-test-fixed.json`, and `nih-f1-selected-test-tuned.json`.
The manifest records every report SHA-256. Immediately before promotion,
recheck that the final directory is absent, then publish the entire batch with
one same-volume directory rename. If authorization, inference, validation, or
promotion fails, exit 2, remove only the newly-created staging directory, and
leave the final directory absent. Add tests for
missing record, changed checkpoint, changed threshold path, and successful four
report generation from tiny fixtures. Add a test that pre-creates the final
directory, asserts exit 2 before prediction collection, and verifies every
existing byte is unchanged. Inject a directory-promotion failure and assert
that no final directory or staging directory remains.

Implement the reusable `run_final_test_batch(record, output_dir)` with the same
preconditions as the CLI: call `validate_selection_record`, reject an existing
`output_dir`, create a unique sibling staging directory with `tempfile`, call
the internal test evaluator four times (reference/selected x fixed/tuned),
write `manifest.json` with SHA-256s, fsync each file and the staging directory,
then call `staging_dir.replace(output_dir)` exactly once. On any exception,
remove only that staging directory and re-raise a `ValueError`/`OSError` without
touching an existing output directory.

- [ ] **Step 6: Run focused tests and commit**

~~~powershell
& ".venv/Scripts/python.exe" -m pytest tests/test_evaluation.py tests/test_metrics_thresholds.py -q
git add baseline/evaluation.py evaluate.py tune_thresholds.py select_candidate.py run_final_test.py tests/test_evaluation.py
git commit -m "feat: gate frozen checkpoint evaluation"
~~~

Expected: PASS, including empty-list label resolution and programmatic refusal
of every unauthorized test evaluation.

---

### Task 4: Add capped `pos_weight`, validation-AUPRC early stopping, and run metadata

**Files:**
- Create: `baseline/losses.py`
- Modify: `baseline/engine.py:1-103`
- Modify: `train.py:32-51`
- Modify: `smoke_test.py:25-66`
- Create: `tests/test_losses.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- `compute_pos_weight(positive_counts: torch.Tensor, negative_counts: torch.Tensor, cap: float = 20.0) -> torch.Tensor`
- `build_loss(loss_config: Mapping[str, Any] | None, positive_counts: torch.Tensor | None = None, negative_counts: torch.Tensor | None = None) -> nn.Module`
- `evaluate(model, loader, criterion, device, threshold=0.5, amp_enabled=False, label_cols=None) -> tuple[float, dict[str, Any]]`
- `fit(model, train_loader, val_loader, config, device, label_cols, run_provenance: Mapping[str, str]) -> list[dict[str, Any]]` with configured `training.loss`, `selection_metric`, `patience`, `min_delta`, and `max_hours`.
- `_set_seed(seed: int, deterministic: bool = False) -> None` applies the
  deterministic policy before loaders or models are constructed.
- `train.py --config PATH [--epochs N] [--max-hours HOURS] [--output-dir PATH]` for isolated one-epoch acceptance runs.

- [ ] **Step 1: Write failing loss tests**

~~~python
def test_default_loss_is_plain_bce():
    criterion = build_loss(None)
    assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    logits = torch.randn(3, 2, requires_grad=True)
    value = criterion(logits, torch.rand(3, 2))
    value.backward()
    assert torch.isfinite(logits.grad).all()


def test_pos_weight_is_finite_capped_and_train_count_based():
    weights = compute_pos_weight(
        torch.tensor([1.0, 10.0]), torch.tensor([100.0, 10.0]), cap=20.0
    )
    assert weights.tolist() == [20.0, 1.0]
    criterion = build_loss(
        {"name": "bce_pos_weight", "cap": 20.0},
        torch.tensor([1.0]),
        torch.tensor([100.0]),
    )
    assert criterion.pos_weight.tolist() == [20.0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pos_weight_loss_moves_its_buffer_to_cuda():
    criterion = build_loss(
        {"name": "bce_pos_weight", "cap": 20.0},
        torch.tensor([1.0]), torch.tensor([3.0]),
    ).to("cuda")
    assert criterion.pos_weight.device.type == "cuda"
    assert torch.isfinite(criterion(torch.zeros(1, 1, device="cuda"),
                                    torch.ones(1, 1, device="cuda")))


def test_unknown_loss_and_zero_positive_fail():
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss({"name": "asl"})
    with pytest.raises(ValueError, match="no positives"):
        compute_pos_weight(torch.tensor([0.0]), torch.tensor([3.0]))
~~~

Run: `& ".venv/Scripts/python.exe" -m pytest tests/test_losses.py -q`
Expected: FAIL because the loss factory is absent.

- [ ] **Step 2: Implement the loss factory**

Use `BCEWithLogitsLoss()` when `loss_config` is missing or its name is `bce`.
For `bce_pos_weight`, require one positive and negative count per label,
compute `negative / positive`, clamp to `[0, cap]`, and return
`BCEWithLogitsLoss(pos_weight=weights)`. Reject non-finite counts, non-positive
cap, mismatched lengths, and zero positives. Do not add ASL, focal loss,
samplers, or augmentation changes.

- [ ] **Step 3: Write failing engine tests for selection and stopping**

Patch `train_one_epoch` and `evaluate` with deterministic fakes so no NIH image
is loaded. Feed validation macro AUPRC values `0.20, 0.21, 0.205, 0.204` with
`patience=2` and `min_delta=0.0005`; assert `best.pt` records epoch 2 and the
loop stops after epoch 4. Assert the checkpoint stores `label_cols` and
`selection_metric: macro_auprc`. Assert the exact checkpoint metadata and
history envelope keys declared below, including all 14 finite training counts
and capped weights for a synthetic pos-weight loader. Assert source config and
labels hashes equal the fixture bytes at fit start. Add a regression using the
current `configs/baseline.yaml` training block with fixture data: its resolved
defaults are plain BCE, macro AUROC selection, no early stopping, zero
`min_delta`, and no wall-clock limit.
In a second test, patch `time.monotonic` so `max_hours` is exceeded after the
first completed epoch and assert the loop stops before epoch 2. Read
`history.json` after each mocked epoch and assert it is complete JSON; implement
this by writing `history.json.tmp` and replacing `history.json` atomically.

Run: `& ".venv/Scripts/python.exe" -m pytest tests/test_engine.py -q`
Expected: FAIL because the current engine always selects macro AUROC and has no
patience or wall-clock handling.

- [ ] **Step 4: Implement configured training behavior**

In `fit`, inspect the loss name first. Only `bce_pos_weight` calls
`training_label_counts(train_loader, label_cols)`; plain BCE passes no counts,
so existing tiny fixtures with absent classes remain valid. Call
`build_loss(config["training"].get("loss"), positive_counts, negative_counts).to(device)` and keep ordinary BCE
equivalent when the loss block is absent. Legacy defaults are exactly
`selection_metric="macro_auroc"`, `patience=None` (disabled), `min_delta=0.0`,
and `max_hours=None` (disabled), preserving the current baseline behavior.
Configured `patience` must be a positive integer, `min_delta` finite and
non-negative, and `max_hours` finite and positive. A metric improves only when finite and greater than
`best_score + min_delta`; save the first finite epoch unconditionally. Track
`bad_epochs`, stop at `bad_epochs >= patience`, and check wall time between
completed epochs. This is an epoch-boundary guard: a single epoch is never
interrupted, but no new epoch may start after the budget has elapsed. Pass
`label_cols` into `evaluate` so per-label history is emitted in checkpoint order.

Extend `save_checkpoint` with optional `label_cols` and `run_metadata` fields
while preserving the current five required keys. Checkpoint `run_metadata` has
exactly these required fields: `loss_name`, `positive_counts`,
`negative_counts`, `pos_weight`, `selection_metric`, `patience`, `min_delta`,
`max_hours`, `source_config`, `source_config_sha256`, `labels_csv`,
`labels_csv_sha256`, `deterministic`, `cudnn_benchmark`,
`cudnn_deterministic`, `deterministic_algorithms`, `elapsed_seconds`, and
`stop_reason`;
count/weight fields are `null` for plain BCE and 14 finite values for
`bce_pos_weight`. Save `history.json` atomically after every epoch with this
envelope:

~~~json
{
  "schema_version": 1,
  "label_cols": ["Atelectasis"],
  "run_metadata": {
    "loss_name": "bce_pos_weight",
    "positive_counts": [1],
    "negative_counts": [9],
    "pos_weight": [9.0],
    "selection_metric": "macro_auprc",
    "patience": 2,
    "min_delta": 0.0005,
    "max_hours": 2.0,
    "source_config": "configs/nih_f1_posweight.yaml",
    "source_config_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "labels_csv": "data/labels.csv",
    "labels_csv_sha256": "c0b1784dce4215b12396d2e610d70469f91b0c068d6096081fbfee184df93318",
    "deterministic": true,
    "cudnn_benchmark": false,
    "cudnn_deterministic": true,
    "deterministic_algorithms": true,
    "elapsed_seconds": 12.5,
    "stop_reason": "running"
  },
  "epochs": []
}
~~~

On exit, atomically rewrite the envelope with `stop_reason` equal to
`completed`, `early_stopping`, or `max_hours`, and final elapsed seconds. The
best checkpoint contains the selection metric and no test metrics. Update
`train.py` to pass resolved labels into `fit` and add optional positive
`--epochs`, `--max-hours`, and `--output-dir` overrides, applied to the resolved
config before training and therefore saved in checkpoints. Before constructing
loaders, `train.py` hashes the untouched config file and labels CSV into this
exact mapping (before applying any CLI overrides):

~~~python
run_provenance = {
    "source_config": str(Path(args.config).resolve()),
    "source_config_sha256": sha256_file(args.config),
    "labels_csv": str(Path(config["data"]["csv_path"]).resolve()),
    "labels_csv_sha256": sha256_file(config["data"]["csv_path"]),
}
~~~

`fit` copies it to every checkpoint and history write. The dynamic
`elapsed_seconds` and `stop_reason` values are updated atomically at each epoch
and on exit. Update
`smoke_test.py` to construct the configured loss conditionally from the training
frame and call `.to(device)`; default config still uses plain BCE.
At the start of `fit`, reject an existing non-empty `output_dir` with
`ValueError("output directory already exists and is non-empty")`; create only a
new or empty directory. Write each checkpoint through
`<name>.pt.tmp` followed by `Path.replace` so a crash cannot leave a truncated
`best.pt`/`last.pt`. Add an engine test that pre-creates a marker file and
asserts no checkpoint or history byte changes.
When `config.get("deterministic", False)` is true, `_set_seed` applies the
deterministic policy above and `train.py` must not enable cuDNN benchmarking.
Legacy configs default to `False`, preserving their historical runtime behavior.
Record the four deterministic flags in `run_metadata` exactly as observed after
initialization, so a relaxed CUDA run cannot be mistaken for the matched arm.

- [ ] **Step 5: Run regression tests and commit**

~~~powershell
& ".venv/Scripts/python.exe" -m pytest tests/test_losses.py tests/test_engine.py tests/test_baseline.py tests/test_nih.py -q
git add baseline/losses.py baseline/engine.py train.py smoke_test.py tests/test_losses.py tests/test_engine.py
git commit -m "feat: add imbalance loss and validation early stopping"
~~~

---

### Task 5: Materialize the controlled NIH configurations and pre-registered protocol

**Files:**
- Create: `configs/nih_f1_bce.yaml`
- Create: `configs/nih_f1_posweight.yaml`
- Create: `docs/experiments/2026-08-30-nih-f1-run.md`
- Create: `docs/experiments/2026-08-30-input-inventory.json`
- Read: `docs/experiments/nih-f1-reference-validation-fixed.json`

**Interfaces:** Both configs must be accepted by `train.py` and `smoke_test.py`.

- [ ] **Step 1: Add ordinary BCE configuration**

Create `configs/nih_f1_bce.yaml` with these exact values:

~~~yaml
seed: 42
device: auto
deterministic: true
output_dir: D:/ChestXRobustRuns/nih_f1_bce
data:
  csv_path: data/labels.csv
  image_root: data/raw/images
  image_col: path
  label_cols: []
  image_size: 224
  val_split: 0.2
  batch_size: 64
  num_workers: 4
  prefetch_factor: 2
  persistent_workers: true
model:
  name: resnet18
  pretrained: true
training:
  epochs: 10
  learning_rate: 0.0003
  weight_decay: 0.0001
  threshold: 0.5
  amp: true
  selection_metric: macro_auprc
  patience: 2
  min_delta: 0.0005
  max_hours: 2
  loss:
    name: bce
~~~

- [ ] **Step 2: Add the `pos_weight` configuration**

Create `configs/nih_f1_posweight.yaml` with every value above unchanged,
`output_dir: D:/ChestXRobustRuns/nih_f1_posweight`, and:

~~~yaml
  loss:
    name: bce_pos_weight
    cap: 20.0
~~~

- [ ] **Step 3: Write the pre-registered, append-only protocol**

Create `docs/experiments/2026-08-30-nih-f1-run.md` with separate headings
`Validation selection`, `Final NIH test`, `CheXpert readiness`, and `Data safety`.
Its provenance header must contain `plan_commit`, `config_commit`, and
`inventory_sha256`. Set `plan_commit` to the hash printed by the first command
in Step 4 and set `inventory_sha256` to the digest of the committed inventory
file before the first freeze commit. Make a second, metadata-only commit that
appends the first freeze commit hash as `config_commit`; both commits must exist
before any acceptance training. Protocol decision text is frozen before
training; Task 7 appends measured result sections without rewriting the
registered rules.
The prescribed ordering is:

1. Record git commit, Python/PyTorch/torchvision versions, GPU name, SHA-256 and
   size of `data/labels.csv` and the historical checkpoint, plus NIH image file
   count and summed bytes.
2. Run focused tests and one-epoch smoke checks for both configs.
3. Train both candidates without loading `test`.
4. Fit thresholds on `val` for each completed best checkpoint and create fixed
   0.5/frozen-threshold validation reports.
5. Run `select_candidate.py` and commit its validation-only selection record.
6. Only after that commit, run the historical reference and the selected
   candidate on `test` with fixed and frozen thresholds.
7. Record per-label positive counts, validation selection values, final test point
   estimates, incomplete runs, and CheXpert status. State that no confidence
    intervals or external robustness claims were produced.

Create `docs/experiments/2026-08-30-input-inventory.json` from the frozen values
in this plan, with no future-generated values. The implementation plan is a
separate planning commit made before this freeze commit; copy its full hash into
the protocol provenance header.

~~~json
{
  "schema_version": 1,
  "measured_on": "2026-08-30",
  "labels_csv": {
    "path": "data/labels.csv",
    "sha256": "c0b1784dce4215b12396d2e610d70469f91b0c068d6096081fbfee184df93318",
    "bytes": 10076512
  },
  "reference_checkpoint": {
    "path": "checkpoints/imagenet_pretrained/best.pt",
    "sha256": "be8ef6905f358488ed6d5dd2da8879c656084ee5d184795587a5ccef3584b972",
    "bytes": 134322153,
    "training_provenance": "legacy_unavailable"
  },
  "nih_images": {"path": "data/raw/images", "file_count": 112120, "bytes": 45057440698}
}
~~~

- [ ] **Step 4: Freeze configs and protocol after the plan commit, before any acceptance training**

~~~powershell
git rev-parse --verify HEAD
git add configs/nih_f1_bce.yaml configs/nih_f1_posweight.yaml docs/experiments/2026-08-30-nih-f1-run.md docs/experiments/2026-08-30-input-inventory.json
git commit -m "docs: define controlled NIH F1 runs"
~~~

Expected: the first command prints the already-committed plan hash, and the
freeze commit succeeds before either config is passed to `train.py`. Before
training, append only the resulting freeze commit hash to the protocol's
`config_commit` field and make a second metadata-only commit. Do not edit the
plan or protocol decision rules after this point; append only measured results
in the final record.

- [ ] **Step 5: Run one-batch checks and isolated one-epoch acceptance runs**

~~~powershell
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$labelsHash = (Get-FileHash data/labels.csv -Algorithm SHA256).Hash.ToLowerInvariant()
$checkpointHash = (Get-FileHash checkpoints/imagenet_pretrained/best.pt -Algorithm SHA256).Hash.ToLowerInvariant()
$nihImages = Get-ChildItem data/raw/images -Recurse -File | Measure-Object Length -Sum
if ($labelsHash -ne "c0b1784dce4215b12396d2e610d70469f91b0c068d6096081fbfee184df93318") {
    throw "labels.csv differs from the frozen pre-training inventory."
}
if ($checkpointHash -ne "be8ef6905f358488ed6d5dd2da8879c656084ee5d184795587a5ccef3584b972") {
    throw "Historical checkpoint differs from the frozen inventory."
}
if (($nihImages.Count -ne 112120) -or ($nihImages.Sum -ne 45057440698)) {
    throw "NIH image inventory differs from the frozen pre-training inventory."
}
$bceSmokeDir = "D:/ChestXRobustRuns/2026-08-30-smoke/nih_f1_bce"
$posWeightSmokeDir = "D:/ChestXRobustRuns/2026-08-30-smoke/nih_f1_posweight"
if ((Test-Path -LiteralPath $bceSmokeDir) -or (Test-Path -LiteralPath $posWeightSmokeDir)) {
    throw "Smoke output already exists; choose a new registered run date instead of overwriting it."
}
& ".venv/Scripts/python.exe" evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/imagenet_pretrained/best.pt --split val --threshold 0.5 --legacy-inventory docs/experiments/2026-08-30-input-inventory.json --output docs/experiments/nih-f1-reference-validation-fixed.json
& ".venv/Scripts/python.exe" -c "import json; m=json.load(open('docs/experiments/nih-f1-reference-validation-fixed.json', encoding='utf-8'))['metrics']; expected={'macro_auroc':0.8203765791291598,'micro_auroc':0.8679637209089436,'macro_f1':0.13576456439173093,'micro_f1':0.18699751861042183}; assert all(abs(m[k]-v) <= 1e-4 for k,v in expected.items()), (m, expected)"
$acceptanceClock = [System.Diagnostics.Stopwatch]::StartNew()
& ".venv/Scripts/python.exe" smoke_test.py --config configs/nih_f1_bce.yaml
& ".venv/Scripts/python.exe" smoke_test.py --config configs/nih_f1_posweight.yaml
& ".venv/Scripts/python.exe" train.py --config configs/nih_f1_bce.yaml --epochs 1 --max-hours 0.5 --output-dir $bceSmokeDir
& ".venv/Scripts/python.exe" -c "import json, pathlib; p=pathlib.Path(r'D:/ChestXRobustRuns/2026-08-30-smoke/nih_f1_bce'); h=json.load(open(p/'history.json', encoding='utf-8')); assert h['run_metadata']['stop_reason']=='completed', h['run_metadata']; assert len(h['epochs'])==1; assert (p/'best.pt').is_file() and (p/'last.pt').is_file()"
if ($acceptanceClock.Elapsed.TotalHours -ge 1) {
    throw "Acceptance budget exhausted after BCE; record pos_weight acceptance as incomplete."
}
& ".venv/Scripts/python.exe" train.py --config configs/nih_f1_posweight.yaml --epochs 1 --max-hours 0.5 --output-dir $posWeightSmokeDir
& ".venv/Scripts/python.exe" -c "import json, pathlib; p=pathlib.Path(r'D:/ChestXRobustRuns/2026-08-30-smoke/nih_f1_posweight'); h=json.load(open(p/'history.json', encoding='utf-8')); assert h['run_metadata']['stop_reason']=='completed', h['run_metadata']; assert len(h['epochs'])==1; assert (p/'best.pt').is_file() and (p/'last.pt').is_file()"
if ($acceptanceClock.Elapsed.TotalHours -ge 1) {
    throw "Acceptance runs exceeded one hour; record the day as incomplete."
}
~~~

Expected: both batch checks print 14-logit shapes and finite losses; both
one-epoch directories contain `best.pt`, `last.pt`, and valid `history.json`.
The pos-weight run records `bce_pos_weight` and finite weights. `max_hours` is
an epoch-boundary guard, so if one epoch itself crosses its budget, record the
overrun and do not start another acceptance item after the one-hour total gate.
Any missing or over-budget acceptance arm is a hard stop: record the experiment
as incomplete and do not execute Task 7 Step 4 threshold/report generation,
Step 5 selection, or Step 6 final test. `Task 4` implementation is already
complete at this point; do not describe this runtime stop as undoing that code.
The historical fixed-threshold assertion runs before `smoke_test.py` or either
one-epoch training command; a mismatch stops all GPU work.

---

### Task 6: Add a read-only CheXpert-small U-drive preflight

**Files:**
- Create: `baseline/chexpert_preflight.py`
- Create: `preflight_chexpert.py`
- Create: `configs/chexpert_preflight.yaml`
- Create: `tests/test_chexpert_preflight.py`

**Interfaces:**
- `check_chexpert_preflight(root: str | Path, sample_size: int = 32, required_free_bytes: int = 30 * 1024**3, archive_bytes: int | None = None, extracted_bytes: int | None = None, worker_count: int = 4, disk_usage_fn: Callable = shutil.disk_usage) -> dict[str, Any]`
- `preflight_chexpert.py --config PATH --report PATH`

- [ ] **Step 1: Write failing preflight tests**

Create a temporary CheXpert-like tree with `train.csv`, `valid.csv`, and two
valid JPGs. Define the source contract exactly:

~~~python
CHEXPERT_METADATA_COLUMNS = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"]
CHEXPERT_OBSERVATION_COLUMNS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]
~~~

Assert a valid fixture returns
`status: ok`, row/sample counts, and the future strict shared mapping. Add tests
for a missing header, a `Path` escaping the root, an unreadable JPG, and mocked
free space below the requirement; each returns `status: blocked` with a specific
error. Mock `disk_usage_fn` raising `OSError` and require
`status: blocked`, `reason: drive_unavailable`. Test both official path form
`CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg` and root-relative
form `train/patient00001/study1/view1_frontal.jpg`. No test accesses `E:` or the
network. Pass explicit fixture `archive_bytes` and `extracted_bytes`; assert the
computed required free space is their sum plus 10 GiB (or the configured
minimum, whichever is larger). A missing size manifest returns
`status: blocked`, `reason: size_metadata_missing`. Assert the report echoes `uncertainty_policy: zero`,
`patient_split_required: true`, and `conversion_enabled: false`, but creates no
split or converted labels.

Run: `& ".venv/Scripts/python.exe" -m pytest tests/test_chexpert_preflight.py -q`
Expected: FAIL because the preflight function is absent.

- [ ] **Step 2: Implement capacity, schema, path, and sample checks**

Implement this contract:

1. Resolve `root` non-strictly, verify its drive/anchor is mounted, call
   `disk_usage_fn(root.anchor)`, and require free bytes at least
   `max(required_free_bytes, archive_bytes + extracted_bytes + 10 GiB)`. Require
   explicit non-negative `archive_bytes` and `extracted_bytes` from the official
   release manifest; if either is missing, return `blocked`,
   `reason: size_metadata_missing`. Catch `OSError`/`FileNotFoundError` and return
   `status: blocked`, `reason: drive_unavailable` instead of propagating. Then
   return `blocked` if the dataset root is missing or not a directory. The
   earlier 62.56 GiB observation is not reused.
2. Require readable `train.csv` and `valid.csv` and all 14 source observation
   columns. Do not rename or convert labels.
3. Sample the first `sample_size` non-empty `Path` values from each CSV. Normalize
   separators to POSIX. Reject absolute paths, drive-qualified paths, and `..`.
   If the first component equals `root.name` (the official CSV form), strip
   exactly that component; otherwise treat the value as root-relative. Resolve
   the candidate and require `candidate.is_relative_to(root.resolve())` before
   calling `PIL.Image.open(candidate).verify()`.
4. Re-open the bounded sample through
   `ThreadPoolExecutor(max_workers=worker_count)` and report
   `parallel_read_ok`; any concurrent read failure blocks readiness. This is a
   removable-drive I/O check, not model inference.
5. Return JSON-serializable row counts, sampled/missing/unreadable counts, free
   and required bytes, and:

~~~python
STRICT_SHARED_LABEL_MAP = [
    {"canonical": "Atelectasis", "nih": "Atelectasis", "chexpert": "Atelectasis"},
    {"canonical": "Cardiomegaly", "nih": "Cardiomegaly", "chexpert": "Cardiomegaly"},
    {"canonical": "Consolidation", "nih": "Consolidation", "chexpert": "Consolidation"},
    {"canonical": "Edema", "nih": "Edema", "chexpert": "Edema"},
    {"canonical": "Pneumonia", "nih": "Pneumonia", "chexpert": "Pneumonia"},
    {"canonical": "Pneumothorax", "nih": "Pneumothorax", "chexpert": "Pneumothorax"},
    {"canonical": "Pleural_Effusion", "nih": "Effusion", "chexpert": "Pleural Effusion"},
]
~~~

Test that every `nih` value exists in `baseline.labels.LABEL_COLUMNS`, every
`chexpert` value exists in the source observation columns, and the normalized
`Pleural_Effusion` concept maps explicitly to both source names.

The CLI writes a report only with `--report` and exits 0 for `ok`, 2 for
`blocked`. It is read-only for the data tree; the report is the only write.

- [ ] **Step 3: Add the explicit preflight config**

Create `configs/chexpert_preflight.yaml`:

~~~yaml
root: E:/ChestXRobustData/CheXpert-v1.0-small
sample_size: 32
required_free_gib: 30.0
archive_bytes: null
extracted_bytes: null
worker_count: 4
future_shared_label_map:
  - {canonical: Atelectasis, nih: Atelectasis, chexpert: Atelectasis}
  - {canonical: Cardiomegaly, nih: Cardiomegaly, chexpert: Cardiomegaly}
  - {canonical: Consolidation, nih: Consolidation, chexpert: Consolidation}
  - {canonical: Edema, nih: Edema, chexpert: Edema}
  - {canonical: Pneumonia, nih: Pneumonia, chexpert: Pneumonia}
  - {canonical: Pneumothorax, nih: Pneumothorax, chexpert: Pneumothorax}
  - {canonical: Pleural_Effusion, nih: Effusion, chexpert: Pleural Effusion}
uncertainty_policy: zero
patient_split_required: true
conversion_enabled: false
~~~

The size fields must be populated from the official release manifest before an
`ok` result is possible; `null` deliberately yields `size_metadata_missing`.
The last three fields document a future contract. The preflight must not use
them to convert labels, create splits, or train. A later CheXpert experiment is
invalid until it persists patient-disjoint train/validation/test membership.

- [ ] **Step 4: Run tests and the real-drive preflight**

~~~powershell
& ".venv/Scripts/python.exe" -m pytest tests/test_chexpert_preflight.py -q
& ".venv/Scripts/python.exe" preflight_chexpert.py --config configs/chexpert_preflight.yaml --report docs/experiments/chexpert-preflight-2026-08-30.json
~~~

Expected: `status=ok` with measured rows/sample/readability, or `status=blocked`
with the exact access/capacity/layout reason. Neither outcome authorizes
conversion, training, or a robustness claim.

- [ ] **Step 5: Commit**

~~~powershell
git add baseline/chexpert_preflight.py preflight_chexpert.py configs/chexpert_preflight.yaml tests/test_chexpert_preflight.py
git commit -m "feat: add CheXpert staging preflight"
~~~

---

### Task 7: Execute the one-day protocol and perform the final verification gate

**Files:**
- Modify: `docs/experiments/2026-08-30-nih-f1-run.md`
- Read: `docs/experiments/2026-08-30-input-inventory.json`
- Create: `docs/experiments/nih-f1-selection.json`
- Create: `docs/experiments/nih-f1-bce-thresholds.json`
- Create: `docs/experiments/nih-f1-posweight-thresholds.json`
- Create: `docs/experiments/nih-f1-reference-thresholds.json`
- Create: `docs/experiments/nih-f1-reference-validation-fixed.json`
- Create: `docs/experiments/nih-f1-reference-validation-tuned.json`
- Create: `docs/experiments/nih-f1-bce-validation-fixed.json`
- Create: `docs/experiments/nih-f1-bce-validation-tuned.json`
- Create: `docs/experiments/nih-f1-posweight-validation-fixed.json`
- Create: `docs/experiments/nih-f1-posweight-validation-tuned.json`
- Create: `docs/experiments/chexpert-preflight-2026-08-30.json`
- Create: `docs/experiments/nih-f1-final-test/nih-f1-reference-test-fixed.json`
- Create: `docs/experiments/nih-f1-final-test/nih-f1-reference-test-tuned.json`
- Create: `docs/experiments/nih-f1-final-test/nih-f1-selected-test-fixed.json`
- Create: `docs/experiments/nih-f1-final-test/nih-f1-selected-test-tuned.json`
- Create: `docs/experiments/nih-f1-final-test/manifest.json`

**Timeboxes:** implementation and automated tests stop at four hours; the two
one-epoch acceptance runs stop at one hour total; full candidates have the
configured two-hour limit each; validation selection plus the final batch stop
at two hours; CheXpert preflight uses remaining time and never delays the NIH
result. If an earlier gate overruns, record it as incomplete and drop the
CheXpert preflight before weakening any leakage or test gate. Training limits
are checked at epoch boundaries; record any single-epoch overrun explicitly.
If either acceptance or full candidate is incomplete, stop before candidate
selection and do not generate the final NIH test batch; single-candidate
downgrade is not allowed.

- [ ] **Step 1: Run the complete test suite before GPU work**

~~~powershell
& ".venv/Scripts/python.exe" -m pytest -q
~~~

Expected: exit code 0. Any failure blocks training and is fixed in its earlier
task.

- [ ] **Step 2: Verify acceptance artifacts and record the environment**

Confirm both one-epoch smoke directories from Task 5 contain `best.pt`,
`last.pt`, and valid `history.json`, then run:

~~~powershell
git rev-parse HEAD
$checkpointDrive = Get-PSDrive D -ErrorAction Stop
if ($checkpointDrive.Free -lt 5GB) {
    throw "D: has less than 5 GiB free for isolated checkpoints."
}
$checkpointDrive | Select-Object Name, Used, Free
& ".venv/Scripts/python.exe" -c "import torch, torchvision, sys; print(sys.version); print(torch.__version__); print(torchvision.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
~~~

Expected: finite losses, 14-logit output, at least 5 GiB free on `D:`, and a
visible commit/version/device record.

- [ ] **Step 3: Train both candidates without touching NIH test**

~~~powershell
$bceRunDir = "D:/ChestXRobustRuns/nih_f1_bce"
$posWeightRunDir = "D:/ChestXRobustRuns/nih_f1_posweight"
if ((Test-Path -LiteralPath $bceRunDir) -or (Test-Path -LiteralPath $posWeightRunDir)) {
    throw "Registered output already exists; do not overwrite it."
}
& ".venv/Scripts/python.exe" train.py --config configs/nih_f1_bce.yaml
& ".venv/Scripts/python.exe" -c "import json, pathlib; p=pathlib.Path(r'D:/ChestXRobustRuns/nih_f1_bce'); h=json.load(open(p/'history.json', encoding='utf-8')); assert h['run_metadata']['stop_reason'] in ('completed','early_stopping'), h['run_metadata']; assert len(h['epochs']) >= 1; assert (p/'best.pt').is_file() and (p/'last.pt').is_file()"
& ".venv/Scripts/python.exe" train.py --config configs/nih_f1_posweight.yaml
& ".venv/Scripts/python.exe" -c "import json, pathlib; p=pathlib.Path(r'D:/ChestXRobustRuns/nih_f1_posweight'); h=json.load(open(p/'history.json', encoding='utf-8')); assert h['run_metadata']['stop_reason'] in ('completed','early_stopping'), h['run_metadata']; assert len(h['epochs']) >= 1; assert (p/'best.pt').is_file() and (p/'last.pt').is_file()"
~~~

Expected: each output directory contains `last.pt`, `best.pt`, and
`history.json`; best selection uses validation `macro_auprc`; neither command
loads `test`. A `max_hours` stop, missing checkpoint, empty history, or any
non-finite epoch is a hard stop and prevents Task 4 onward.

- [ ] **Step 4: Fit thresholds and create validation reports**

The historical fixed-threshold identity already passed before acceptance
training in Task 5. The unverified exploratory tuned macro F1 near `0.3058` is
not an acceptance target. Fit each threshold artifact exactly once:

~~~powershell
& ".venv/Scripts/python.exe" tune_thresholds.py --config configs/nih_f1_bce.yaml --checkpoint D:/ChestXRobustRuns/nih_f1_bce/best.pt --split val --output docs/experiments/nih-f1-bce-thresholds.json
& ".venv/Scripts/python.exe" tune_thresholds.py --config configs/nih_f1_posweight.yaml --checkpoint D:/ChestXRobustRuns/nih_f1_posweight/best.pt --split val --output docs/experiments/nih-f1-posweight-thresholds.json
& ".venv/Scripts/python.exe" tune_thresholds.py --config configs/baseline.yaml --checkpoint checkpoints/imagenet_pretrained/best.pt --split val --legacy-inventory docs/experiments/2026-08-30-input-inventory.json --output docs/experiments/nih-f1-reference-thresholds.json
~~~

Run `evaluate.py --split val` for each checkpoint at fixed 0.5 and with its
frozen artifact using these exact commands:

~~~powershell
& ".venv/Scripts/python.exe" evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/imagenet_pretrained/best.pt --split val --thresholds docs/experiments/nih-f1-reference-thresholds.json --legacy-inventory docs/experiments/2026-08-30-input-inventory.json --output docs/experiments/nih-f1-reference-validation-tuned.json
& ".venv/Scripts/python.exe" evaluate.py --config configs/nih_f1_bce.yaml --checkpoint D:/ChestXRobustRuns/nih_f1_bce/best.pt --split val --threshold 0.5 --output docs/experiments/nih-f1-bce-validation-fixed.json
& ".venv/Scripts/python.exe" evaluate.py --config configs/nih_f1_bce.yaml --checkpoint D:/ChestXRobustRuns/nih_f1_bce/best.pt --split val --thresholds docs/experiments/nih-f1-bce-thresholds.json --output docs/experiments/nih-f1-bce-validation-tuned.json
& ".venv/Scripts/python.exe" evaluate.py --config configs/nih_f1_posweight.yaml --checkpoint D:/ChestXRobustRuns/nih_f1_posweight/best.pt --split val --threshold 0.5 --output docs/experiments/nih-f1-posweight-validation-fixed.json
& ".venv/Scripts/python.exe" evaluate.py --config configs/nih_f1_posweight.yaml --checkpoint D:/ChestXRobustRuns/nih_f1_posweight/best.pt --split val --thresholds docs/experiments/nih-f1-posweight-thresholds.json --output docs/experiments/nih-f1-posweight-validation-tuned.json
~~~

All six reports must say `evaluation_split: val`. Do not generate or open a real
NIH test report yet.

- [ ] **Step 5: Commit the pre-test selection record**

~~~powershell
& ".venv/Scripts/python.exe" select_candidate.py --reference-report docs/experiments/nih-f1-reference-validation-tuned.json --bce-report docs/experiments/nih-f1-bce-validation-tuned.json --posweight-report docs/experiments/nih-f1-posweight-validation-tuned.json --output docs/experiments/nih-f1-selection.json
git add docs/experiments/nih-f1-selection.json docs/experiments/*validation*.json docs/experiments/*thresholds.json
git commit -m "exp: freeze NIH validation selection"
~~~

Inspect that the record has `selection_split: val`, selected checkpoint and
threshold artifact, and no `test` key. This commit is the gate before any real
NIH test command.

- [ ] **Step 6: Generate the one allowed final NIH test batch**

Only after Step 5 succeeds, invoke the sole test entry point:

~~~powershell
& ".venv/Scripts/python.exe" run_final_test.py --selection-record docs/experiments/nih-f1-selection.json --output-dir docs/experiments/nih-f1-final-test
~~~

Expected: the final directory appears in one rename with the four declared JSON
files and a hash manifest, corresponding exactly to the record's reference and
selected entries. No manual path
substitution is allowed. Append macro/micro F1, AUROC, AUPRC, every per-label
count/metric, and the exploratory point-estimate caveat to the run record.

- [ ] **Step 7: Verify tests and data safety**

~~~powershell
& ".venv/Scripts/python.exe" -m pytest -q
$labelsPresent = Test-Path data/labels.csv
$imagesPresent = Test-Path data/raw/images
$checkpointPresent = Test-Path checkpoints/imagenet_pretrained/best.pt
$labelsHashAfter = (Get-FileHash data/labels.csv -Algorithm SHA256).Hash.ToLowerInvariant()
$checkpointHashAfter = (Get-FileHash checkpoints/imagenet_pretrained/best.pt -Algorithm SHA256).Hash.ToLowerInvariant()
$nihImagesAfter = Get-ChildItem data/raw/images -Recurse -File | Measure-Object Length -Sum
$nihImagesAfter | Select-Object Count, Sum
if (-not ($labelsPresent -and $imagesPresent -and $checkpointPresent)) {
    throw "A frozen NIH input is missing."
}
if ($labelsHashAfter -ne "c0b1784dce4215b12396d2e610d70469f91b0c068d6096081fbfee184df93318") {
    throw "labels.csv changed during the experiment."
}
if ($checkpointHashAfter -ne "be8ef6905f358488ed6d5dd2da8879c656084ee5d184795587a5ccef3584b972") {
    throw "Historical checkpoint changed during the experiment."
}
if (($nihImagesAfter.Count -ne 112120) -or ($nihImagesAfter.Sum -ne 45057440698)) {
    throw "NIH image inventory changed during the experiment."
}
git status --short
~~~

Expected: tests exit 0, all three paths return True, and no existing NIH file or
checkpoint changed. Recomputed hashes and NIH image count/sum exactly equal the
values registered in Task 5.

- [ ] **Step 8: Commit the measured record**

~~~powershell
git add docs/experiments
git commit -m "exp: record one-day NIH F1 results"
~~~

The record must state external status `ok`, `blocked`, or `not attempted`, identify
the next BRAX/VinDr action, and defer NIH deletion, CheXpert conversion, ASL,
self-supervised pretraining, ensembles, DICOM support, and multi-seed confirmation.

## Self-Review Checklist

- Threshold fitting, per-label metrics, named split loading, `pos_weight`,
  validation early stopping, two controlled configs, CheXpert preflight, and
  final test sequencing each have a task.
- No task permits real NIH test output before the selection record is committed.
- `evaluate.py` is validation-only; `run_final_test.py` verifies content hashes
  and publishes one write-once final directory with a single rename.
- Patient-addressable CSVs without a persisted split are rejected by both
  training and named evaluation loaders.
- The historical checkpoint is treated as a reference, not retroactively called
  F1-optimal.
- Degenerate labels have explicit `None`/macro-exclusion behavior.
- Ordinary BCE remains the default for legacy configs and existing tests.
- The `pos_weight` criterion moves to the model device and records its exact
  training-only counts and weights.
- CheXpert preflight cannot be mistaken for external robustness evidence.
- CheXpert's normalized effusion concept has explicit NIH/source mappings, and
  an unavailable drive becomes a machine-readable blocker.
- No task deletes, moves, archives, or overwrites NIH data.
- Checkpoint writes stay off the removable drive.
- Experiment configs and the protocol are committed before acceptance training.
- The hard budget is explicit: ten epochs, two hours per run, patience two, and
  one predeclared final test batch after selection.
