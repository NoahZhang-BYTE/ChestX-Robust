# NIH ChestX-ray14 Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert official NIH ChestX-ray14 metadata into a portable, patient-safe 14-label CSV and load it through the existing multi-label training pipeline.

**Architecture:** Add one canonical label module, one NIH preparation/validation module, and two small CLI entry points. The loader will use persisted `split` values whenever present while retaining deterministic legacy CSV splitting, so the training model and `BCEWithLogitsLoss` stay unchanged.

**Tech Stack:** Python 3.12, pandas, Pillow, PyTorch, torchvision, pytest, PyYAML.

## Global Constraints

- Use exactly the NIH 14 labels, defined once in `baseline.labels.LABEL_COLUMNS`; do not model `No Finding`.
- Write relative image paths and labels as numeric 0/1 multi-hot columns.
- Persist fixed `train`, `val`, and `test` split values with seed 42; never split NIH images dynamically during training.
- Derive splits by patient when `Patient ID` is present; reject leakage rather than silently accepting it.
- Keep model logits and `nn.BCEWithLogitsLoss`; do not add softmax, CrossEntropyLoss, augmentation systems, or rebalancing logic.
- Support Windows paths through `pathlib.Path`.
- Do not download, fabricate, or modify real NIH images or labels.

---

## File Structure

- Create `baseline/labels.py`: canonical NIH label names and class count.
- Create `baseline/nih.py`: metadata conversion, split generation, labels CSV validation, and distribution reporting.
- Create `prepare_nih.py`: command-line conversion wrapper.
- Create `validate_data.py`: command-line validation/reporting wrapper.
- Create `smoke_test.py`: command-line one-batch forward/loss wrapper.
- Modify `baseline/data.py`: use stored `split` values and default to canonical labels when the NIH schema is present.
- Modify `configs/baseline.yaml`: use `path` and canonical label names for prepared NIH data.
- Modify `README.md`: document preparation, validation, smoke test, and training commands.
- Modify `tests/test_baseline.py`: retain legacy coverage and add split-aware loader regression coverage.
- Create `tests/test_nih.py`: temporary NIH metadata/image fixtures for converter, validator, and smoke pipeline coverage.

### Task 1: Canonical Labels

**Files:**
- Create: `baseline/labels.py`
- Modify: `tests/test_baseline.py`

**Interfaces:**
- Produces: `LABEL_COLUMNS: tuple[str, ...]` and `NUM_CLASSES: int`.
- Consumed by: NIH conversion, validation, loader, config, and tests.

- [ ] **Step 1: Write the failing test**

```python
from baseline.labels import LABEL_COLUMNS, NUM_CLASSES


def test_nih_label_definition_has_the_fixed_14_class_order():
    assert LABEL_COLUMNS == (
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
    )
    assert NUM_CLASSES == 14
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_baseline.py -q`

Expected: failure because `baseline.labels` does not yet exist.

- [ ] **Step 3: Write the minimal implementation**

```python
LABEL_COLUMNS = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
)
NUM_CLASSES = len(LABEL_COLUMNS)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_baseline.py -q`

Expected: PASS.

### Task 2: NIH Metadata Conversion and Patient Splits

**Files:**
- Create: `baseline/nih.py`
- Create: `prepare_nih.py`
- Create: `tests/test_nih.py`

**Interfaces:**
- Consumes: `prepare_nih_metadata(metadata_path, image_root, output_path, seed=42, train_list_path=None, test_list_path=None)`.
- Produces: an on-disk CSV with `image_id`, `path`, `split`, `patient_id`, then `LABEL_COLUMNS`; returns the DataFrame.

- [ ] **Step 1: Write the failing tests**

```python
def test_prepare_nih_writes_multihot_labels_and_patient_safe_splits(tmp_path):
    metadata = write_nih_fixture(tmp_path)
    frame = prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")

    assert list(frame.columns) == ["image_id", "path", "split", "patient_id", *LABEL_COLUMNS]
    assert frame.loc[frame.image_id.eq("a.png"), ["Cardiomegaly", "Effusion"]].iloc[0].tolist() == [1, 1]
    assert frame.loc[frame.image_id.eq("b.png"), list(LABEL_COLUMNS)].iloc[0].tolist() == [0] * NUM_CLASSES
    assert set(frame.split) <= {"train", "val", "test"}
    assert frame.groupby("patient_id").split.nunique().max() == 1


def test_prepare_nih_rejects_unknown_non_empty_finding(tmp_path):
    metadata = write_nih_fixture(tmp_path, finding="Imaginary disease")
    with pytest.raises(ValueError, match="Unknown finding labels"):
        prepare_nih_metadata(metadata, tmp_path / "images", tmp_path / "labels.csv")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_nih.py -q`

Expected: import failure because `baseline.nih` does not yet exist.

- [ ] **Step 3: Implement conversion and CLI**

Implement `prepare_nih_metadata` with required metadata columns `Image Index`, `Finding Labels`, and `Patient ID`. Resolve each image beneath `image_root`, reject missing/ambiguous paths, split unique patient IDs deterministically with `numpy.random.default_rng(seed)`, convert pipe-separated labels to 0/1 fields, and write CSV. Add an argparse CLI with `--metadata`, `--image-root`, `--output`, `--seed`, `--train-list`, and `--test-list` arguments.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_nih.py -q`

Expected: PASS.

### Task 3: Strict Data Validation and Statistics

**Files:**
- Modify: `baseline/nih.py`
- Create: `validate_data.py`
- Modify: `tests/test_nih.py`

**Interfaces:**
- Consumes: `validate_labels_csv(csv_path, data_root)`.
- Produces: validated DataFrame and printed `train`/`val`/`test` counts plus positive count/rate for every canonical label; raises `ValueError` for malformed content or leakage.

- [ ] **Step 1: Write failing tests**

```python
def test_validator_reports_split_counts_and_rejects_patient_leakage(tmp_path, capsys):
    labels_csv = prepare_fixture_labels_csv(tmp_path)
    validate_labels_csv(labels_csv, tmp_path / "images")
    assert "Train:" in capsys.readouterr().out

    leaked = pd.read_csv(labels_csv)
    leaked.loc[1, "split"] = "test"
    leaked.to_csv(labels_csv, index=False)
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_labels_csv(labels_csv, tmp_path / "images")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_nih.py -q`

Expected: failure because `validate_labels_csv` does not yet exist.

- [ ] **Step 3: Implement validation and CLI**

Validate the required schema, non-null cells, label values limited to 0/1, allowed split values, unique `image_id`, each relative path existing below `data_root`, and disjoint patient-ID sets across all pairs of splits. Print image totals and each label's `positive / total = percent` for every split. Add CLI positional/options for `--csv` and `--data-root`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_nih.py -q`

Expected: PASS.

### Task 4: Split-Aware Data Loading

**Files:**
- Modify: `baseline/data.py`
- Modify: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `build_dataloaders(csv_path, image_root, image_col="path", label_cols=LABEL_COLUMNS, ...)`.
- Produces: train and validation loaders built from persisted `split` rows; existing random split fallback when the `split` column is absent.

- [ ] **Step 1: Write a failing test**

```python
def test_dataloaders_use_persisted_train_and_val_splits(tmp_path):
    csv_path = write_split_fixture(tmp_path)
    train_loader, val_loader, label_cols = build_dataloaders(
        csv_path=csv_path, image_root=tmp_path, image_col="path", label_cols=["finding_a", "finding_b"], batch_size=2
    )

    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1
    assert label_cols == ["finding_a", "finding_b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_baseline.py -q`

Expected: failure because the current loader ignores `split` and performs a random split.

- [ ] **Step 3: Implement minimal loader change**

If `split` is a CSV column, require non-empty `train` and `val` rows and select them directly. Ignore `test` for training. Otherwise preserve `_split_frame` behavior. When `label_cols` is empty, use `LABEL_COLUMNS` if every canonical label exists, otherwise retain legacy inference excluding path/metadata columns.

- [ ] **Step 4: Run test to verify it passes**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_baseline.py -q`

Expected: PASS.

### Task 5: One-Batch Pipeline Smoke Test and Documentation

**Files:**
- Create: `smoke_test.py`
- Modify: `configs/baseline.yaml`
- Modify: `README.md`
- Modify: `tests/test_nih.py`

**Interfaces:**
- Consumes: an existing YAML config and prepared `labels.csv`.
- Produces: printed image/label/logit shapes and dtypes, then scalar BCE-with-logits loss; nonzero exit on shape mismatch.

- [ ] **Step 1: Write the failing smoke test**

```python
def test_smoke_pipeline_returns_matching_nih_target_and_logit_shapes(tmp_path):
    config_path = write_smoke_config(tmp_path)
    result = run_smoke_test(config_path)

    assert result.images_shape == (2, 3, 32, 32)
    assert result.labels_shape == (2, NUM_CLASSES)
    assert result.logits_shape == result.labels_shape
    assert result.loss >= 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_nih.py -q`

Expected: import failure because `smoke_test.py` and `run_smoke_test` do not yet exist.

- [ ] **Step 3: Implement smoke command and configure NIH defaults**

Load config/dataloaders/model, pull one training batch, calculate logits and `nn.BCEWithLogitsLoss`, reject unequal shapes, and print `images.shape`, `labels.shape`, `images.dtype`, `labels.dtype`, `labels[:3]`, `logits.shape`, and `loss`. Configure `data.image_col: path` and all canonical labels. Document exact prepare, validate, smoke, and training PowerShell commands.

- [ ] **Step 4: Run the full suite**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest -q`

Expected: all legacy and NIH adapter tests PASS.

- [ ] **Step 5: Verify CLI help is usable**

Run: `& '.\\.venv\\Scripts\\python.exe' prepare_nih.py --help`

Run: `& '.\\.venv\\Scripts\\python.exe' validate_data.py --help`

Run: `& '.\\.venv\\Scripts\\python.exe' smoke_test.py --help`

Expected: each command exits 0 and lists its arguments.

## Plan Self-Review

- Coverage: Tasks 1-5 cover canonical classes, conversion, fixed patient-level splitting, schema/image/leakage checks, class-distribution reporting, split-aware loading, forward/loss smoke verification, documentation, and legacy regression coverage.
- Scope: No model, loss, optimization, augmentation, or download behavior outside the stated data-adapter requirement is included.
- Consistency: All tasks use `LABEL_COLUMNS`, `prepare_nih_metadata`, `validate_labels_csv`, and the persisted `split` contract consistently.
