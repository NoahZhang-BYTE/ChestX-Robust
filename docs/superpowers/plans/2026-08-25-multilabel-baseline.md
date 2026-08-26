# Multi-Label Chest X-Ray Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, runnable multi-label chest X-ray training baseline with configurable data loading, ResNet18/DenseNet121 backbones, BCEWithLogitsLoss, train/validation loops, AUROC/F1 metrics, and checkpointing.

**Architecture:** A small `baseline` Python package owns configuration, CSV/image datasets, model construction, metrics, and training engine. `train.py` wires those components from a YAML file; the CSV schema stays configurable because the competition has not published a final submission contract.

**Tech Stack:** Python 3.12, PyTorch, torchvision, Pillow, pandas, scikit-learn, PyYAML, pytest.

## Global Constraints

- The implementation must run in the existing `.venv` environment with CUDA when available and CPU when not.
- The model must emit one logit per configured label and use `BCEWithLogitsLoss`.
- Dataset input is a CSV containing one image path column and one binary column per label; both names and root are configurable.
- Validation metrics must include AUROC and F1, with safe handling when a validation label has only one class.
- Checkpoints must include model state, optimizer state, epoch, configuration, and validation metrics.

### Task 1: Test the baseline contracts

**Files:**
- Create: `tests/test_baseline.py`

**Interfaces:**
- Tests import `baseline.data.MultiLabelImageDataset`, `baseline.models.build_model`, `baseline.metrics.multilabel_metrics`, and `baseline.engine.save_checkpoint`.

- [ ] **Step 1: Write failing tests**

Create tests that build temporary PNG images and a CSV, assert the dataset returns image tensors and binary multi-label targets, assert ResNet18 emits `(batch, labels)` logits, assert metrics return AUROC/F1 keys for a single-class edge case, and assert checkpoint files contain required state keys.

- [ ] **Step 2: Run the tests to verify they fail**

Run `pytest -q`. Expected result: collection fails with `ModuleNotFoundError: No module named 'baseline'`.

### Task 2: Implement data and model contracts

**Files:**
- Create: `baseline/__init__.py`
- Create: `baseline/data.py`
- Create: `baseline/models.py`

**Interfaces:**
- `MultiLabelImageDataset(frame, image_root, image_col, label_cols, transform=None)` returns `(image_tensor, target_tensor)`.
- `build_dataloaders(csv_path, image_root, image_col, label_cols, val_split, image_size, batch_size, num_workers, seed)` returns `(train_loader, val_loader, resolved_label_cols)`.
- `build_model(name, num_classes, pretrained=False)` returns a torchvision backbone with a multi-label linear head.

- [ ] **Step 1: Implement the minimal dataset and model**
- [ ] **Step 2: Run the focused tests and verify they pass**

Run `pytest -q tests/test_baseline.py -k 'dataset or model'`.

### Task 3: Implement metrics and checkpoint behavior

**Files:**
- Create: `baseline/metrics.py`
- Create: `baseline/engine.py`

**Interfaces:**
- `multilabel_metrics(targets, probabilities, threshold=0.5)` returns `macro_auroc`, `micro_auroc`, `macro_f1`, `micro_f1`, and `sample_f1`.
- `save_checkpoint(path, model, optimizer, epoch, config, metrics)` writes a PyTorch checkpoint dictionary.
- `train_one_epoch(...)`, `evaluate(...)`, and `fit(...)` implement the BCEWithLogitsLoss loop and save `last.pt` plus the best validation checkpoint.

- [ ] **Step 1: Implement safe metrics and checkpoint saving**
- [ ] **Step 2: Run focused tests and verify they pass**

Run `pytest -q tests/test_baseline.py -k 'metric or checkpoint'`.

### Task 4: Add config-driven training entrypoint and documentation

**Files:**
- Create: `baseline/config.py`
- Create: `configs/baseline.yaml`
- Create: `train.py`
- Create: `requirements.txt`
- Create: `README.md`

**Interfaces:**
- `load_config(path)` reads YAML into a dictionary.
- `train.py --config configs/baseline.yaml` resolves the device, infers label count, trains, validates, and writes checkpoints under the configured output directory.

- [ ] **Step 1: Implement config loading and CLI wiring**
- [ ] **Step 2: Run all tests**

Run `pytest -q`.

- [ ] **Step 3: Run a synthetic end-to-end smoke test**

Create a temporary two-label image CSV and run `train.py` for one epoch on CPU or CUDA. Expected result: `last.pt` and `best.pt` are created and epoch metrics are printed.
