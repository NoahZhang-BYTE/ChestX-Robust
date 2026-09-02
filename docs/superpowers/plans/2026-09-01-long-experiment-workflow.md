# Long Experiment Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish a restart-safe, validation-only model-selection workflow for B2/B3/B4 that preserves artifacts, gates unsafe training, and stops before B5 pending human confirmation.

**Architecture:** Keep the current training implementation and B2 output untouched while B2 is running. Add independent analysis and orchestration utilities that use atomic JSON/CSV writes, stage markers, immutable experiment directories, and explicit safety gates. Reuse the existing evaluation and frozen-test contracts instead of changing split or metric semantics.

**Tech Stack:** Python 3.12, PyTorch, pandas, NumPy, scikit-learn, Pillow, Matplotlib, YAML, Windows PSAPI/Event Log where available.

## Global Constraints

- Do not change DenseNet121, B2 batch size 32, AMP, optimizer, learning rate, loss, split, augmentation, CUDA/PyTorch installation, drivers, or Windows settings.
- Do not start a duplicate B2 process while the current process exists.
- Validation may select checkpoints and thresholds; test is frozen-only and never tunes.
- Existing output directories and artifacts must not be overwritten.
- Any unsafe commit headroom, CUDA/data/checkpoint/hardware error blocks later formal training.
- B5 is never auto-started; stop for human confirmation.

### Task 1: Workflow state and registry

**Files:** Create `workflow_state.py`, `workflow_runner.py`, `workflow_state.json`, `outputs/experiment_registry.csv`, `EXPERIMENT_LOG.md`, `WORKFLOW_BLOCKED.md` only when blocked. Tests in `tests/test_workflow_state.py`.

- [ ] Add atomic JSON/CSV helpers and stage transitions that skip completed stages and refuse invalid transitions.
- [ ] Add B2 process detection and a non-interfering monitor mode.
- [ ] Test atomic replacement, completed-stage skip, and invalid state handling.

### Task 2: B2 post-training analysis

**Files:** Create `analyze_b2_training.py`, `tests/test_analyze_b2_training.py`.

- [ ] Validate epoch 10, finite metrics, checkpoint readability, and history presence before analysis.
- [ ] Write `post_analysis/training_curves.png`, `epoch_summary.csv`, and `analysis_summary.json` with label index/name columns and trend/overfit summaries.

### Task 3: B2 validation tuning, frozen test, hard labels, co-occurrence

**Files:** Create `tune_b2_thresholds.py`, `analyze_hard_labels.py`, `analyze_cooccurrence.py`, `tests/test_b2_postprocessing.py`.

- [ ] Reuse validation inference with `torch.inference_mode()` and 0.01..0.99 threshold candidates, tie-breaking by distance to 0.5.
- [ ] Save validation arrays and threshold metrics without touching test.
- [ ] Run frozen test only after validation thresholds exist.
- [ ] Generate Infiltration/Pneumonia error tables and train-split co-occurrence outputs.

### Task 4: B3/B4 definitions

**Files:** Create `baseline/asl.py`, `configs/formal_b3_densenet121_asl.yaml`, `configs/formal_b4_*.yaml`, `tests/test_asl.py`.

- [ ] Implement and test ASL in isolation.
- [ ] Preserve B2 settings and change only loss for B3.
- [ ] Define B4 scheduler/early-stopping config without starting it automatically.

### Task 5: Comparison and guarded execution

**Files:** Create `compare_b2_b3.py`, extend `workflow_runner.py`, `tests/test_workflow_runner.py`.

- [ ] Compare validation/test metrics and per-label outcomes using the stated priority rules.
- [ ] Require commit/WHEA safety gate before B3/B4.
- [ ] Record blocked exceptions and stop; never launch B5.

### Task 6: Verification

- [ ] Run focused tests, full pytest, and py_compile.
- [ ] Confirm no formal B2/B3/B4 process was started by the verification commands.
- [ ] Update state only for stages proven complete by artifacts.
