# ChestX-Robust Model and Robustness Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a leakage-safe, reproducible F1-focused NIH benchmark, then evaluate progressively stronger imbalance, calibration, augmentation, and cross-domain robustness strategies without changing the data contract prematurely.

**Architecture:** Keep the current NIH 14-label data and the ImageNet-initialized ResNet18 as the controlled reference. Separate prediction collection, validation-only threshold fitting, model selection, and final test execution. Change one training factor at a time, retain independent output directories, and promote a method only after validation evidence and (where applicable) multi-seed or external evidence.

**Tech Stack:** Python 3.12.10, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy, pandas, scikit-learn, Pillow, PyYAML, pytest, PowerShell 7.6, CUDA GPU when available.

## Material Passport

- Origin Skill: experiment-agent (plan) + scientific-brainstorming (prioritisation)
- Origin Date: 2026-08-30
- Verification Status: UNVERIFIED (plan only; no new code or training)
- Current stage: understanding; the north star is discriminating evidence per unit compute.

## Global Constraints

- This planning turn must not edit Python/YAML implementation files or start a training/inference run.
- Do not delete, move, overwrite, or rename `data/labels.csv`, `data/raw/images`, or any existing checkpoint.
- Preserve the current 14 NIH labels and persisted patient-level split until the hidden evaluation ontology is confirmed.
- Fit thresholds only on `val`; `test` and external data may consume a frozen artifact but may never select a model or threshold.
- The historical reference is `checkpoints/imagenet_pretrained/best.pt`; its SHA-256 is `be8ef6905f358488ed6d5dd2da8879c656084ee5d184795587a5ccef3584b972`.
- The current labels file SHA-256 is `c0b1784dce4215b12396d2e610d70469f91b0c068d6096081fbfee184df93318`.
- New screen runs use the same ResNet18/ImageNet initialization, seed 42, persisted split, image size, optimizer, and output contract unless a candidate explicitly declares a single changed factor.
- A screen run is bounded by 10 epochs or 2 wall-clock hours, with validation macro AUPRC early stopping (`patience: 2`, `min_delta: 0.0005`) after the engine supports it.
- Every run must write an immutable config, source hashes, resolved label order, epoch history, stop reason, and checkpoint metadata to a distinct output directory.
- A failed gate, non-finite loss, missing positive class, split mismatch, or incomplete required arm stops the dependent phase; do not infer a ranking from an incomplete run.
- External preflight is readiness evidence only. No external robustness claim is allowed until a frozen model is evaluated on an independent source.

## Current Evidence and Decision Context

The repository currently has a working NIH preparation/validation/smoke path and 33 passing tests. The prepared CSV contains 112,120 images, 30,805 patients, and train/val/test counts of 89,789/11,348/10,983 with no observed patient overlap. The training distribution is long-tailed: NIH `Hernia` has 158 train positives (negative/positive ratio about 567), while `Pneumonia` has 1,169 positives.

The historical ResNet18 reference has validation macro AUROC 0.8204, macro F1 0.1358, and micro F1 0.1870 at a global threshold of 0.5. The existing DenseNet121 checkpoint is stronger on its own validation record (macro AUROC 0.8397, macro F1 0.1698, micro F1 0.2483) but was not a matched experiment and has no completion history. The current engine selects by macro AUROC, exposes no AUPRC or per-label report, and has no independent test gate.

The plan therefore treats threshold selection and class imbalance as the first hypotheses to test. The previously documented tuned validation F1 near 0.3058/0.3638 is motivation only until a reproducible artifact is generated.

## Experiment Ladder

| ID | Priority | Candidate | Main hypothesis | Training cost | Advance condition |
|---|---|---|---|---|---|
| G0 | P0 | Evaluation/provenance gate | The low F1 may be mainly a measurement/threshold/selection problem. | No training; one validation inference pass. | All focused tests pass and the historical validation report reproduces. |
| B0 | P0 | Ordinary BCE control | A clean rerun establishes the current training reference under the new protocol. | One smoke epoch, then up to 2 GPU-hours. | Complete artifact and finite validation history. |
| B1 | P0 | BCE + capped `pos_weight` | Training imbalance suppresses tail-label recall. | Same as B0; two-arm screen cap 4 GPU-hours total. | Validation tuned macro F1 improves without unacceptable AUPRC/label collapse. |
| C0 | P1 | Per-label threshold fitting | A single 0.5 threshold is responsible for a large part of the F1 deficit. | No training; CPU threshold fit after validation inference. | Frozen thresholds improve validation F1 and remain fixed for test. |
| M0 | P1 | DenseNet121 matched control | More capacity/features improve ranking after evaluation is made comparable. | One additional run, up to 2 GPU-hours. | Only compare after B0/B1; do not mix into the loss causal contrast. |
| L1 | P2 | Asymmetric Loss | Easy negatives dominate BCE; asymmetric suppression improves tail AUPRC. | One bounded run, about 1.0-1.2x B0. | Beats B0 on macro AUPRC and does not produce non-finite/degenerate labels. |
| L2 | P2 | Focal or class-balanced BCE | Reweighting hard examples or effective class counts improves rare labels. | One bounded run per loss, about 1.0-1.2x B0. | Test one loss at a time; stop if it only changes calibration without ranking gain. |
| A1 | P2 | Conservative augmentation | Small acquisition/style variation improves invariance without removing pathology. | One run, about 1.1-1.3x B0 per epoch. | Validation gain and no per-label collapse; validation transform unchanged. |
| K1 | P2 | Post-hoc temperature calibration | Probabilities are miscalibrated even when ranking is adequate. | No retraining; one validation fit and one held-out calibration check if available. | ECE/Brier/NLL improve while AUROC/AUPRC and tuned F1 remain within tolerance. |
| D1 | P3 | Multi-source/domain alignment | NIH-only training does not represent acquisition/site shift. | External data staging plus 2-5x B0; governance dependent. | Explicit ontology, patient split, and independent external evaluation available. |
| D2 | P3 | Self-supervised CXR pretraining | Better image representations reduce domain-specific shortcuts. | High cost, typically 5-20x B0; defer until D1 is viable. | External frozen evaluation improves across sites, not just NIH validation. |

## Task 0: Freeze the protocol before implementation

**Files:**
- Track: `docs/superpowers/plans/2026-08-30-one-day-f1-experiment.md`
- Track: `docs/superpowers/plans/2026-08-30-model-robustness-roadmap.md`
- Create: `docs/experiments/2026-08-30-input-inventory.json`

**Required decisions:**

- Add an explicit initial planning commit containing the plan and frozen inventory before any implementation commit. The existing one-day plan currently describes this requirement but its task order does not enforce it.
- Record the labels hash, reference checkpoint hash, image count/size, environment versions, split source, and the limitation that the historical checkpoint lacks the labels-file hash used at its original training time.
- Confirm whether the current patient-level 80/10/10 split is acceptable for the intended benchmark; do not silently call it the official NIH split without the official list files.

**Acceptance:** the plan commit hash is recorded; the inventory parses; no data/checkpoint hash changes; `git status` clearly identifies only intentional plan/inventory files.

**Stop:** if the output ontology or split policy is unresolved, stop all formal training and record the decision as pending.

## Task 1: Build the evaluation and leakage gate (G0/C0)

**Files:**
- Modify: `baseline/metrics.py`
- Modify: `baseline/data.py`
- Create: `baseline/evaluation.py`
- Create: `tune_thresholds.py`
- Create: `evaluate.py`
- Create: `select_candidate.py`
- Create: `run_final_test.py`
- Test: `tests/test_metrics_thresholds.py`, `tests/test_split_loader.py`, `tests/test_evaluation.py`

**Contract:**

- Keep scalar threshold `0.5` backward compatible; add ordered per-label threshold arrays.
- Fit each threshold on validation probabilities only, searching unique probabilities plus `0.0` and `1.0`; resolve F1 ties to the highest threshold.
- Emit macro/micro F1, AUROC, AUPRC, prevalence, and per-label precision/recall/F1/AUROC/AUPRC. A single-class label emits `null` metrics and is excluded from macro means.
- Reject unknown/empty split values, patient leakage, patient-addressable CSVs without persisted split, label-order mismatch, and any request to fit thresholds on `test` or external data.
- Make `run_final_test.py` require a committed selection record and a write-once output directory.

**Acceptance metrics and tests:**

- Synthetic fixtures prove threshold round-trip, label-order validation, test-fit rejection, split isolation, patient disjointness, and degenerate-label handling.
- The historical reference validation report at fixed 0.5 matches the checkpoint values within `1e-4` for the existing metrics.
- The exploratory tuned-F1 value is either reproduced or explicitly marked unreproduced; no test prediction is generated during this task.

**Stop:** any mismatch in the historical report, any leakage test failure, or any attempt to use a test label for fitting blocks all training tasks.

**Cost:** one validation inference pass; normally less than 0.5 GPU-hour, with no parameter updates.

## Task 2: Add controlled losses, early stopping, and provenance (B0/B1)

**Files:**
- Create: `baseline/losses.py`
- Modify: `baseline/engine.py`
- Modify: `train.py`
- Create: `configs/nih_f1_bce.yaml`, `configs/nih_f1_posweight.yaml`
- Test: `tests/test_losses.py`, `tests/test_engine.py`

**Candidate B0: ordinary BCE control**

- **Assumption:** the current BCE objective is a valid control once checkpoint selection and thresholding are fixed.
- **Change:** use the exact current ResNet18/ImageNet, seed 42, split, transforms, AdamW learning rate `0.0003`, weight decay `0.0001`, and batch size 64; change only the output directory and validation selection metric.
- **Acceptance:** complete `history.json`, finite losses/gradients, at least one completed epoch, best checkpoint selected by validation macro AUPRC, and no source/hash mismatch.
- **Stop:** non-finite value, missing validation class, timeout, corrupted artifact, or silent overwrite.
- **Cost:** one smoke epoch plus up to 2 GPU-hours.

**Candidate B1: capped training-set `pos_weight`**

- **Hypothesis:** weighting `negative_count / positive_count` from training rows (cap each weight at 20) increases rare-label recall without destroying ranking.
- **Change:** add only `BCEWithLogitsLoss(pos_weight=...)`; compute counts from `train` rows, fail on zero positives, move weights to the model device, and record raw/capped counts.
- **Acceptance:** tuned validation macro F1 is at least 0.01 absolute above B0, validation macro AUPRC is no more than 0.005 below B0, and no shared label loses more than 0.05 F1 without an explicit degeneracy explanation. These thresholds must be frozen in the selection record before training.
- **Stop:** B1 is ineligible if either arm is incomplete, if weights are non-finite, or if gains occur only after test-driven threshold changes.
- **Cost:** same per-run cap as B0; two-arm screen cap 4 GPU-hours plus smoke checks.

**Common reproducibility changes:**

- Save config/data hashes, resolved labels, environment versions, seed, stop reason, and an atomically written history after every completed epoch.
- Set deterministic worker generators and document any CUDA nondeterminism; do not treat `cudnn.benchmark=True` as reproducibility evidence.
- Never reuse an output directory for a different candidate.

## Task 3: Select the first candidate without test contamination

**Inputs:** B0/B1 best checkpoints, validation predictions, frozen threshold artifacts, and the historical reference report.

**Procedure:**

1. Fit thresholds once on `val` for each eligible checkpoint.
2. Evaluate validation tuned macro F1 and macro AUPRC; rank by tuned macro F1, then macro AUPRC.
3. Write `docs/experiments/nih-f1-selection.json` and commit it before loading any NIH test row.
4. Only then run the single final-test command for the historical reference and selected candidate at fixed 0.5 and their frozen thresholds.

**Acceptance:** test output contains per-label positive counts and all declared metrics; it cannot trigger another training, threshold fit, or candidate choice.

**Stop:** no final test if either required arm is incomplete, the selection record is uncommitted, or the output directory already exists.

**Cost:** validation inference for three checkpoints plus one predeclared test batch; no retraining.

## Task 4: Evaluate later model-strategy candidates (P1/P2)

Run one candidate at a time against the frozen B0 control. Keep validation transforms identical across candidates and use the same split/seed/budget.

**M0: DenseNet121 matched control**

- **Hypothesis:** the stronger existing DenseNet record reflects capacity rather than an accidental run difference.
- **Change:** switch only `model.name` to `densenet121`; keep ImageNet initialization, seed, split, optimizer, selection metric, and threshold protocol fixed.
- **Acceptance:** report macro AUPRC/tuned macro F1 and runtime beside B0; promote only if the gain survives at least two additional seeds or the user explicitly chooses a capacity trade-off.
- **Stop:** do not use the existing unlogged DenseNet checkpoint as causal evidence; stop if memory or runtime exceeds the declared budget.
- **Cost:** one bounded run, approximately 1.5-2x B0 memory/runtime depending on batch size.

**L1: Asymmetric Loss**

- **Hypothesis:** easy negatives dominate BCE in the long tail; asymmetric negative focusing improves macro AUPRC.
- **Change:** add a loss-only candidate with YAML-controlled `gamma_neg=4`, `gamma_pos=1`, `clip=0.05`; no simultaneous sampling or augmentation change.
- **Acceptance:** macro AUPRC improves over B0 and tuned macro F1 does not fall by more than 0.01; gradients remain finite for every label.
- **Stop:** stop after the screen if the gain is only fixed-threshold F1, if any label becomes degenerate, or if the loss cannot be reproduced on the fixture.
- **Cost:** one B0-sized run, about 1.0-1.2x compute.

**L2: Focal or class-balanced BCE**

- **Hypothesis:** focusing hard examples or effective-number weights can improve rare-label ranking, but may trade away calibration.
- **Change:** test focal (`gamma=2`) and class-balanced BCE as separate runs; cap weights at 20 and do not combine them in the first screen.
- **Acceptance:** choose at most one of the two for multi-seed confirmation; require macro AUPRC non-decrease and no unexplained per-label collapse.
- **Stop:** drop a candidate if it only improves one tail label while reducing aggregate macro AUPRC, or if it requires test-specific threshold rescue.
- **Cost:** one bounded run per loss, about 1.0-1.2x B0 each.

**A1: Conservative augmentation**

- **Hypothesis:** mild acquisition variation improves invariance while preserving pathology.
- **Change:** retain resize + horizontal flip and add only small affine perturbation (degrees 7, translate 0.02, scale 0.95-1.05) and brightness/contrast jitter 0.1; validation transform remains deterministic and unchanged.
- **Acceptance:** macro AUPRC and tuned macro F1 improve on B0 on the same split; inspect per-label changes and a fixed image sanity fixture.
- **Stop:** stop if augmentation changes validation preprocessing, harms any clinically important shared label by more than 0.05 F1, or increases runtime beyond 1.3x.
- **Cost:** one run, about 1.1-1.3x per epoch.

## Task 5: Calibration as a separate post-hoc question (K1)

**Hypothesis:** ranking is adequate but probabilities are not reliable for threshold transfer or deployment-like decisions.

**Change:** fit scalar temperature scaling per checkpoint on a calibration subset of `val` only; if no independent calibration split is available, report the result as exploratory and do not reuse the same fitted values for a confirmatory claim. Keep threshold fitting and calibration artifacts separate.

**Acceptance metrics:** expected calibration error, Brier score, and log loss improve; AUROC/AUPRC remain within 0.005 of the uncalibrated model; tuned macro F1 does not fall by more than 0.01. Report calibration curves or binned counts per label where positive counts permit.

**Stop:** no isotonic/vector scaling, test calibration, or threshold re-fitting after seeing test results in this phase. If calibration improves only one label or requires a very small calibration sample, keep it exploratory.

**Cost:** no retraining; one validation inference and CPU fitting (typically minutes).

## Task 6: External robustness ladder (P3)

External work begins only after Task 3 selection is committed and the source contract is approved.

**D0: CheXpert-small readiness preflight**

- **Assumption:** `E:\ChestXRobustData\CheXpert-v1.0-small` may become available under the official terms.
- **Change:** read-only capacity, source header, relative-path, patient-ID, and bounded JPG-readability checks; record `ok` or `blocked`.
- **Acceptance:** machine-readable preflight record with version, row/sample counts, uncertainty counts, and future label contract. Current path is absent, so the expected immediate state is a documented blocker, not a robustness result.
- **Stop:** no download, label conversion, training, or inference if access/space/schema is unresolved.
- **Cost:** no training; minutes to an hour depending on staging and sample size.

**D1: Explicit shared-ontology evaluation**

- Use a declared shared subset only: `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Pneumonia`, `Pneumothorax`, `Pleural_Effusion`.
- Do not map CheXpert `Lung_Opacity` to NIH `Infiltration` or `Lung_Lesion` to NIH `Mass`/`Nodule` without a separately justified contract.
- Preserve patient-disjoint splits and uncertainty policy; never fit thresholds on the external set.

**D2: Domain-generalisation/adaptation candidates**

- **Style/domain randomisation:** use only after an external holdout exists; acceptance is a smaller internal-to-external macro AUPRC/F1 drop, not an NIH-only gain. Cost 1.2-2x B0.
- **CORAL/MMD feature alignment:** requires unlabeled or labeled external images and a frozen protocol; stop if the method improves source validation but worsens external calibration. Cost 1.5-3x B0.
- **GroupDRO/domain-specific normalization:** requires reliable site/group metadata, which NIH currently lacks; defer until source metadata are verified. Cost 1.5-3x B0.
- **Self-supervised CXR pretraining:** high-cost fallback after simpler methods fail; require improvement on at least two independent domains. Cost 5-20x B0.
- **Test-time adaptation:** deprioritise for the first robustness claim because it can leak target-distribution information and complicate reproducibility; require a separately approved protocol if revisited.

**External acceptance:** report image- and study-level macro/micro F1, AUROC, AUPRC, prevalence, per-label positive counts, calibration, and confidence intervals where counts permit. Report the internal-to-external delta and worst-label drop; do not reduce robustness to one aggregate AUROC.

**Stop:** no robustness claim if the external source is unavailable, ontology is ambiguous, patient/study aggregation is missing, or the model/threshold is changed after external labels are observed.

## Formal next-round training gate

Formal training is allowed only when every item below is true:

- [ ] The plan and frozen input inventory have an initial commit hash recorded before implementation changes.
- [ ] Hidden evaluator ontology and the acceptable NIH split policy are documented.
- [ ] Existing 33 tests plus new threshold/split/evaluation tests pass.
- [ ] Historical reference validation metrics reproduce within `1e-4`; any tuned-F1 discrepancy is resolved or explicitly marked unreproduced.
- [ ] Named split loader rejects unknown values, patient leakage, and unsafe fallback behavior.
- [ ] Configs are immutable per candidate, output directories are distinct/write-once, and source/config hashes are captured.
- [ ] One-epoch smoke/acceptance runs for both B0 and B1 pass with finite loss, valid history, and valid checkpoints.
- [ ] The final-test gate is present and no test prediction has been opened before the selection record is committed.

If any box is unchecked, the next action is implementation or diagnosis, not a larger model or a new robustness algorithm.

## Decision rules and budget summary

- Primary model-selection metric: validation tuned macro F1; tie-breaker: validation macro AUPRC.
- Primary ranking metric for imbalance and external robustness: macro AUPRC, accompanied by per-label counts and F1.
- Screen seeds: 42 only; confirm the top two candidates with seeds 42, 43, and 44 before changing the default.
- A candidate that fails the predeclared screen is dropped; do not add unplanned rescue tricks.
- Maximum first-day screen: two required arms (B0/B1), 10 epochs or 2 hours each, plus one smoke epoch per arm.
- Later methods are one-factor ablations, not a cumulative stack. Ensembling, aggressive augmentation, DICOM support, and NIH retirement are out of scope until external acceptance.
- Keep all negative or inconclusive results in the experiment record; a null result is evidence for changing the next hypothesis, not permission to rewrite the gate.

## Expected artifacts before implementation starts

The implementation phase may start only after the repository contains or is ready to create these distinct artifacts:

- `docs/experiments/2026-08-30-input-inventory.json`
- `docs/experiments/nih-f1-reference-validation-fixed.json`
- `docs/experiments/nih-f1-reference-thresholds.json`
- `docs/experiments/nih-f1-bce-thresholds.json`
- `docs/experiments/nih-f1-posweight-thresholds.json`
- `docs/experiments/nih-f1-selection.json`
- `docs/experiments/nih-f1-final-test/`
- `D:\ChestXRobustRuns\nih_f1_bce\` and `D:\ChestXRobustRuns\nih_f1_posweight\` (only when the D drive path has been explicitly created and validated)
- A CheXpert preflight record under `docs/experiments/`, even when its status is `blocked`.

This document is a plan, not a result. No algorithm is promoted until its declared evidence and stop conditions are satisfied.
