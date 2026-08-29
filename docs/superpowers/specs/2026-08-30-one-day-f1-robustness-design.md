# One-Day F1 Experiment and External Evaluation Readiness Design

## Status and Decision

This is a one-day exploration/understanding experiment, not a dataset-replacement
project. Its objective is to establish a defensible F1-focused training and
evaluation loop while preparing one external dataset path without contaminating
model selection.

The current reference is `checkpoints/imagenet_pretrained/best.pt`. It is not a
model trained only on ImageNet: it is an ImageNet-initialized ResNet18 fine-tuned
on the local NIH ChestX-ray14 train split. Its saved validation result at epoch 5
is macro AUROC 0.8204, micro AUROC 0.8680, macro F1 0.1358, micro F1 0.1870,
and sample F1 0.0583 at a global probability threshold of 0.5. Training loss
continued falling after that point while validation loss and AUROC worsened, so
the current loop also needs a validation-controlled stop condition.

The earlier full external-dataset migration plan remains a future reference only.
It is deliberately not the first-day scope because the competition output
ontology is unknown and CheXpert observations are not a drop-in replacement for
the NIH 14 labels. No current data, metadata, or checkpoint will be deleted,
moved, or overwritten by this work.

## Goal and Success Criteria

**Goal:** Within one day, produce one validation-selected NIH model with a
leakage-safe final F1 report, plus verified readiness to begin a later
cross-domain CheXpert experiment. This day cannot itself demonstrate improved
cross-domain robustness because it does not run an external model evaluation.

The day succeeds only when all of the following are true:

1. Before any real NIH test prediction or result is generated or opened, the
   historical reference checkpoint and the two predeclared new candidates have
   validation-only reports. The historical checkpoint is a reference selected
   previously by fixed-threshold macro AUROC, not an F1-selected checkpoint.
2. A same-backbone, same-split `BCE` versus `BCE + pos_weight` comparison runs
   with an explicit configuration, deterministic seed, and separate output
   directory. For each run, its best checkpoint is chosen by validation macro
   AUPRC; its validation-only threshold artifact is then fit exactly once. The
   final candidate is chosen by tuned validation macro F1, using macro AUPRC as
   the predeclared tie-breaker.
3. Only after that selection is written to the experiment record, a final batch
   evaluation reports both the historical reference and the selected candidate
   at fixed 0.5 and at their own frozen validation thresholds. It includes
   macro/micro F1, macro/micro AUROC, macro/micro AUPRC, prevalence, and
   per-label precision, recall, F1, AUROC, and AUPRC. Test values are reported
   as exploratory point estimates with per-label positive counts; they never
   trigger further training, threshold tuning, or candidate selection.
4. A CheXpert-small source location on `E:` has an explicit label contract,
   uncertainty policy, patient-split requirement, and readable-file/schema
   preflight path. If access, download, or preflight is not completed within the
   day, that outcome is recorded as an external-data blocker rather than claimed
   as a robustness result.
5. NIH data remain at `data/labels.csv` and `data/raw/images` in
   `C:\胸片AI多标签高鲁棒`, and existing checkpoints remain intact.

An increase in validation F1 alone is not a successful claim. A prior exploratory
threshold scan suggesting validation macro F1 near 0.3058 is useful motivation,
but it is not independently reproduced yet and is not a test result. A negative
or inconclusive final comparison is still a successful experiment if the protocol
is followed; it is evidence to use when selecting the next intervention.

## Alternatives Considered

### A. Tune only the existing checkpoint

This is the fastest route to a higher F1 score because the model is already
trained. It does not test whether training can improve recall for tail labels or
whether the result generalizes beyond NIH. It is retained as the mandatory
control, not the complete experiment.

### B. Immediately replace NIH with CheXpert-14

This would create a larger, more external-facing dataset pipeline, but it would
mix a new label ontology, uncertain-report labels, access constraints, and model
changes into a one-day result. CheXpert's `Lung_Opacity` is not NIH
`Infiltration`, and `Lung_Lesion` is not NIH `Mass` or `Nodule`. This option is
deferred until the competition output contract is known.

### C. Recommended: controlled NIH F1 loop plus external-data readiness

Keep NIH intact for a clean F1 comparison. Add threshold-safe evaluation and one
class-imbalance intervention (`pos_weight`), then stage CheXpert-small without
training on it unless its preflight finishes early. This produces a measurable
result in one day while preserving a valid route to cross-domain evaluation.

## Experimental Architecture

### 1. Split-safe prediction and evaluation

The data layer will expose a read-only loader for a named persisted split
(`train`, `val`, or `test`) while preserving the existing `build_dataloaders`
training API. Evaluation will load the checkpoint's resolved ordered label list
and reject any mismatch with the evaluation configuration or threshold artifact.

Prediction collection will return labels and sigmoid probabilities separately
from metric calculation. This allows a validation command to fit thresholds and
an evaluation command to consume, but never refit, those thresholds. A test or
external invocation must fail if asked to tune thresholds.

The threshold artifact is JSON with this shape:

```json
{
  "schema_version": 1,
  "fit_split": "val",
  "objective": "binary_f1",
  "label_cols": ["Atelectasis", "Cardiomegaly"],
  "thresholds": [0.21, 0.38]
}
```

For each label, the fitting procedure searches the unique validation
probabilities plus `0.0` and `1.0`, applies predictions with `probability >=
threshold`, chooses the threshold with highest binary F1, and resolves ties to
the highest threshold. A label that lacks either positives or negatives in the
fit split gets sentinel threshold `1.0`, an `available: false` flag, and a
reason of `no_positive` or `no_negative`; it is excluded from tuned macro F1.
Scalar threshold `0.5` remains supported for backward compatibility.

The primary report will include per-label positive and negative counts,
precision, recall (sensitivity), F1, AUROC, and average precision/AUPRC, plus
macro/micro F1, AUROC, and AUPRC. For a label with one observed target class,
all its rate/score metrics are emitted as JSON `null` with a degeneracy flag and
are excluded from macro means; micro AUROC/AUPRC is `null` only if all flattened
targets have one class. Macro values are unweighted means over non-degenerate
labels, and micro values pool all label decisions. This prevents an AUROC-only
view from masking poor tail-label recall, consistent with the long-tail chest
X-ray literature summarized by the CXR-LT challenge overview (Medical Image
Analysis 2024, DOI `10.1016/j.media.2024.103224`).

### 2. One controlled imbalance comparison

The first new training intervention is `BCEWithLogitsLoss(pos_weight=...)`. For
each label, the weight is computed only from the persisted NIH training rows as
`negative_count / positive_count`, capped at 20. A label with zero training
positives is a configuration error rather than a silent infinite weight. The
control run uses ordinary BCE. Both runs use the existing ImageNet-initialized
ResNet18, current NIH split, seed 42, image size, transform family, AdamW with
the existing fixed learning rate `0.0003` and weight decay `0.0001`, so loss
weighting is the only material difference. `densenet121.yaml` is not part of
this first-day comparison.

`pos_weight` is chosen ahead of Asymmetric Loss because it is lower-risk to add
and produces an interpretable first ablation. Asymmetric Loss is a named
next-round candidate, not a required first-day deliverable; its motivation is
the suppression of easy negatives in multi-label imbalance (Ridnik et al., ICCV
2021, DOI `10.1109/ICCV48922.2021.00015`). Sampling, focal loss, multi-stage
training, aggressive augmentation, and ensembles are explicitly out of scope
for this day so their effects are not confounded.

Each run writes an independent config-resolved checkpoint directory under
`D:\ChestXRobustRuns`, never to the removable drive. Training records a
machine-readable history. Each run must first complete a one-epoch smoke run,
then has a hard budget of at most 10 epochs or two hours of wall-clock time
(checked between completed epochs), whichever occurs first. Early stopping uses
validation macro AUPRC with `patience: 2` and `min_delta: 0.0005`; this metric is
threshold-independent and more informative than AUROC for severe class
imbalance. For each completed run, the macro-AUPRC-selected `best.pt` is fixed
before validation-only threshold tuning. Tuned validation macro F1 ranks only
those two final checkpoints, with macro AUPRC as the tie-breaker; it must never
be used to reselect an epoch. NIH test values are neither generated nor viewed
before this decision is recorded.

### 3. External-data readiness, not external model selection

CheXpert-v1.0-small is the first external source because it is large enough for
future pretraining/fine-tuning and can be staged on the U drive:
`E:\ChestXRobustData\CheXpert-v1.0-small`. It requires acceptance of Stanford's
data-use terms. The day-one external deliverable is strictly an access and
preflight record, not conversion, training, or inference. Images and any archive
staging remain on `E:`; checkpoints, generated metadata, and reports remain on
`D:` or in the repository. Before downloading or extracting, preflight must
recheck that `E:` is mounted and has enough current space for the archive, the
published extracted size, and 10 GiB of headroom. The observed free space of
about 62.56 GiB is only a dated indication, not approval to extract. Once data
are present, preflight checks CSV headers, source paths, and a small readable JPG
sample; it also records whether worker loading from the removable drive is
stable.

The first cross-domain contract contains only the seven defensibly shared
concepts below, in this exact order:

```text
Atelectasis
Cardiomegaly
Consolidation
Edema
Pneumonia
Pneumothorax
Pleural_Effusion
```

NIH `Effusion` maps to `Pleural_Effusion` only in this explicitly documented
future cross-domain subset. No mapping is allowed from `Lung_Opacity` to
`Infiltration` or from `Lung_Lesion` to `Mass`/`Nodule`. This is a contract for a
future independently trained seven-output model (or separately designed output
projection), not permission to evaluate the existing fourteen-output NIH
checkpoint on CheXpert. No CheXpert `-1` or missing-label conversion runs on day
one. The future first-pass `zero` policy must count `-1` and blank values
separately and treat both as an exploratory training assumption, not a ground
truth claim. Patient identifiers must drive a persisted patient-disjoint
train/validation/test split before a CheXpert run can be accepted.

BRAX is the preferred later external evaluation set because it has a different
country/hospital domain and CheXpert-style observations, but its PhysioNet
credentialed access requirements make it a post-day-one task. VinDr-CXR is a
fallback external source with a strict subset mapping only; native DICOM support
and its `Nodule/Mass` ontology must not be silently coerced. This separation is
important because internal chest-X-ray performance is known not to guarantee
cross-hospital performance (Zech et al., PLOS Medicine 2018, DOI
`10.1371/journal.pmed.1002683`).

## Day Plan and Gates

| Timebox | Deliverable | Stop or pivot condition |
| --- | --- | --- |
| Hours 0-2 | Tests and implementation for split loading, prediction export, metrics, and frozen threshold JSON | Do not train until a tiny fixture proves a test loader cannot fit thresholds. |
| Hours 2-3 | Reproduce the historical checkpoint's validation threshold scan and write its validation artifact | If it materially differs from the prior exploratory scan, investigate label/config/checkpoint mismatch before comparing losses. Do not run NIH test. |
| Hours 3-4 | `pos_weight`, early stopping, experiment configs, and focused tests | If tests or a one-epoch smoke run fail, fix the training loop before spending GPU time. |
| Hours 4-8 | Matched BCE and `pos_weight` screens, each within the stated epoch/wall-time budget | If a run cannot finish, retain its incomplete status and do not infer a ranking. |
| Hours 8-9 | Freeze each eligible best checkpoint, fit its validation thresholds, and write the validation-only selection record | Do not invoke an NIH test loader until the selected candidate is recorded. |
| Hours 9-10 | One predeclared final batch: historical reference plus selected candidate, each at fixed 0.5 and frozen thresholds | Do not train, retune, or revise candidate selection after output is opened. |
| Hours 10-12 / parallel waiting time | CheXpert access and U-drive preflight | Lack of access or a failed preflight is a documented blocker, not an NIH F1 or robustness failure. |

## File and Interface Boundaries

The implementation should stay narrowly partitioned:

- `baseline/metrics.py`: pure NumPy/scikit-learn threshold fitting, threshold
  application, aggregate and per-label metrics.
- `baseline/data.py`: persisted named split loader and training-label count
  helper; no NIH-specific ontology changes.
- `baseline/losses.py`: loss factory containing only `bce` and
  `bce_pos_weight` in the first-day implementation.
- `baseline/engine.py`: reusable prediction collection, configured loss,
  validation AUPRC early stopping, and checkpoint selection metadata.
- `evaluate.py` and `tune_thresholds.py`: separate CLIs that enforce split
  roles rather than hiding evaluation behavior inside `train.py`.
- `preflight_chexpert.py`: a narrow, read-only external readiness CLI that
  checks staging capacity, expected source layout/CSV headers, and a bounded JPG
  sample when data are present. It does not convert labels or train a model.
- `configs/nih_f1_bce.yaml`, `configs/nih_f1_posweight.yaml`, and a
  CheXpert preflight config: immutable, separate experiment definitions.
- Focused test files for thresholds, evaluation, losses, and the CheXpert
  preflight parser; existing NIH regression tests must continue to pass
  unchanged.

## Non-Goals and Safety Boundaries

- Do not delete, move, archive, or replace `data/labels.csv`,
  `data/raw/images`, or any checkpoint in this one-day cycle.
- Do not alter `baseline/labels.py` to force a CheXpert ontology into the NIH
  constant.
- Do not train on external data, tune thresholds on test/external data, or use
  external metrics to choose the NIH loss/model.
- Do not claim cross-domain robustness without a completed, frozen-model
  external evaluation. Schema preflight is readiness evidence only.
- Do not convert CheXpert labels, train a CheXpert model, or evaluate any model
  externally in this first-day cycle. Those actions require a separate approved
  external-data implementation plan.
- Do not add DICOM handling, view/study aggregation, bootstrap confidence
  intervals, self-supervised pretraining, a new backbone, or ensemble logic in
  this iteration.

## Verification Strategy

Unit tests will use tiny PNG/JPG fixtures and synthetic predictions to prove:

1. scalar thresholds preserve current behavior while per-label threshold arrays
   apply in label order;
2. fitting uses validation data only and rejects `test`/external fitting;
3. threshold JSON round-trips label order and rejects mismatches;
4. per-label metrics use the declared `null`/macro-exclusion behavior for labels
   with one observed class;
5. named split loaders select only their requested split and preserve existing
   train/validation behavior;
6. `pos_weight` is finite, derives from train rows only, and leaves ordinary
   BCE as the default when no loss block is supplied;
7. early stopping and best-checkpoint selection are driven by the configured
   validation metric, not test metrics;
8. CheXpert preflight rejects a missing drive, insufficient computed capacity,
   incorrect source headers, unsafe source paths, and unreadable sampled JPGs.

Before reporting completion, the repository test suite, focused smoke tests,
and the exact evaluation commands used for each result must all pass. The result
record will distinguish measured test results, validation-selection values, and
unavailable external-data milestones.

## Evidence and Assumptions

- CXR-LT's challenge overview reports that long-tail chest-X-ray methods benefit
  from imbalance-aware methods and evaluates with mAP because AUROC can be
  overly optimistic under imbalance: DOI `10.1016/j.media.2024.103224`.
- Asymmetric Loss is retained as a later controlled alternative, not mixed into
  the first comparison: DOI `10.1109/ICCV48922.2021.00015`.
- Self-supervised chest-X-ray pretraining remains a later robustness candidate,
  not a first-day dependency: Tiu et al., Nature Biomedical Engineering 2022,
  DOI `10.1038/s41551-022-00936-9`.
- CheXpert acquisition and use must follow the official Stanford dataset terms:
  https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- BRAX, MIMIC-CXR-JPG, and VinDr-CXR have credential/access and preparation
  constraints; their availability must be rechecked when the external-evaluation
  phase begins.
