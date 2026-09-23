# 2026 NCCCU AI Challenge Contract

Source: http://www.ncccu.org.cn/index/Paper/case1.html
Repository snapshot date: 2026-09-22

This document records the competition contract separately from the historical
NIH ChestX-ray14 experiments in this repository. The competition data must not
be committed to Git or redistributed.

## Task

- Multi-label chest X-ray classification with 10 target classes.
- A study may contain one or more chest X-ray images.
- A label is positive only when it is positively mentioned in the report and
  is present in the corresponding image.
- Primary ranking metric: macro-averaged AUC.
- Tie-break metric: macro-F1.
- Submission limit: two per day.

## Published splits

| Split | Studies | Labels |
| --- | ---: | --- |
| Train | 18,513 | Available |
| Validation | 859 | Available |
| Test | 1,414 | Hidden |

Images are JPG or PNG and do not have a fixed resolution.

## Published fields

| Field | Meaning |
| --- | --- |
| `Subject_id` | Unique patient identifier |
| `Study_id` | Unique study identifier; one study may contain multiple images |
| `Predict_class` | Disease class ID |
| `ViewPosition` | X-ray projection/view |
| `PatientOrientation` | Patient orientation during acquisition |
| `Rows` | Image height |
| `Columns` | Image width |
| `Support Devices` | `1` present, `0` absent, `-1` uncertain, missing if not mentioned |

Treat `Subject_id` as the grouping key whenever a split has to be created. Do
not allow one subject to cross train, validation, or any local holdout split.
The official validation split must remain intact for model selection.

## Submission contract

- Filename: `submission.csv`.
- Encoding: UTF-8.
- Published columns: `Study_id`, `Predict_class`, `Probability`.
- `Probability` must be in the closed interval `[0, 1]`.
- The published text describes one row per predicted class and uses `0.5` as
  the default inclusion threshold.

The website simultaneously says that every disease probability is required and
that only classes over a threshold should be emitted. Omitting low-probability
study/class pairs is not obviously compatible with macro-AUC. The class-ID map
and CSV example are also embedded as images rather than machine-readable text.
Therefore, before generating a submission, use the downloaded official
`sample_submission.csv` as the authoritative contract and confirm all of the
following:

1. The exact 10 class IDs and their output order.
2. Whether all `1,414 x 10 = 14,140` study/class rows are required or rows below
   the decision threshold must be omitted.
3. Whether probabilities are evaluated at study level and, if so, the required
   aggregation for studies containing multiple images.
4. Whether a study with no class above the F1 threshold needs a fallback row.
5. Whether the column spelling and capitalization shown above are exact.

Do not tune the probability ranking for macro-AUC with a hard threshold.
Thresholds only affect macro-F1 and must be fitted on the labelled validation
set. Preserve raw sigmoid probabilities for AUC submission unless the sample
file explicitly establishes a different contract.

## Runtime contract

| Resource | Published limit |
| --- | --- |
| GPU | NVIDIA T4 16 GB x1 |
| CPU | 8 cores |
| Memory | 32 GB |
| Disk | At least 100 GB SSD |
| Driver | At least 570.117 |
| CUDA / cuDNN | CUDA 12.8 / cuDNN 9 |
| Python | 3.12 |
| Framework | PyTorch 2.10.0 / torchvision 0.25.0 |
| Uncompressed model files | At most 500 MB total |
| End-to-end latency | At most 100 ms per image |

Benchmark latency on a T4 with image decoding, preprocessing, model execution,
multi-image study aggregation, and output handling included. Report warm-up,
batch size, precision, and percentile latency rather than relying on a desktop
GPU timing.

## Data and pretrained-weight policy

The rules permit only the organizer-provided training dataset and prohibit
external labelled data. The current repository's B0-B5 checkpoints were
trained on NIH ChestX-ray14. They are historical research artifacts and must
not be submitted, ensembled, distilled, or used to generate pseudo-labels for
this competition.

The public page does not explicitly settle whether generic ImageNet pretrained
weights count as prohibited external data. Obtain written organizer
confirmation before using `pretrained: true`; otherwise train with
`pretrained: false`. Data augmentation is explicitly allowed.

## B4/B5 migration strategy

The B4/B5 work remains the methodological starting point, but not a source of
competition model parameters. The following parts can be inherited in full:

- the complementary-loss hypothesis: square-root positive-weight BCE for B4
  and Asymmetric Loss for B5;
- the two-branch training, validation, checkpoint-selection, and probability
  ensemble workflow;
- subject-isolated splits and explicit study-level evaluation;
- validation-only model selection, ensemble-weight selection, threshold
  fitting, and calibration;
- frozen protocols, provenance records, and one-way final-test discipline;
- conclusions from the long-tail negative experiments, so previously rejected
  approaches are not repeated without a new official-data hypothesis.

The following NIH-derived artifacts must not be inherited:

- 14-class model weights or checkpoints;
- optimizer or scheduler state tied to those checkpoints;
- per-class thresholds, calibration parameters, or label prevalence priors;
- global or per-class B4/B5 ensemble weights;
- the NIH label order and any NIH test-set model-selection conclusion.

Both branches must be initialized under the confirmed competition pretrained-
weight policy and retrained on the organizer-provided 10-class data. Historical
hyperparameters are starting hypotheses only; official validation evidence must
re-establish checkpoint epochs, loss settings, thresholds, and ensemble weights.

### Formal competition baseline

The first formal baseline should reproduce the B4/B5 experiment structure on
the official data before adding new model families:

1. Build a 10-output B4 branch with square-root positive-weight BCE.
2. Build a 10-output B5 branch with Asymmetric Loss.
3. Select each branch's checkpoint using the official validation set and a
   metric order aligned with the competition: macro-AUC first, macro-F1 second.
4. Persist image-level logits and probabilities together with `Subject_id`,
   `Study_id`, view metadata, checkpoint hash, label order, and preprocessing.
5. Compare fixed, max, mean, noisy-or, and learned validation-only aggregation
   at study level. Freeze one aggregation rule before test inference.
6. Measure B4/B5 error and ranking complementarity at study level before fitting
   any ensemble. Compare each branch with a global-weight probability ensemble;
   use per-class weights only if the validation gain is stable under resampling.
7. Fit F1 thresholds on validation predictions only. Preserve unthresholded
   study-level probabilities for macro-AUC and use thresholds only where the
   confirmed submission contract requires decisions.

An ensemble is not justified merely because the losses differ. Promotion
requires a reproducible official-validation gain over the stronger single
branch, supported by per-class deltas and subject-level bootstrap stability.
When complementarity is weak, submit the stronger single branch instead of
adding ensemble complexity and latency.

## Repository migration gate

The training core already supports an explicit ordered `data.label_cols` list,
but the NIH-specific final evaluation and B4/B5 ensemble scripts assume the
canonical 14 NIH labels. A competition run is ready only after:

- official files are stored locally under ignored `data/` paths;
- the official 10-class ID map is transcribed and independently checked;
- training metadata has one row per image with a persisted official split;
- studies and subjects do not leak across splits;
- an explicit ordered 10-label list is set in the competition config;
- image-level predictions are aggregated to exactly one probability per
  study/class according to the confirmed rule;
- B4 and B5 are retrained independently on the official 10-class data;
- all thresholds, calibration values, and ensemble weights are re-fitted using
  official validation data only;
- study-level complementarity demonstrates that the ensemble improves on the
  stronger official-data branch;
- the output is validated against the official sample submission;
- the packaged model passes both the 500 MB and T4 latency gates.

The website's references to retinal images and glaucoma in the submission
section are inconsistent with the chest X-ray task and should be treated as
template residue, not as a model target definition.
