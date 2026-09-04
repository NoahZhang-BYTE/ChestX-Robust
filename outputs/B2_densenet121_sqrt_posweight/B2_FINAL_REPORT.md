# B2 Final Report

## Protocol Integrity

- B2 training is complete at 10 epochs. The auditable history is epochs 3-10; no values for epochs 1-2 were reconstructed.
- Selected checkpoint: epoch 9 (`best.pt`), because it had the best observed macro AUROC (0.8373033353) and macro AUPRC (0.2681604423).
- Per-label thresholds were fitted once on validation, frozen before test, and SHA256-verified by the test runner. Test did not affect checkpoint selection or threshold fitting.

## Configuration

- Backbone: densenet121; ImageNet pretrained: True; image size: 224; batch: 32.
- Loss: BCEWithLogitsLoss + train-only sqrt-capped pos_weight (cap 20.0); AdamW, lr 0.0003, weight decay 0.0001.
- AMP: True; DataLoader: num_workers=0, prefetch_factor=None, persistent_workers=False.
- B1 used num_workers=4. B2 used num_workers=0 because of Windows multiprocessing/commit instability; this changes loading throughput, not samples, model, loss definition, optimizer, batch size, augmentation, split, or evaluation protocol.

## Training Trajectory

| Epoch | Train loss | Val loss | Macro AUROC | Macro AUPRC | Macro F1 @ 0.5 |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.380420 | 0.400747 | 0.828378 | 0.244755 | 0.275611 |
| 4 | 0.368244 | 0.406395 | 0.821143 | 0.244206 | 0.255797 |
| 5 | 0.364881 | 0.393969 | 0.830748 | 0.246232 | 0.277441 |
| 6 | 0.355906 | 0.398173 | 0.831263 | 0.256718 | 0.274126 |
| 7 | 0.346653 | 0.397680 | 0.831850 | 0.262398 | 0.305333 |
| 8 | 0.336634 | 0.397733 | 0.837183 | 0.263470 | 0.314958 |
| 9 | 0.325912 | 0.397738 | 0.837303 | 0.268160 | 0.311993 |
| 10 | 0.312500 | 0.409465 | 0.830997 | 0.259694 | 0.311372 |

Minimum validation loss was epoch 5 (0.393969). Epoch 10 versus epoch 9: val loss 0.011728; macro AUROC -0.006306; macro AUPRC -0.008466. This is mild overfitting / generalization degradation, so B2 was not extended.

## Validation Threshold Analysis

| Metric | Fixed 0.5 | Validation tuned |
| --- | --- | --- |
| Macro F1 | 0.312339 | 0.331803 |
| Micro F1 | 0.350108 | 0.373286 |
| Sample F1 | 0.150683 | 0.179944 |
| Macro AUROC | 0.837305 | 0.837305 |
| Macro AUPRC | 0.268164 | 0.268164 |

Threshold tuning uses the fixed candidate set 0.01..0.99, independently maximizes binary F1 per class, and resolves equal maxima by choosing the candidate nearest 0.5. AUROC and AUPRC are threshold-independent.

## Frozen-Threshold Test

| Metric | Fixed 0.5 | Validation-frozen tuned |
| --- | --- | --- |
| Macro AUROC | 0.838520 | 0.838520 |
| Micro AUROC | 0.870466 | 0.870466 |
| Macro AUPRC | 0.260723 | 0.260723 |
| Micro AUPRC | 0.300335 | 0.300335 |
| Macro F1 | 0.313892 | 0.323169 |
| Micro F1 | 0.362060 | 0.366610 |
| Sample F1 | 0.150024 | 0.166047 |

## Per-label Validation and Frozen-Test Results

| Label | Val threshold | Val F1 tuned | Test AUROC | Test AUPRC | Test precision | Test recall | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Atelectasis | 0.40 | 0.403590 | 0.811261 | 0.347152 | 0.340360 | 0.502287 | 0.405765 |
| Cardiomegaly | 0.66 | 0.331624 | 0.895873 | 0.237502 | 0.314050 | 0.305221 | 0.309572 |
| Effusion | 0.44 | 0.531088 | 0.889393 | 0.545477 | 0.475091 | 0.631664 | 0.542302 |
| Infiltration | 0.32 | 0.406103 | 0.705687 | 0.332331 | 0.281198 | 0.566480 | 0.375834 |
| Mass | 0.61 | 0.339230 | 0.862229 | 0.355780 | 0.415385 | 0.385714 | 0.400000 |
| Nodule | 0.39 | 0.305256 | 0.775329 | 0.247131 | 0.255341 | 0.387346 | 0.307787 |
| Pneumonia | 0.36 | 0.104938 | 0.739220 | 0.028077 | 0.045000 | 0.168224 | 0.071006 |
| Pneumothorax | 0.43 | 0.413060 | 0.872975 | 0.322254 | 0.356711 | 0.517544 | 0.422334 |
| Consolidation | 0.33 | 0.228484 | 0.790653 | 0.126203 | 0.122727 | 0.426316 | 0.190588 |
| Edema | 0.54 | 0.246334 | 0.901185 | 0.176011 | 0.200969 | 0.397129 | 0.266881 |
| Emphysema | 0.79 | 0.476895 | 0.931858 | 0.396237 | 0.497959 | 0.409396 | 0.449355 |
| Fibrosis | 0.67 | 0.182306 | 0.813240 | 0.093016 | 0.171717 | 0.213836 | 0.190476 |
| Pleural_Thickening | 0.52 | 0.234234 | 0.821727 | 0.171496 | 0.179104 | 0.361446 | 0.239521 |
| Hernia | 0.73 | 0.442105 | 0.928656 | 0.271461 | 0.352941 | 0.352941 | 0.352941 |

## B1 versus B2 Validation Comparison

| Metric | B1 BCE | B2 sqrt-capped pos_weight | B2 - B1 |
| --- | --- | --- | --- |
| macro_auroc | 0.837589 | 0.837305 | -0.000284 |
| macro_auprc | 0.261171 | 0.268164 | 0.006993 |
| macro_f1_fixed05 | 0.140947 | 0.312339 | 0.171392 |
| macro_f1_tuned | 0.326333 | 0.331803 | 0.005470 |
| micro_f1_tuned | 0.371068 | 0.373286 | 0.002218 |
| sample_f1_tuned | 0.172316 | 0.179944 | 0.007628 |

Both rows use identical saved validation targets and the B2 grid threshold protocol. This comparison is validation-only.

## Rare-5 Validation Comparison

| Label | B1 AUPRC | B2 AUPRC | B1 tuned F1 | B2 tuned F1 | B2 - B1 F1 |
| --- | --- | --- | --- | --- | --- |
| Pneumonia | 0.050559 | 0.058887 | 0.113886 | 0.104938 | -0.008948 |
| Edema | 0.212204 | 0.182948 | 0.291188 | 0.246334 | -0.044853 |
| Emphysema | 0.415647 | 0.411188 | 0.482972 | 0.476895 | -0.006077 |
| Fibrosis | 0.080699 | 0.105471 | 0.161512 | 0.182306 | 0.020794 |
| Hernia | 0.182729 | 0.328428 | 0.257426 | 0.442105 | 0.184680 |

## Hard-label Error Analysis

| Label | TP | FP | FN | TN |
| --- | --- | --- | --- | --- |
| Infiltration | 1014 | 2592 | 776 | 6601 |
| Pneumonia | 18 | 382 | 89 | 10494 |
| Hernia | 6 | 11 | 11 | 10955 |

Train-split top conditional co-occurrences:
- Infiltration: Effusion=0.201690, Atelectasis=0.161364, Nodule=0.076985
- Pneumonia: Infiltration=0.432849, Edema=0.242087, Effusion=0.185629
- Hernia: Infiltration=0.139241, Atelectasis=0.120253, Mass=0.120253

## Conclusions and B3 Hypothesis

- B2 improves validation macro AUPRC and fixed-threshold F1 relative to BCE; all B1/B2 model comparisons are validation-only.
- Pneumonia remains the difficult rare class: its frozen-test AUPRC and F1 are the lowest, despite a nontrivial AUROC. This is a precision-recall limitation rather than only a threshold issue.
- Infiltration has high support but relatively low discrimination, so it is not explained by class rarity alone. Hernia provides a counterexample: it is rare but retains high rank discrimination.
- B3 is a pre-specified ASL experiment (gamma_neg=4, gamma_pos=1, clip=0.05) under otherwise B2-matched settings. Its primary validation metric is macro AUPRC, then macro AUROC, then validation-tuned macro F1. B2 test results must not select B3 or its checkpoints.

## Artifact Index

- `checkpoint_inventory.csv`, `post_analysis/checkpoint_selection.json`, `post_analysis/training_curves.png`
- `thresholds/y_true_val.npy`, `thresholds/y_prob_val.npy`, `thresholds/threshold_tuning_val.csv`, `thresholds/thresholds_frozen_for_test.json`, `thresholds/threshold_freeze.json`
- `test/test_fixed05_metrics.json`, `test/test_tuned_metrics.json`, `test/test_per_label_comparison.csv`
- `post_analysis/hard_labels/` and `post_analysis/label_cooccurrence.*`
