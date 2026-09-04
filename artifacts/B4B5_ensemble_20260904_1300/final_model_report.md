# Final Model Report

## Final Candidate

- Model: DenseNet121 probability-level ensemble
- Composition: `0.4 x B4 Weighted BCE + 0.6 x B5 ASL`
- B4 source: `artifacts/B4_densenet121_sqrt_posweight_scheduler_eval_20260904_025154`
- B5 source: `artifacts/B5_densenet121_asl_eval_20260904_1200`
- Weight selection: validation macro AUROC maximization
- Threshold fitting: validation only, per-class F1 maximization
- Test used for selection or threshold fitting: no
- Split counts: train 89,789; validation 11,348; test 10,983

## Overall Metrics

| Split | Macro AUROC | Micro AUROC | Macro AUPRC | Micro AUPRC | Macro F1 (0.5) | Micro F1 (0.5) | Sample F1 (0.5) | Macro F1 (tuned) | Micro F1 (tuned) | Sample F1 (tuned) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 0.848453 | 0.878903 | 0.285659 | 0.338083 | 0.330856 | 0.386382 | 0.172166 | 0.353755 | 0.381057 | 0.171046 |
| Test | 0.846701 | 0.884084 | 0.281650 | 0.341410 | 0.336586 | 0.395567 | 0.166647 | 0.340710 | 0.371963 | 0.159261 |

## Per-Class Test Metrics

| Class | Support | Threshold | AUROC | AUPRC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atelectasis | 1093 | 0.536621 | 0.823764 | 0.366948 | 0.398768 | 0.384224 | 0.414456 |
| Cardiomegaly | 249 | 0.620020 | 0.906918 | 0.273795 | 0.340708 | 0.379310 | 0.309237 |
| Effusion | 1238 | 0.474121 | 0.891555 | 0.540427 | 0.550642 | 0.492976 | 0.623586 |
| Infiltration | 1790 | 0.487402 | 0.724881 | 0.349637 | 0.391346 | 0.343460 | 0.454749 |
| Mass | 630 | 0.526514 | 0.876160 | 0.374206 | 0.440367 | 0.424779 | 0.457143 |
| Nodule | 648 | 0.444092 | 0.787368 | 0.258097 | 0.326142 | 0.276940 | 0.396605 |
| Pneumonia | 107 | 0.375293 | 0.765919 | 0.036287 | 0.072316 | 0.041131 | 0.299065 |
| Pneumothorax | 570 | 0.473975 | 0.879393 | 0.347747 | 0.443145 | 0.351909 | 0.598246 |
| Consolidation | 380 | 0.393701 | 0.801296 | 0.136535 | 0.196217 | 0.126524 | 0.436842 |
| Edema | 209 | 0.482227 | 0.909040 | 0.211860 | 0.300380 | 0.249211 | 0.377990 |
| Emphysema | 298 | 0.638770 | 0.941894 | 0.408066 | 0.476651 | 0.458204 | 0.496644 |
| Fibrosis | 159 | 0.448242 | 0.835991 | 0.111202 | 0.162988 | 0.111628 | 0.301887 |
| Pleural_Thickening | 332 | 0.507617 | 0.823429 | 0.172591 | 0.256471 | 0.210425 | 0.328313 |
| Hernia | 17 | 0.572119 | 0.886202 | 0.355697 | 0.413793 | 0.500000 | 0.352941 |

## Error Analysis

### Lowest-Performance Classes

1. **Pneumonia**: AUROC `0.7659`, AUPRC `0.0363`, F1 `0.0723`. The ranking signal is above random, but the very low support (107/10,983) and precision `0.0411` dominate the error. This is primarily a rare-class precision/imbalance limitation, not evidence of an absent ranking signal.
2. **Fibrosis**: AUROC `0.8360`, AUPRC `0.1112`, F1 `0.1630`. AUROC is materially higher than F1, while precision is `0.1116`; thresholded classification remains weak despite usable discrimination.
3. **Consolidation**: AUROC `0.8013`, AUPRC `0.1365`, F1 `0.1962`. Recall reaches `0.4368`, but precision is only `0.1265`, indicating substantial false-positive burden.
4. **Pleural_Thickening**: AUROC `0.8234`, AUPRC `0.1726`, F1 `0.2565`; both precision and recall are moderate-to-low.
5. **Edema**: AUROC `0.9090` but F1 `0.3004`, showing a threshold/precision-recall mismatch rather than poor discrimination.

The lowest AUROC is Infiltration (`0.7249`), followed by Pneumonia (`0.7659`) and Nodule (`0.7874`). The largest gap between discrimination and thresholded F1 occurs in rare classes, especially Pneumonia and Fibrosis. The ensemble does not remove the long-tail error pattern.

## Paper-Ready Description

We developed a probability-level dual-loss ensemble based on two DenseNet121 branches. The first branch used weighted binary cross-entropy to emphasize difficult and under-represented labels, while the second used Asymmetric Loss to suppress gradients from abundant easy negatives. Ensemble weights were selected exclusively on the validation set by maximizing macro AUROC, yielding a 0.4/0.6 weighting of the weighted-BCE and ASL branches. Per-class decision thresholds were then fitted exclusively on validation predictions and frozen before one independent test evaluation. The ensemble achieved test macro AUROC 0.8467, micro AUROC 0.8841, macro AUPRC 0.2816, micro AUPRC 0.3414, and tuned macro F1 0.3407. The complementary losses improved the balance between macro-level discrimination and micro-level multilabel performance, although rare classes such as Pneumonia remained the main limitation.

## Final Status

- Final candidate: **DenseNet121 B4+B5 ensemble (0.4/0.6)**
- B6 or any new backbone: not trained
- Test evaluation: completed once using frozen validation-derived thresholds
- Existing B1, B4, and B5 checkpoints: unchanged
- Regression tests: `101 passed`

