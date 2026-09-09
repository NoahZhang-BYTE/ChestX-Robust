# Frozen ensemble final analysis

Test set contains 10,983 images. The frozen candidate is evaluated once with validation-only per-class thresholds; no test value is used for selection or fitting.

## Ranking metrics

- Macro AUROC: **0.8467**; micro AUROC: **0.8841**
- Macro AUPRC: **0.2816**; micro AUPRC: **0.3414**

## Thresholded metrics

- Macro F1: **0.3408** at frozen validation thresholds
- Lowest AUROC: **Infiltration**; lowest F1: **Pneumonia**

The plots are descriptive: AUROC/AUPRC assess ranking, while confusion counts and F1 depend on the frozen decision thresholds. Rare-label precision remains the main limitation.
