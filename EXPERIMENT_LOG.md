# Experiment Log

## B2

- Purpose: establish the DenseNet121 14-label baseline with class-imbalance weighting.
- Unique variable: `BCEWithLogitsLoss` with train-only sqrt-capped `pos_weight`.
- Configuration: ImageNet-pretrained DenseNet121, 224px, batch 32, AdamW `3e-4`, AMP, `num_workers=0`, 10 epochs.
- Current state: running; observed through epoch 8 at workflow initialization.
- Next action: wait for epoch 10 and validate the completion gate before post-analysis.
