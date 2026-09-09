# Experiment Log

## Provenance reconciliation — 2026-09-09

- Read-only verification confirmed complete B4 and B5 frozen evaluation bundles and the B4+B5 ensemble artifact hashes.
- B3 remains pending for standalone test evaluation because no B3 test bundle was found.
- Canonical manifest: `artifacts/canonical_run_manifest.json`.
- `workflow_state.json` and `outputs/experiment_registry.csv` were atomically regenerated from that manifest.
- No training, GPU workload, or test replay was started during reconciliation.

## B2

- Purpose: establish the DenseNet121 14-label baseline with class-imbalance weighting.
- Unique variable: `BCEWithLogitsLoss` with train-only sqrt-capped `pos_weight`.
- Configuration: ImageNet-pretrained DenseNet121, 224px, batch 32, AdamW `3e-4`, AMP, `num_workers=0`, 10 epochs.
- Current state: running; observed through epoch 8 at workflow initialization.
- Next action: wait for epoch 10 and validate the completion gate before post-analysis.
