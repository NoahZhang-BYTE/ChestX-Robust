# Historical B4 Recovery Status (superseded by 2026-09-09 reconciliation)

This document preserves the recovery and BSOD evidence from the original run.
It is a historical artifact, not the current stage ledger. See
`artifacts/canonical_run_manifest.json` for the reconciled status.

**Generated:** 2026-09-02 (Asia/Singapore)  
**Experiment:** DenseNet121 + BCEWithLogitsLoss with train-only sqrt-capped `pos_weight`  
**Artifact directory:** `outputs/B4_densenet121_sqrt_posweight_scheduler/`

## Status

**B4_NOT_FAILED**  
**B4_INTERRUPTED_BY_SYSTEM_BSOD**  
**SAFE_RESUME_FROM_EPOCH=9**

The first resume attempt after epoch 6 was interrupted by Windows BugCheck
`0x50` (`PAGE_FAULT_IN_NONPAGED_AREA`). After MemTest86 passed, the explicitly
authorized resume completed epochs 7 and 8, then a second BugCheck `0x50`
interrupted epoch 9 before its checkpoint boundary. These are system-level
interruptions, not model or training-algorithm failures. No automatic resume or
new training may start until `HARDWARE_STABILITY_BLOCK` is explicitly lifted by
the user.

## Read-only checkpoint verification

Source: `outputs/B4_densenet121_sqrt_posweight_scheduler/last.pt`

| Field | Verified value |
|---|---|
| `torch.load` | PASS (`map_location='cpu'`) |
| completed epoch | 8 |
| next epoch | 9 |
| model state | present |
| optimizer state | present |
| AMP scaler state | present |
| scheduler state | present |
| checkpoint history | 8 entries, epochs 1–8 |

`best_macro_auroc.pt` and `best_macro_auprc.pt` also load successfully; both
remain epoch-7 checkpoints (the best metrics were not exceeded at epoch 8).
No checkpoint was modified after the second interruption.

## History verification

`outputs/B4_densenet121_sqrt_posweight_scheduler/history.csv` is present with
8 rows for contiguous epochs 1–8. All numeric fields are finite, including
train loss, validation loss, AUROC/AUPRC, F1 metrics, and per-label records.
The final durable row is epoch 8 (`val_loss=0.4015612572610147`,
`macro_auroc=0.8308607455926422`, `macro_auprc=0.2660993309384287`).

## Second interruption evidence

- Resume process reached and durably saved epoch 8; epoch 9 produced no row or
  checkpoint.
- Windows rebooted at `2026-09-03 19:02:52` after `Kernel-Power 41`.
- WER recorded BugCheck `0x50` at `2026-09-03 19:03:22` and wrote
  `C:\Windows\Minidump\090326-30125-01.dmp`.
- `resume_20260903.err.log` is empty; no Python traceback was emitted.
- After reboot there are no B4 Python training processes. Test was not run.

## Preserved artifacts

- `outputs/B4_densenet121_sqrt_posweight_scheduler/last.pt`
- `outputs/B4_densenet121_sqrt_posweight_scheduler/history.csv`
- `outputs/B4_densenet121_sqrt_posweight_scheduler/best_macro_auroc.pt`
- `outputs/B4_densenet121_sqrt_posweight_scheduler/best_macro_auprc.pt`
- `outputs/B4_densenet121_sqrt_posweight_scheduler/resume_bsod_20260902.json`
- `WORKFLOW_BLOCKED.md`
- `outputs/B4_densenet121_sqrt_posweight_scheduler/run_metadata.json`

Test evaluation was not run. Automatic training, resume, B3/B5, benchmark, and
GPU stress actions remain prohibited under the active hardware-stability block.
