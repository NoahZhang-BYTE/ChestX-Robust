# Workflow blocked: B4 resource gate

- Stage: B4 DenseNet121 sqrt-capped `pos_weight` training
- Date: 2026-09-02 (Asia/Singapore)
- Reason: Windows committed memory crossed the user-defined 80% stop line.
- Training process: PID 17828 (launcher PID 10372); both exited after the controlled stop.
- Threshold-crossing snapshot: 58.93 GB / 73.64 GB committed (80.03%); RAM 15.15 / 31.64 GB; B4 RSS 1.88 GB; GPU 2,584 / 8,151 MiB.
- Post-stop snapshot: 52.65 GB / 73.64 GB committed (71.5%); RAM 13.21 / 31.64 GB.
- WHEA: no events returned from the last-12-hour WHEA-Logger query.
- stderr: empty at stop.
- Last durable training record: epoch 6; `history.csv`, `last.pt`, `best_macro_auroc.pt`, and `best_macro_auprc.pt` preserved.
- Test: not run.
- Recovery: not attempted; no machine-learning hyperparameters were changed.
