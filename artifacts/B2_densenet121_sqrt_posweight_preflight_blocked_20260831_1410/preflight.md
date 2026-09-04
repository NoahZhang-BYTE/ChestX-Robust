# B2 Preflight Blocked

The B2 configuration was updated to use `num_workers: 0`. The only algorithmic loss remains `BCEWithLogitsLoss` with train-only sqrt-capped pos_weight.

No smoke or formal training was started because the Windows commit gate was not met:

- Physical RAM total: 31.637 GB
- Physical RAM used: 16.153 GB
- Physical RAM available: 15.484 GB
- Commit used: 125.130 GB
- Commit limit: 127.637 GB
- Commit headroom: 2.507 GB
- Commit utilization: 98.036%
- GPU memory used / total: 0 / 8151 MiB

At the time of this check, no Python, MATLAB, Docker, or WSL process was running. The largest visible user process private-memory consumers were Steam (1.070 GB), ChatGPT Classic (0.750 GB), ChatGPT (0.730 GB), MsMpEng (0.690 GB), and mysqld (0.570 GB). These visible processes do not account for the near-exhausted system commit limit.

OpenBLAS, OMP, and MKL thread settings were not applied because the training process was not started.
