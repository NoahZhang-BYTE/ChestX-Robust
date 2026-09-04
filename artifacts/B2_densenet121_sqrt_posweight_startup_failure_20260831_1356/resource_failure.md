# B2 Startup Resource Failure

Command:

```powershell
.\.venv\Scripts\python.exe formal_train.py --config configs/formal_b2_densenet121_sqrt_posweight.yaml
```

The process completed configuration, train-only pos_weight calculation, CUDA criterion placement, and DenseNet121 creation. It did not enter the first training batch. The observed native error was:

```text
OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
```

At the failure point, Windows committed memory was 136,857,030,656 bytes against a commit limit of 137,049,681,920 bytes, leaving approximately 192 MiB of commit headroom. GPU memory was 0 MiB used and 7,891 MiB free.

The process was stopped after it made no progress for three minutes. No checkpoint, history, validation, or test artifact was produced. This directory contains only the pre-training configuration and train-only pos_weight table.
