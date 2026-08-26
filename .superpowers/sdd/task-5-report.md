# Task 5 Report: One-Batch Pipeline Smoke Test and Documentation

## Scope

- Added `smoke_test.py` with `run_smoke_test(config_path)` and a `SmokeTestResult` containing image, label, and logit shapes plus a finite scalar loss.
- The runner loads the YAML configuration, uses the existing dataloader/model builders and training device helpers, obtains one training batch, compares raw logit and target shapes, and computes `nn.BCEWithLogitsLoss()` without sigmoid or softmax.
- Updated `configs/baseline.yaml` to use `data.image_col: path`, `data.image_root: data/raw/images`, and `data.label_cols: []`.
- Documented NIH prepare, validation, smoke-test, and training PowerShell commands in `README.md`.
- Added a valid PNG, prepared-CSV fixture and a one-batch smoke-test assertion to `tests/test_nih.py`.

## RED Evidence

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_nih.py -q
```

Result: expected collection failure, `ModuleNotFoundError: No module named 'smoke_test'` at `from smoke_test import run_smoke_test`; 1 collection error in 3.07s.

## GREEN Evidence

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_nih.py -q
```

Result: `23 passed in 10.57s`.

### Output-completeness RED/GREEN

After the controller requested `logits.dtype` output, the smoke test was extended to assert it.

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_nih.py -q
```

Result: expected assertion failure because the output contained `logits.shape` but not `logits.dtype`; `1 failed, 22 passed in 9.64s`.

After adding the `logits.dtype` print line, the focused command is rerun during final verification.

Final focused result: `23 passed in 9.55s`.

## Full Verification

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Result: `30 passed in 11.21s`.

Final result after the output-completeness change: `30 passed in 9.50s`.

```powershell
& '.\.venv\Scripts\python.exe' prepare_nih.py --help
& '.\.venv\Scripts\python.exe' validate_data.py --help
& '.\.venv\Scripts\python.exe' smoke_test.py --help
```

Result: all three commands exited 0 and displayed their expected arguments. `smoke_test.py` displays optional `--config CONFIG` and defaults it to `configs/baseline.yaml`.

```powershell
git diff --check
```

Result: no whitespace errors; Git emitted only LF-to-CRLF working-tree warnings for pre-existing repository line-ending policy.

## Self-review

The smoke path has no NIH download or fabricated real data. The test fixture’s three valid PNGs and prepared-style labels CSV are test-only. The model architecture remains unchanged and the loss remains raw-logit `BCEWithLogitsLoss`.
