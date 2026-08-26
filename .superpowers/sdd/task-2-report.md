# Task 2: NIH Metadata Conversion and Patient Splits

## Status

Complete.

## Scope Delivered

- Added `baseline.nih.prepare_nih_metadata` to convert NIH metadata with the required `Image Index`, `Finding Labels`, and `Patient ID` fields into a CSV ordered as `image_id`, `path`, `split`, `patient_id`, then the canonical 14 NIH label columns.
- Resolves images recursively below `image_root`, stores portable POSIX relative paths, and rejects missing images, duplicate metadata image IDs, and ambiguous repeated image files.
- Converts pipe-delimited labels into numeric multi-hot values, maps `No Finding` to an all-zero target, and rejects unknown non-empty findings.
- Creates deterministic patient-level 80/10/10 splits using `numpy.random.default_rng(seed)`, preserving a lone patient for training while allowing tiny validation/test partitions to be empty.
- Supports paired official train/test list files. Test membership remains fixed; validation is created only by splitting official-training patients; list coverage, image overlap, and patient leakage are rejected.
- Added `prepare_nih.py` with `--metadata`, `--image-root`, `--output`, `--seed`, `--train-list`, and `--test-list`.

## TDD Evidence

1. Added `tests/test_nih.py` before the module existed; it failed at collection with `ModuleNotFoundError: No module named 'baseline.nih'`.
2. Added the converter and CLI; the NIH tests passed.
3. Added a `No Finding|Mass` boundary test; it failed because `Mass` was encoded positive, then passed after `No Finding` was made all-zero.
4. Added a single-patient split test; it failed because the patient was assigned to test, then passed after reserving one training patient.

## Verification

- `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_nih.py -q`: 8 passed.
- `& '.\\.venv\\Scripts\\python.exe' -m pytest -q`: 14 passed.
- `& '.\\.venv\\Scripts\\python.exe' prepare_nih.py --help`: exits 0 and lists all requested arguments.
- `git diff --check`: clean.

## Concerns

- The converter intentionally requires paired official list files to cover every metadata image. This makes incomplete list inputs fail early rather than silently assigning unpublished split membership.
- Task 3 is still responsible for validating a prepared CSV independently; this task only validates the source metadata and conversion inputs.

## Review Fix: Empty Metadata

- Added `test_prepare_nih_rejects_schema_valid_empty_metadata` before changing the converter. The RED run failed with the reported `KeyError` while the converter selected absent label columns from an empty frame.
- `_validate_metadata` now rejects a schema-valid empty NIH metadata CSV with `ValueError: NIH metadata contains no rows` before image indexing or label construction.
- Focused verification: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests\\test_nih.py -q` reported `9 passed in 3.70s`.
