### Task 3: Strict Data Validation and Statistics

**Files:**
- Modify: `baseline/nih.py`
- Create: `validate_data.py`
- Modify: `tests/test_nih.py`

**Interfaces:**
- Consumes: `validate_labels_csv(csv_path, data_root)`.
- Produces: validated DataFrame and printed `train`/`val`/`test` counts plus positive count/rate for every canonical label; raises `ValueError` for malformed content or leakage.

- [ ] **Step 1: Write failing tests**

```python
def test_validator_reports_split_counts_and_rejects_patient_leakage(tmp_path, capsys):
    labels_csv = prepare_fixture_labels_csv(tmp_path)
    validate_labels_csv(labels_csv, tmp_path / "images")
    assert "Train:" in capsys.readouterr().out

    leaked = pd.read_csv(labels_csv)
    leaked.loc[1, "split"] = "test"
    leaked.to_csv(labels_csv, index=False)
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_labels_csv(labels_csv, tmp_path / "images")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_nih.py -q`

Expected: failure because `validate_labels_csv` does not yet exist.

- [ ] **Step 3: Implement validation and CLI**

Validate the required schema, non-null cells, label values limited to 0/1, allowed split values, unique `image_id`, each relative path existing below `data_root`, and disjoint patient-ID sets across all pairs of splits. Print image totals and each label's `positive / total = percent` for every split. Add CLI positional/options for `--csv` and `--data-root`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `& '.\\.venv\\Scripts\\python.exe' -m pytest tests/test_nih.py -q`

Expected: PASS.

## Controller Clarification

The output contract includes `patient_id` from Task 2, so treat it as required
for validation and always fail on cross-split patient overlap. `data_root` is
the same directory passed as `--image-root` during conversion; paths in the CSV
are POSIX-relative to that root. Reports should print all 14 canonical labels
even when a split has zero rows, using `0 / 0 = 0.0%` for its rate.


