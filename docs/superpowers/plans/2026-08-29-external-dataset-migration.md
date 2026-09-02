# External Chest X-ray Dataset Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the local NIH ChestX-ray14 training corpus with a CheXpert-based 14-observation task, benchmark it on an independent hospital dataset, and retire the old images only after reproducibility and acceptance gates pass.

**Architecture:** First freeze the current NIH checkpoint and measure the effect of validation-only per-class threshold tuning. Stage CheXpert-v1.0-small under a new `data/staging/chexpert` namespace without altering the NIH files, convert its uncertainty-aware labels into an explicit 14-observation contract, and train a fresh ImageNet-initialized model in a new checkpoint directory. Run controlled BCE/Asymmetric Loss/class-balanced experiments, then evaluate the frozen model on BRAX; MIMIC-CXR-JPG, VinDr-CXR, and PadChest remain secondary external candidates rather than being silently mixed into training.

**Tech Stack:** Python 3.12, PyTorch/torchvision, pandas, NumPy, scikit-learn, Pillow, PyYAML, PowerShell, PhysioNet/Stanford dataset portals.

## Global Constraints

- Do not delete or overwrite `data/` or `checkpoints/` until the new dataset passes validation, smoke testing, a short training run, and a reproducibility check.
- Keep the current NIH checkpoint `checkpoints/imagenet_pretrained/best.pt` as the pre-migration reference; its 14 outputs are not semantically interchangeable with CheXpert labels.
- The deletion branch assumes the target output ontology is allowed to change. If the hidden evaluator requires the current NIH 14 labels, stop before deletion and use CheXpert only for pretraining/domain adaptation.
- Use an explicit ordered `label_cols` list for every external dataset; never rely on automatic label inference.
- Every split must be patient/study-group disjoint, persisted in the metadata CSV, and contain non-empty train and validation rows.
- Fit thresholds only on the internal validation split; use external test sets once for final reporting.
- Record dataset version, access terms, label mapping, uncertain-label policy, split policy, file manifest, and SHA-256 hashes.
- Treat PhysioNet datasets as credentialed data: CITI Data or Specimens Only training, signed DUA, and PhysioNet Credentialed Health Data License 1.5.0 are required.
- Free space on `C:` changed from 94.81 GiB to 131.13 GiB during the 2026-08-29 inspection while NIH images remained 41.963 GiB, so capacity is dynamic. Recheck immediately before acquisition and before extraction; never depend on an earlier snapshot or on free space inferred from dataset image counts.

## Dataset Decision

### Gate 0: confirm the output contract before deletion

The current checkpoint is a NIH-specific 14-output model. A CheXpert 14-output model has a different ontology despite having the same output count. Only seven concepts are defensibly comparable across the two tasks:

```text
Atelectasis, Cardiomegaly, Consolidation, Edema,
Pneumonia, Pneumothorax, Pleural Effusion (NIH: Effusion)
```

Do not map `Lung Opacity` to NIH `Infiltration`, or `Lung Lesion` to NIH `Mass`/`Nodule`. If the final evaluator expects NIH `Infiltration`, `Mass`, `Nodule`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, or `Hernia`, replacing the corpus would make the model contract incompatible. In that case keep NIH and treat CheXpert as auxiliary training data.

The remainder of this plan assumes the task is allowed to switch to the full CheXpert observation ontology.

### Current baseline signal

The logged run is not an ImageNet-only result: `model.pretrained: true` initializes ResNet18 from ImageNet, then `train.py` fine-tunes it on the local NIH `data/labels.csv` corpus. Training NIH again with the same config would repeat the same experiment. The existing checkpoint peaks at epoch 5 with validation macro AUROC `0.8204`; validation loss is lowest at epoch 4 and then rises while training loss continues falling, so epochs 6-10 show overfitting. A read-only validation sweep on 2026-08-29 also found:

| Decision rule | Macro F1 | Micro F1 |
|---|---:|---:|
| Fixed threshold `0.5` | 0.1357 | 0.1870 |
| Per-label validation-optimal thresholds | 0.3058 | 0.3638 |

This is diagnostic evidence, not a test result. It justifies implementing threshold tuning before changing the model, but the thresholds must be frozen and evaluated on the untouched NIH test split before reporting a gain. Do not use the old NIH thresholds for CheXpert or any external dataset.

### Recommended primary replacement: CheXpert-v1.0-small

The official Stanford AIMI page reports 224,316 chest radiographs from 65,240 patients for the CheXpert corpus, with associated reports and a Stanford research-use agreement. The exact row count of the `v1.0-small` download must be measured after extraction rather than inferred from the full-corpus figure. The dataset has 14 report-derived observations with uncertainty/missing values and expert-labeled reference subsets. Use the official portal at <https://aimi.stanford.edu/datasets/chexpert-chest-x-rays> and cite DOI `10.71718/y7pj-4v93` plus the CheXpert paper DOI `10.1609/aaai.v33i01.3301590`.

CheXpert is the best first replacement when scale and a consistent cross-hospital label ontology matter, but it is not an NIH 14-class drop-in. The canonical internal label order is:

```text
No_Finding, Enlarged_Cardiomediastinum, Cardiomegaly, Lung_Opacity,
Lung_Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax,
Pleural_Effusion, Pleural_Other, Fracture, Support_Devices
```

Train all 14 CheXpert observations for the replacement model so BRAX and MIMIC-CXR-JPG can be evaluated with the same label family. Restrict old-versus-new comparison tables to the seven concepts in Gate 0 and clearly label them as different datasets, not paired test results.

### External robustness candidates

| Dataset | Verified evidence and size | Access | Recommended role |
|---|---|---|---|
| BRAX v1.0.0 | 40,967 images, 24,959 studies, 19,351 patients; the same 14 CheXpert-style report labels; paper DOI `10.1038/s41597-022-01608-8`, data DOI `10.13026/ae9a-f727` | PhysioNet credentialed access, CITI training, DUA | **First external robustness set**: Brazil, PNG and DICOM available, stable `PatientID` and `AccessionNumber` |
| MIMIC-CXR-JPG v2.1.0 | 377,110 JPG images and 227,827 labeled studies; official `subject_id`, `study_id`, and split files; paper DOI `10.48550/arXiv.1901.07042`, data DOI `10.13026/jsn5-t979` | PhysioNet credentialed access, CITI training, DUA; the current portal reports the data unavailable in this region | Best large US external set if regional access becomes available; do not block the migration on it |
| VinDr-CXR v1.0.0 | 18,000 PA DICOM scans, 15,000 train/3,000 test, 17 radiologists, 22 local plus 6 global labels; paper DOI `10.1038/s41597-022-01498-w`, data DOI `10.13026/3akn-b287` | PhysioNet credentialed access, CITI training, DUA | High-quality Vietnam evaluation on an explicit mapped subset; `Nodule/Mass` is combined and patient IDs were removed, so it is not a full CheXpert-14 drop-in |
| PadChest | More than 160,000 images from about 67,000 patients, 174 findings; paper DOI `10.1016/j.media.2020.101797` | BIMCV request/data-use terms must be confirmed at acquisition time | Optional Spain domain-shift set; extensive ontology mapping makes it a later phase |

Official references: <https://aimi.stanford.edu/datasets/chexpert-chest-x-rays>, <https://physionet.org/content/mimic-cxr-jpg/2.1.0/>, <https://physionet.org/content/vindr-cxr/1.0.0/>, <https://physionet.org/content/brax/1.0.0/>, and PadChest paper DOI `10.1016/j.media.2020.101797` (PMID `32877839`).

### Acceptance gate before deleting NIH

If the hidden test or final application still expects the NIH 14 labels, do not delete NIH: use CheXpert as pretraining/domain adaptation and retain a copy of NIH for final fine-tuning and evaluation. Delete the old data only if the target contract has been confirmed to use the CheXpert ontology, the BRAX evaluation is complete, and the old metadata/checkpoint hashes have been preserved.

---

### Task 1: Freeze and inventory the current baseline

**Files:**
- Create: `docs/data-migration/2026-08-29-nih-baseline-inventory.md`
- Create: `docs/data-migration/nih-file-sha256.txt`
- Read-only: `data/labels.csv`, `data/raw/images`, `checkpoints/imagenet_pretrained/best.pt`

**Interfaces:**
- Produces a recovery record containing row counts, split counts, label prevalence, checkpoint metrics, data size, and file hashes.

- [ ] **Step 1: Write the inventory command output**

Run from `C:\胸片AI多标签高鲁棒`:

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py validate_data.py --csv data\labels.csv --data-root data\raw\images
Get-ChildItem data\raw\images -Recurse -File | Measure-Object Length -Sum
Get-FileHash data\labels.csv -Algorithm SHA256
Get-FileHash checkpoints\imagenet_pretrained\best.pt -Algorithm SHA256
```

Expected facts verified on 2026-08-29: 112,120 images/rows; train/val/test rows 89,789/11,348/10,983; no patient leakage; image payload 45,057,440,698 bytes (41.963 GiB). Expected SHA-256 values are `C0B1784DCE4215B12396D2E610D70469F91B0C068D6096081FBFEE184DF93318` for `data/labels.csv` and `BE8EF6905F358488ED6D5DD2DA8879C656084EE5D184795587A5CCEF3584B972` for the ImageNet checkpoint.

- [ ] **Step 2: Save the facts and checkpoint metadata**

Write the command output plus the checkpoint's `epoch`, `metrics`, and embedded `config.data` to `docs/data-migration/2026-08-29-nih-baseline-inventory.md`. Save the two SHA-256 lines to `docs/data-migration/nih-file-sha256.txt`.

- [ ] **Step 3: Run the existing test suite before migration**

```powershell
& $py -m pytest -q
```

Expected: `33 passed` (verified in 20.58 seconds on 2026-08-29); a future failure is a migration blocker and must be recorded before changing data paths.

---

### Task 2: Acquire and stage CheXpert without touching the NIH paths

**Files:**
- Create directories: `data\staging\chexpert\raw`, `data\staging\chexpert\converted`
- Create: `docs/data-migration/chexpert-access.md`

**Interfaces:**
- Consumes the Stanford AIMI download after the user accepts the current research-use agreement.
- Produces a staged archive and an access record; no repository source changes.

- [ ] **Step 1: Create the staging directories**

```powershell
New-Item -ItemType Directory -Force data\staging\chexpert\raw, data\staging\chexpert\converted | Out-Null
```

- [ ] **Step 2: Download through the official portal**

Open <https://aimi.stanford.edu/datasets/chexpert-chest-x-rays>, select the download link, authenticate, and accept the Stanford agreement. Confirm that the selected artifact is the downsampled `v1.0-small` release before saving it as `data\staging\chexpert\raw\CheXpert-v1.0-small.zip`; if the portal offers only the full-resolution corpus, stop and choose a larger target drive instead of renaming or partially extracting it. Do not use an unofficial mirror for the production run.

- [ ] **Step 3: Verify and extract the archive**

```powershell
Get-FileHash data\staging\chexpert\raw\CheXpert-v1.0-small.zip -Algorithm SHA256
Expand-Archive -LiteralPath data\staging\chexpert\raw\CheXpert-v1.0-small.zip -DestinationPath data\staging\chexpert\raw -Force
Test-Path data\staging\chexpert\raw\CheXpert-v1.0-small\train.csv
```

Expected: `train.csv` exists and the extracted image tree is below `data\staging\chexpert\raw\CheXpert-v1.0-small`. Recheck free space before extraction and keep enough room for both the archive and extracted tree until the SHA-256 and row-count validation succeeds.

- [ ] **Step 4: Record access and version information**

Write the portal URL, access date, archive SHA-256, dataset version, accepted terms, and the exact extracted path to `docs/data-migration/chexpert-access.md`. Do not commit images or credentials.

---

### Task 3: Add a source adapter and generic data validator

**Files:**
- Create: `baseline/external.py`
- Create: `prepare_chexpert.py`
- Create: `validate_external.py`
- Create: `tests/test_external.py`
- Modify: `README.md`

**Interfaces:**
- `baseline.external.CHEXPERT_LABEL_COLUMNS -> tuple[str, ...]`
- `baseline.external.prepare_chexpert_metadata(source_csv, image_root, output_csv, official_valid_csv=None, split_seed=42, uncertainty_policy="zero") -> pandas.DataFrame`
- `baseline.external.validate_external_labels(csv_path, data_root, label_cols, split_col="split", patient_col="patient_id", metadata_cols=()) -> pandas.DataFrame`
- `prepare_chexpert.py --source-csv ... --valid-csv ... --image-root ... --output ... --uncertainty-policy zero`
- `validate_external.py --csv ... --data-root ... --labels ...`

- [ ] **Step 1: Write failing adapter tests**

`tests/test_external.py` must create a temporary CheXpert-style CSV and PNG files, then assert:

```python
def test_prepare_chexpert_maps_uncertain_and_missing_to_zero(tmp_path):
    frame = prepare_chexpert_metadata(
        tmp_path / "train.csv",
        tmp_path / "images",
        tmp_path / "labels.csv",
        official_valid_csv=tmp_path / "valid.csv",
        uncertainty_policy="zero",
    )
    assert list(frame.columns) == [
        "image_id", "path", "split", "patient_id",
        *CHEXPERT_LABEL_COLUMNS,
    ]
    assert set(frame["split"]) <= {"train", "val", "test"}


def test_external_validator_rejects_patient_leakage_and_missing_paths(tmp_path):
    csv_path, image_root = make_external_fixture(tmp_path)
    with pytest.raises(ValueError, match="patient"):
        validate_external_labels(
            csv_path, image_root,
            ["Atelectasis", "Cardiomegaly"],
        )
```

The fixture must include CheXpert-style `Path` values containing patient directory components, all 14 observations, one uncertain value `-1`, one missing value, and a deliberate cross-split patient collision. The expected initial policy is explicit `zero`: uncertain and missing values become zero, and their counts are recorded for sensitivity analysis. A future masked-loss implementation may add `ignore`, but the current BCE pipeline must not silently consume `NaN`.

- [ ] **Step 2: Run the focused tests to verify they fail**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests\test_external.py -q
```

Expected: FAIL because the adapter and validator do not yet exist.

- [ ] **Step 3: Implement the CheXpert conversion contract**

The adapter must:

1. Read CheXpert `train.csv` with `Path`, `Sex`, `Age`, `Frontal/Lateral`, `AP/PA`, and 14 observation columns; read the official `valid.csv` when it is present.
2. Use `Path` as `image_id` and a POSIX relative `path` under the staged image root.
3. Derive `patient_id` from the patient directory component in each path and reject any path that cannot yield a stable patient identifier.
4. Preserve all `train.csv` rows as the source pool, create `val` by deterministic patient-level splitting from that pool with seed 42, and mark the official `valid.csv` rows as `test` without using them for training or threshold fitting. If `valid.csv` is absent, fail the conversion rather than inventing a test split.
5. Select the 14 ordered CheXpert labels from the Dataset Decision section, normalize spaces to underscores, and never infer labels from arbitrary columns.
6. Apply `uncertainty_policy="zero"` by mapping CheXpert `-1` and blank values to 0, and write the number of affected values per label to the conversion report. Reject `ignore` until the training loss has an explicit validity-mask interface; never write `NaN` into a CSV consumed by the current BCE loss.
7. Reject duplicate image IDs, missing image files, invalid paths, empty train/val splits, and patient overlap.

- [ ] **Step 4: Implement generic validation and CLI output**

`validate_external_labels` must check all required columns, permit only explicitly configured extra metadata columns such as `study_id`, require unique `image_id`, validate relative paths inside `data_root`, open every image, enforce numeric binary labels after conversion, require non-empty train/val/test where declared, reject patient overlap, and print per-label positive counts. The CLI must print split counts and positive rates and return a non-zero exit code on any failure.

- [ ] **Step 5: Run focused and existing tests**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests\test_external.py tests\test_baseline.py -q
```

Expected: PASS with no changes to the NIH behavior.

---

### Task 4: Produce the canonical CheXpert metadata and configuration

**Files:**
- Create: `configs/chexpert.yaml`
- Create: `docs/data-migration/chexpert-label-mapping.md`
- Create: `data/staging/chexpert/converted/labels.csv`

**Interfaces:**
- Consumes `data/staging/chexpert/raw/CheXpert-v1.0-small/train.csv` and its image root.
- Produces a 14-output CheXpert dataset contract for `train.py`.

- [ ] **Step 1: Write the conversion command**

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py prepare_chexpert.py `
  --source-csv data\staging\chexpert\raw\CheXpert-v1.0-small\train.csv `
  --valid-csv data\staging\chexpert\raw\CheXpert-v1.0-small\valid.csv `
  --image-root data\staging\chexpert\raw\CheXpert-v1.0-small `
  --output data\staging\chexpert\converted\labels.csv `
  --uncertainty-policy zero
```

Expected: a CSV with `image_id,path,split,patient_id` plus the 14 explicit CheXpert labels.

- [ ] **Step 2: Write the dedicated configuration**

`configs/chexpert.yaml` must contain these exact migration-critical values:

```yaml
seed: 42
device: auto
output_dir: checkpoints/chexpert_imagenet
data:
  csv_path: data/staging/chexpert/converted/labels.csv
  image_root: data/staging/chexpert/raw/CheXpert-v1.0-small
  image_col: path
  label_cols:
    - No_Finding
    - Enlarged_Cardiomediastinum
    - Cardiomegaly
    - Lung_Opacity
    - Lung_Lesion
    - Edema
    - Consolidation
    - Pneumonia
    - Atelectasis
    - Pneumothorax
    - Pleural_Effusion
    - Pleural_Other
    - Fracture
    - Support_Devices
  val_split: 0.2
  image_size: 224
  batch_size: 64
  num_workers: 4
  prefetch_factor: 2
  persistent_workers: true
model:
  name: resnet18
  pretrained: true
training:
  epochs: 15
  learning_rate: 0.0003
  weight_decay: 0.0001
  threshold: 0.5
  amp: true
```

- [ ] **Step 3: Document the ontology decision**

`docs/data-migration/chexpert-label-mapping.md` must list the exact source-to-internal mapping for all 14 observations, record the chosen uncertainty policy, and define the seven-label legacy comparison subset. State explicitly that no mapping is made from `Lung_Opacity` to NIH `Infiltration` or from `Lung_Lesion` to NIH `Mass`/`Nodule`.

---

### Task 5: Validate, smoke test, and run a short migration experiment

**Files:**
- Modify: `docs/data-migration/2026-08-29-nih-baseline-inventory.md`
- Create: `docs/data-migration/chexpert-validation.md`

- [ ] **Step 1: Validate the converted data**

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py validate_external.py `
  --csv data\staging\chexpert\converted\labels.csv `
  --data-root data\staging\chexpert\raw\CheXpert-v1.0-small `
  --labels No_Finding Enlarged_Cardiomediastinum Cardiomegaly Lung_Opacity Lung_Lesion Edema Consolidation Pneumonia Atelectasis Pneumothorax Pleural_Effusion Pleural_Other Fracture Support_Devices
```

Expected: zero schema/path errors, zero patient intersections, and non-zero positive counts for every selected class in train and validation.

- [ ] **Step 2: Run a loader/model smoke test**

```powershell
& $py smoke_test.py --config configs\chexpert.yaml
```

Expected: one batch loads as RGB tensors of shape `(batch, 3, 224, 224)` and logits have shape `(batch, 14)`.

- [ ] **Step 3: Run a one-epoch acceptance run**

Temporarily create `configs/chexpert_smoke.yaml` by copying `configs/chexpert.yaml`, changing `output_dir` to `checkpoints/chexpert_smoke` and `training.epochs` to `1`, then run:

```powershell
& $py train.py --config configs\chexpert_smoke.yaml
```

Expected: `last.pt` and `best.pt` appear under `checkpoints/chexpert_smoke`, the checkpoint config points to the staged CheXpert paths, and the model output dimension is 14.

- [ ] **Step 4: Record validation evidence**

Write row counts, per-label prevalence, image-read failures, split intersections, smoke output, and checkpoint metadata to `docs/data-migration/chexpert-validation.md`.

---

### Task 6: Train, evaluate, and optimize thresholds without leakage

**Files:**
- Create: `evaluate.py`
- Create: `tune_thresholds.py`
- Create: `docs/data-migration/chexpert-results.md`
- Create: `tests/test_evaluation.py`
- Modify: `baseline/data.py`
- Modify: `baseline/engine.py` only if a reusable evaluation helper is needed

**Interfaces:**
- `baseline.data.build_dataloader_for_split(csv_path, image_root, split, image_col, label_cols, image_size, batch_size, num_workers) -> tuple[DataLoader, list[str]]`
- `evaluate.py --config configs/chexpert.yaml --checkpoint checkpoints/chexpert_imagenet/best.pt --split test`
- `tune_thresholds.py --config configs/chexpert.yaml --checkpoint checkpoints/chexpert_imagenet/best.pt --split val --output docs/data-migration/chexpert-thresholds.json`

- [ ] **Step 1: Write evaluation tests**

Test that evaluation loads the saved label order, refuses a checkpoint whose label list differs from the config, computes per-label AUROC/AUPRC/F1/sensitivity/specificity, and never fits thresholds on `test`. For each label, tune across the unique validation probabilities, maximize binary F1, and break ties by choosing the highest threshold. If a validation label has no positives, write threshold `1.0` plus an `unavailable` flag rather than manufacturing a result.

- [ ] **Step 2: Establish the threshold-only NIH control before deleting images**

```powershell
& ".\.venv\Scripts\python.exe" tune_thresholds.py `
  --config configs\baseline.yaml `
  --checkpoint checkpoints\imagenet_pretrained\best.pt `
  --split val `
  --output docs\data-migration\nih-thresholds.json
& ".\.venv\Scripts\python.exe" evaluate.py `
  --config configs\baseline.yaml `
  --checkpoint checkpoints\imagenet_pretrained\best.pt `
  --split test `
  --thresholds docs\data-migration\nih-thresholds.json
```

Report fixed-0.5 and frozen-threshold test metrics side by side. The already observed validation change from macro/micro F1 `0.1357/0.1870` to `0.3058/0.3638` is not a substitute for this test result.

- [ ] **Step 3: Train the full CheXpert replacement model**

```powershell
& ".\.venv\Scripts\python.exe" train.py --config configs\chexpert.yaml
```

Expected: use `checkpoints/chexpert_imagenet/best.pt` for selection, not `last.pt`; preserve the old NIH checkpoints untouched.

- [ ] **Step 4: Tune thresholds on validation only**

```powershell
& ".\.venv\Scripts\python.exe" tune_thresholds.py `
  --config configs\chexpert.yaml `
  --checkpoint checkpoints\chexpert_imagenet\best.pt `
  --split val `
  --output docs\data-migration\chexpert-thresholds.json
```

Use a fixed method such as per-class F1 maximization with a documented tie-break rule. Do not inspect test labels while selecting thresholds.

- [ ] **Step 5: Evaluate the held-out test split once**

```powershell
& ".\.venv\Scripts\python.exe" evaluate.py `
  --config configs\chexpert.yaml `
  --checkpoint checkpoints\chexpert_imagenet\best.pt `
  --split test `
  --thresholds docs\data-migration\chexpert-thresholds.json
```

Report per-class metrics, macro/micro AUROC, macro/micro F1, AUPRC, calibration summary, and the exact checkpoint/config hashes in `docs/data-migration/chexpert-results.md`.

---

### Task 7: Run controlled imbalance, uncertainty, and augmentation ablations

**Files:**
- Create: `baseline/losses.py`
- Create: `tests/test_losses.py`
- Create: `configs/experiments/chexpert_bce.yaml`
- Create: `configs/experiments/chexpert_asl.yaml`
- Create: `configs/experiments/chexpert_cb_bce.yaml`
- Create: `configs/experiments/chexpert_focal.yaml`
- Create: `configs/experiments/chexpert_asl_conservative_aug.yaml`
- Create: `docs/data-migration/chexpert-ablation-matrix.md`
- Modify: `baseline/engine.py`
- Modify: `baseline/data.py`
- Modify: `baseline/external.py`

**Interfaces:**
- `baseline.losses.build_loss(loss_config: Mapping[str, Any], positive_counts: Tensor, negative_counts: Tensor) -> nn.Module`
- Supported names: `bce`, `asymmetric`, `class_balanced_bce`, and `focal`
- `data.augmentation` accepts only `baseline` or `conservative`
- `uncertainty_policy` accepts `zero` or `paper_informed`; `paper_informed` maps uncertain Atelectasis and Edema to positive and all other uncertain/missing values to zero

- [ ] **Step 1: Write failing loss and augmentation tests**

Test that all four losses accept logits/targets shaped `(batch, labels)`, return a finite scalar, backpropagate finite gradients, and reject unknown loss names. Assert that ASL with `gamma_neg=4`, `gamma_pos=1`, and `clip=0.05` downweights an easy negative more strongly than plain BCE, and that class-balanced weights are finite and capped at the configured maximum. Test that validation transforms are identical across augmentation modes and deterministic for a fixed image.

- [ ] **Step 2: Run the focused tests to verify they fail**

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests\test_losses.py tests\test_baseline.py -q
```

Expected: FAIL because the loss factory and configurable augmentation policy do not exist.

- [ ] **Step 3: Implement the loss factory and preserve BCE defaults**

Use logits directly and reduction `mean`. For ASL, expose `gamma_neg`, `gamma_pos`, and `clip`; start from the official-paper configuration `4`, `1`, and `0.05` but keep them in YAML. For class-balanced BCE, compute per-label positive weights from the effective number `(1 - beta) / (1 - beta ** n)` with `beta=0.9999`, normalize the mean weight to `1`, and cap any weight at `20`. For focal loss, start with `gamma=2`. A missing `training.loss` block must remain exactly equivalent to the current `BCEWithLogitsLoss` behavior.

Primary references: Asymmetric Loss DOI `10.48550/arXiv.2009.14119`, class-balanced loss DOI `10.48550/arXiv.1901.05555`, and focal loss DOI `10.1109/ICCV.2017.324`.

- [ ] **Step 4: Add one conservative medical-image augmentation mode**

Keep the existing resize and horizontal flip as `baseline`. Define `conservative` as resize plus random horizontal flip, `RandomAffine(degrees=7, translate=(0.02, 0.02), scale=(0.95, 1.05))`, and `ColorJitter(brightness=0.1, contrast=0.1)` before tensor conversion and ImageNet normalization. Do not add aggressive crops, vertical flips, MixStyle, or AugMix in this migration; consider them only after the first external-domain error analysis.

- [ ] **Step 5: Materialize independent experiment configurations**

Every configuration must use the same CheXpert CSV, patient split, ResNet18 ImageNet initialization, batch size, learning rate, seed 42, and epoch budget. Change exactly one factor at a time and write to a distinct `checkpoints/chexpert_<experiment>` directory. Keep the initial uncertainty policy `zero`; add a separate metadata/config pair for `paper_informed` rather than overwriting the canonical CSV.

- [ ] **Step 6: Run smoke tests, then the one-seed screen**

```powershell
$py = ".\.venv\Scripts\python.exe"
Get-ChildItem configs\experiments\chexpert_*.yaml | ForEach-Object {
  & $py smoke_test.py --config $_.FullName
}
& $py -m pytest -q
```

Train BCE first, followed by ASL, class-balanced BCE, focal, and finally conservative augmentation on the best loss. Rank the screen using validation macro AUPRC, with macro AUROC and validation-only tuned macro F1 as secondary metrics. Never select a method using CheXpert test or BRAX results.

- [ ] **Step 7: Confirm the top two candidates across seeds**

Repeat only the top two validation candidates with seeds `42`, `43`, and `44`. Report mean and standard deviation for macro AUPRC, macro AUROC, and tuned macro/micro F1. Promote a new default only when its mean macro AUPRC exceeds BCE and no shared label suffers an unexplained collapse.

- [ ] **Step 8: Record the ablation decision**

Write configuration hashes, uncertainty policy, seed-level results, thresholds, runtime, and the chosen checkpoint to `docs/data-migration/chexpert-ablation-matrix.md`. Treat the `paper_informed` uncertainty run as a separate ablation because CheXpert found disease-dependent uncertainty behavior; do not attribute its change to the loss function.

---

### Task 8: Add cross-dataset robustness evaluation

**Files:**
- Create: `docs/data-migration/external-evaluation-matrix.md`
- Create: `baseline/adapters/__init__.py`
- Create: `baseline/adapters/brax.py`
- Create: `prepare_brax.py`
- Create: `data/staging/brax/converted/labels.csv`
- Create: `tests/test_external_evaluation.py`
- Modify: `evaluate.py`

**Interfaces:**
- `baseline.adapters.brax.prepare_brax_metadata(source_csv, image_root, output_csv, uncertainty_policy="zero") -> pandas.DataFrame`
- Output columns: `image_id,path,split,patient_id,study_id,<CHEXPERT_LABEL_COLUMNS>` with every row assigned to `test`
- `evaluate.py --group-col study_id --group-reduction max` aggregates view probabilities before calculating study-level metrics

- [ ] **Step 1: Obtain BRAX as the first credentialed external set**

Apply for BRAX through PhysioNet, complete CITI training, sign the DUA, and stage it under `data\staging\brax` without mixing it into CheXpert training files. Use the provided PNG tree for the first evaluation and retain `PatientID` and `AccessionNumber`. If BRAX access is denied, use VinDr-CXR as the declared subset fallback; do not silently switch sources.

- [ ] **Step 2: Write failing adapter and study-aggregation tests**

Create a fixture with two images in one `AccessionNumber`, a second patient/study, all 14 source labels, one `-1`, and one blank. Assert the canonical label order, relative PNG paths, `PatientID -> patient_id`, `AccessionNumber -> study_id`, `split == test`, and documented uncertainty counts. For aggregation, assert that two view probabilities are reduced by per-label maximum and that the shared study target is counted once.

- [ ] **Step 3: Convert the source through an explicit ontology map**

For BRAX, rename the source columns to the same 14-label CheXpert order and run the primary evaluation with the same `zero` uncertainty policy as the selected CheXpert model. Record counts of `-1` and blank values per label and add a sensitivity table that excludes uncertain entries when positive counts permit. For a future VinDr fallback, create and document a strict subset map and do not split `Nodule/Mass` into two labels. Keep all external images out of threshold fitting and training.

- [ ] **Step 4: Validate the BRAX conversion**

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py prepare_brax.py `
  --source-csv data\staging\brax\raw\master_spreadsheet.csv `
  --image-root data\staging\brax\raw `
  --output data\staging\brax\converted\labels.csv `
  --uncertainty-policy zero
& $py validate_external.py `
  --csv data\staging\brax\converted\labels.csv `
  --data-root data\staging\brax\raw `
  --metadata-cols study_id `
  --labels No_Finding Enlarged_Cardiomediastinum Cardiomegaly Lung_Opacity Lung_Lesion Edema Consolidation Pneumonia Atelectasis Pneumothorax Pleural_Effusion Pleural_Other Fracture Support_Devices
```

Expected: 40,967 image rows, 24,959 unique studies, 19,351 unique patients, readable relative PNG paths, and no use of BRAX data in training or threshold selection. Stop if measured release counts differ without a documented version explanation.

- [ ] **Step 5: Run the fixed CheXpert model on the external set**

Use the CheXpert validation thresholds unchanged and aggregate multiple views by study-level maximum before the primary calculation. Report image- and study-level metrics, the delta from CheXpert test to BRAX, and patient-clustered bootstrap 95% confidence intervals per label when the positive count permits.

- [ ] **Step 6: Write the robustness decision**

Conclude whether the model is suitable for replacement based on a predeclared criterion: no missing class, no patient leakage, acceptable per-class AUPRC, and a documented external performance drop rather than a single aggregate AUROC.

---

### Task 9: Retire the NIH data only after the acceptance gate

**Files:**
- Modify: `README.md`
- Modify: `configs/baseline.yaml` only if the new dataset is intentionally made the default
- Create: `docs/data-migration/retirement-record.md`

- [ ] **Step 1: Confirm the deletion scope**

The only old training-data targets are `data\labels.csv` and `data\raw\images`; the raw NIH metadata under `data\raw` and all old checkpoints are separate recovery artifacts. Re-run:

```powershell
Test-Path data\labels.csv
Test-Path data\raw\images
Get-ChildItem data\raw\images -Recurse -File | Measure-Object Length -Sum
```

Do not delete until the output matches the inventory and the acceptance gate is signed off.

- [ ] **Step 2: Prefer recoverable archival**

Move the exact old data targets to an explicitly named archive outside the active data path, for example `D:\chestx-robust-archive\nih-2026-08-29\data\labels.csv` and `D:\chestx-robust-archive\nih-2026-08-29\data\raw\images`, after verifying the destination has at least 42 GiB free and the SHA-256 inventory is preserved.

- [ ] **Step 3: Delete only after archive verification**

If permanent deletion is still required after archive verification, use the two exact targets only:

```powershell
Remove-Item -LiteralPath data\labels.csv -Force
Remove-Item -LiteralPath data\raw\images -Recurse -Force
```

Immediately verify `Test-Path data\labels.csv` and `Test-Path data\raw\images` are both `False`, and record the timestamp, operator, archive location, and recovery status in `docs/data-migration/retirement-record.md`.

- [ ] **Step 4: Update the default documentation/configuration**

Only after the new run is accepted, change README commands and any default config to point to `configs/chexpert.yaml`. Keep the historical NIH inventory and old checkpoint paths documented as a prior experiment.

---

## Self-Review Checklist

- The plan covers acquisition, schema conversion, uncertainty handling, patient-level splitting, validation, training, threshold fitting, imbalance-loss ablations, conservative augmentation, test evaluation, external robustness, and retirement.
- No old data is deleted before a reversible archive and acceptance gate.
- Gate 0 prevents an apparently 14-output CheXpert model from being substituted for an incompatible NIH 14-output contract.
- CheXpert concepts are not falsely renamed to NIH concepts; the seven-label legacy comparison subset is explicit while the replacement model retains the full CheXpert-14 ontology.
- Existing NIH behavior remains covered by the current tests, while external behavior gets focused tests before implementation.
- The training and evaluation artifacts use independent output paths and carry the new label order in their configs.
- Loss, uncertainty policy, and augmentation are changed one factor at a time; external BRAX results never select a model or threshold.
