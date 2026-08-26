# NIH ChestX-ray14 Adapter Design

## Goal

Add a reproducible adapter that converts NIH ChestX-ray14 metadata into the
project's portable wide multi-label CSV contract without changing the core
model or loss design.

## Data Contract

`prepare_nih.py` will read the official `Data_Entry_2017.csv` and write a CSV
with these leading columns: `image_id`, `path`, `split`, and `patient_id`.
It will then include the canonical 14 NIH disease columns in their fixed
order. `path` remains relative to the supplied image root. `No Finding` becomes
an all-zero disease vector.

The canonical label list will be defined once in `baseline/labels.py` and used
by conversion, validation, data loading, and the smoke test. The model output
dimension remains derived from the supplied label-column list; for NIH this is
always 14.

## Splitting

When official NIH train/test lists are not provided, the conversion script will
split unique `Patient ID` values with deterministic seed 42: 80% train, 10%
validation, and 10% test. A patient is assigned exactly one split and the
assignment is serialized into `labels.csv`. Optional official train/test list
arguments will be accepted. With those lists, the script will preserve their
patient membership for train/test and deterministically create validation only
from training patients.

## Loading and Training

`build_dataloaders` will recognize a valid `split` column and use the stored
train and validation rows directly. It will preserve the existing deterministic
random-split fallback for legacy smoke-test CSVs without `split`. Images will
be opened as RGB to match the existing ImageNet-style ResNet18/DenseNet121
input shape `[B, 3, H, W]`. Targets are `torch.float32` and have the canonical
label dimension.

## Validation and Smoke Test

A lightweight validation command will check the schema, binary/no-missing
labels, valid split names, unique image IDs, image existence, and patient
leakage. It will print split sizes and per-split positive counts/rates for all
14 conditions. Any patient overlap is an error.

A smoke-test command will load one batch, print the image and target shapes and
dtypes, run the configured model forward, verify that logits and labels have
identical shapes, and calculate `BCEWithLogitsLoss`. It will not run a full
training epoch or change model/loss behavior.

## Error Handling

The converter will reject missing required NIH metadata columns, unknown
non-empty finding labels, ambiguous image paths, and missing images. The
validator will fail on invalid data rather than silently accepting leakage or
malformed labels. If a source cannot expose patient IDs, conversion will emit a
clear warning before performing its deterministic image-level fallback.

## Tests

Tests will build tiny temporary PNG and NIH metadata fixtures. They will prove
multi-hot conversion, all-zero No Finding conversion, deterministic
patient-level separation, schema/image/leakage validation, split-aware loading,
and one batch's forward/loss compatibility. Existing baseline tests remain
green.
