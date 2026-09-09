# ChestX-Robust

A project aiming at developing robust multi-label chest X-ray classification models that generalize reliably across datasets, acquisition settings, and hidden test environments. It provides a configurable baseline built on the NIH ChestX-ray14 dataset, with reproducible data preparation, training, validation, and evaluation workflows.

This is a minimal, configurable baseline for a hidden-test-set medical image competition. It includes an adapter for prepared NIH ChestX-ray14 metadata.

## Current project status

The current evidence-backed candidate is the B4+B5 DenseNet121 probability
ensemble (`0.4 x B4 + 0.6 x B5`) with validation-only weight and threshold
selection. Its frozen test report is under
`artifacts/B4B5_ensemble_20260904_1300/`. The full project audit, including
stale workflow records, reproducibility risks, and the recommended cleanup
order, is in [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md).

The repository contains historical training, recovery, and evaluation records.
The reconciled status is stored in
`artifacts/canonical_run_manifest.json`; `workflow_state.json` and
`outputs/experiment_registry.csv` are generated ledger views. The manifest
keeps `B3_test` pending because no standalone B3 test bundle was found, while
the B4/B5 final candidate is complete. No workflow command may implicitly start
training from a stale stage.

## Expected local data

```text
data/
  labels.csv
  raw/images/
    images_001/image_00001.png
```

The default configuration expects `labels.csv` with a relative `path` column and the canonical NIH disease columns:

```csv
image_id,path,split,patient_id,Atelectasis,Cardiomegaly,...,Hernia
00000001_000.png,images_001/00000001_000.png,train,1,0,1,...,0
```

Leave `data.label_cols` empty to select the canonical 14 NIH labels from a prepared CSV. Set it to an explicit ordered list for a different dataset schema.

## Install and verify

The project has been verified with Python 3.12, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and an RTX 5070 Laptop GPU. From the project root:

```powershell
uv pip install --python '.\.venv\Scripts\python.exe' numpy pandas pillow scikit-learn pyyaml tqdm pytest
```

Install the CUDA PyTorch build appropriate for the machine when creating a new environment:

```powershell
uv pip install --python '.\.venv\Scripts\python.exe' torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Prepare NIH data

Download the NIH image archives and `Data_Entry_2017.csv`, extract images below `data/raw/images`, then run:

```powershell
& '.\.venv\Scripts\python.exe' prepare_nih.py --metadata data/raw/Data_Entry_2017.csv --image-root data/raw/images --output data/labels.csv
```

Validate the prepared metadata and image paths:

```powershell
& '.\.venv\Scripts\python.exe' validate_data.py --csv data/labels.csv --data-root data/raw/images
```

Run one CPU or CUDA batch through the loader, model, and loss before training:

```powershell
& '.\.venv\Scripts\python.exe' smoke_test.py --config configs/baseline.yaml
```

## Train

```powershell
& '.\.venv\Scripts\python.exe' train.py --config configs/baseline.yaml
```

The default backbone is ResNet18. Set `model.name` to `densenet121` to use DenseNet121. The loss is `BCEWithLogitsLoss`; logits are converted to probabilities with sigmoid for metrics.

For CUDA training, the default configuration uses AMP mixed precision, pinned-memory non-blocking transfers, four persistent data-loader workers, and a batch size of 64. If GPU memory is exhausted, lower `data.batch_size` to 32 or 16; if GPU utilization remains low, increase `data.num_workers` gradually (typically up to the number of physical CPU cores).

Each run writes `last.pt` and the best validation checkpoint `best.pt` under `output_dir`. Checkpoints contain model and optimizer states, epoch, resolved configuration, and validation metrics.

## Validate

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Reported metrics are macro/micro AUROC, macro/micro F1, and sample F1. AUROC is reported as `nan` for a label whose validation split contains only one class; F1 remains defined with zero division handled as zero.

## Final frozen ensemble analysis

The completed B4+B5 ensemble analysis is reproducible from the persisted test
arrays and validation-only thresholds:

```powershell
uv run --isolated --no-project --python 3.13 --with "matplotlib==3.11.1" --with "pandas==3.0.0" --with "scikit-learn==1.7.1" python final_analysis.py
```

Outputs are written to `artifacts/B4B5_ensemble_20260904_1300/final_analysis/`:
ROC/PR curves, per-class performance bars, frozen-threshold confusion counts,
prevalence-vs-F1/AUPRC, ranked FP/FN cases, CSV tables, and a short report.
