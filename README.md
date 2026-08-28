# ChestX-Robust

A project aiming at developing robust multi-label chest X-ray classification models that generalize reliably across datasets, acquisition settings, and hidden test environments. It provides a configurable baseline built on the NIH ChestX-ray14 dataset, with reproducible data preparation, training, validation, and evaluation workflows.

This is a minimal, configurable baseline for a hidden-test-set medical image competition. It includes an adapter for prepared NIH ChestX-ray14 metadata.

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

Each run writes `last.pt` and the best validation checkpoint `best.pt` under `output_dir`. Checkpoints contain model and optimizer states, epoch, resolved configuration, and validation metrics.

## Validate

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Reported metrics are macro/micro AUROC, macro/micro F1, and sample F1. AUROC is reported as `nan` for a label whose validation split contains only one class; F1 remains defined with zero division handled as zero.
