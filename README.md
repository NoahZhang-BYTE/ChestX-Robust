# Chest X-ray Multi-label Baseline

This is a minimal, configurable baseline for a hidden-test-set medical image competition. The official input/output contract is not published yet, so the training CSV schema is deliberately configurable.

## Expected local data

```text
data/
  labels.csv
  images/
    image_001.png
```

`labels.csv` must contain one image path column (default `image`) and one binary column per finding:

```csv
image,atelectasis,effusion
image_001.png,0,1
image_002.png,1,0
```

Set `data.label_cols` in `configs/baseline.yaml` to an explicit ordered list when the competition labels are known. Leave it empty to infer all columns except `image_col`.

## Install and verify

The project has been verified with Python 3.12, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and an RTX 5070 Laptop GPU. From the project root:

```powershell
uv pip install --python '.\.venv\Scripts\python.exe' numpy pandas pillow scikit-learn pyyaml tqdm pytest
```

Install the CUDA PyTorch build appropriate for the machine when creating a new environment:

```powershell
uv pip install --python '.\.venv\Scripts\python.exe' torch torchvision --index-url https://download.pytorch.org/whl/cu128
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
