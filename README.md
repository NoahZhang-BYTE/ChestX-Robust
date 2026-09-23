# ChestX-Robust

A project aiming at developing robust multi-label chest X-ray classification models that generalize reliably across datasets, acquisition settings, and hidden test environments. It provides a configurable baseline built on the NIH ChestX-ray14 dataset, with reproducible data preparation, training, validation, and evaluation workflows.

The 2026 NCCCU competition contract is now published: 10 classes, organizer-only
training data, study-level submission rows, macro-AUC as the primary metric,
and macro-F1 as the tie-breaker. See
[docs/COMPETITION_SPEC.md](docs/COMPETITION_SPEC.md) before adapting this code.
The existing NIH models, labels, and reported scores are historical research
artifacts, not competition-ready models.

## Competition readiness

Do not submit the existing B4+B5 ensemble. It was trained for the 14-label NIH
schema and uses data that is outside the organizer-provided competition set.
The competition requires a new 10-output model trained only from the official
data and an inference path that aggregates one or more images by `Study_id`.
The B4/B5 loss pairing, ensemble workflow, leakage controls, validation-only
tuning discipline, and long-tail experiment conclusions remain the starting
methodology. The old weights, checkpoints, thresholds, and per-class ensemble
parameters do not transfer. The first formal competition baseline is a clean
retraining of both branches on the official 10-class data, followed by explicit
study-level aggregation and complementarity analysis.

The official class map and submission example are currently embedded as images
on the website. Add the downloaded `sample_submission.csv` and official metadata
under the ignored `data/` directory before implementing the final label map or
submission writer. The published prose is ambiguous about whether all ten
probabilities or only classes at or above `0.5` must be emitted, so the sample
file must be treated as authoritative.

## Historical NIH project status

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

## Historical NIH data

```text
data/
  labels.csv
  raw/images/
    images_001/image_00001.png
```

The existing default configuration expects `labels.csv` with a relative `path`
column and the canonical NIH disease columns:

```csv
image_id,path,split,patient_id,Atelectasis,Cardiomegaly,...,Hernia
00000001_000.png,images_001/00000001_000.png,train,1,0,1,...,0
```

Leave `data.label_cols` empty only for the historical NIH workflow. A competition
configuration must set an explicit ordered list matching the official 10-class
mapping; do not rely on automatic metadata-column inference.

## Install and verify

The project has been verified locally with Python 3.12, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, and an RTX 5070 Laptop GPU. The competition runtime is
PyTorch 2.10.0, torchvision 0.25.0, CUDA 12.8, and an NVIDIA T4; final packaging
and latency must be verified in that environment. From the project root:

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
