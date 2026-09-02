"""Render a deterministic B2 final report from frozen B2 analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: object, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    result.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return result


def _count_hard_labels(hard_dir: Path, label: str) -> dict[str, int]:
    return {kind: len(_rows(hard_dir / label / f"{kind}.csv")) for kind in ("TP", "FP", "FN", "TN")}


def generate_b2_final_report(output_dir: str | Path = "outputs/B2_densenet121_sqrt_posweight") -> Path:
    output_dir = Path(output_dir)
    post = output_dir / "post_analysis"
    summary = _load_json(post / "analysis_summary.json")
    selection = _load_json(post / "checkpoint_selection.json")
    validation = _load_json(output_dir / "thresholds" / "validation_metrics.json")
    test_fixed = _load_json(output_dir / "test" / "test_fixed05_metrics.json")
    test_tuned = _load_json(output_dir / "test" / "test_tuned_metrics.json")
    comparison = _load_json(post / "B1_vs_B2_validation_summary.json")
    history = pd.read_csv(output_dir / "history.csv")
    val_per_label = _rows(output_dir / "thresholds" / "threshold_tuning_val.csv")
    test_per_label = {row["label_name"]: row for row in _rows(output_dir / "test" / "test_per_label_comparison.csv")}
    cooccurrence = _load_json(post / "summary.json")
    config = yaml.safe_load((output_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    hard_dir = post / "hard_labels"

    content: list[str] = [
        "# B2 Final Report",
        "",
        "## Protocol Integrity",
        "",
        "- B2 training is complete at 10 epochs. The auditable history is epochs 3-10; no values for epochs 1-2 were reconstructed.",
        f"- Selected checkpoint: epoch {selection['selected_epoch']} (`best.pt`), because it had the best observed macro AUROC ({_fmt(selection['macro_auroc'], 10)}) and macro AUPRC ({_fmt(selection['macro_auprc'], 10)}).",
        "- Per-label thresholds were fitted once on validation, frozen before test, and SHA256-verified by the test runner. Test did not affect checkpoint selection or threshold fitting.",
        "",
        "## Configuration",
        "",
        f"- Backbone: {config['model']['name']}; ImageNet pretrained: {config['model']['pretrained']}; image size: {config['data']['image_size']}; batch: {config['data']['batch_size']}.",
        f"- Loss: BCEWithLogitsLoss + train-only sqrt-capped pos_weight (cap {config['training']['loss']['cap']}); AdamW, lr {config['training']['learning_rate']}, weight decay {config['training']['weight_decay']}.",
        f"- AMP: {config['training']['amp']}; DataLoader: num_workers={config['data']['num_workers']}, prefetch_factor={config['data']['prefetch_factor']}, persistent_workers={config['data']['persistent_workers']}.",
        "- B1 used num_workers=4. B2 used num_workers=0 because of Windows multiprocessing/commit instability; this changes loading throughput, not samples, model, loss definition, optimizer, batch size, augmentation, split, or evaluation protocol.",
        "",
        "## Training Trajectory",
        "",
    ]
    content.extend(_table(
        ["Epoch", "Train loss", "Val loss", "Macro AUROC", "Macro AUPRC", "Macro F1 @ 0.5"],
        [[int(row.epoch), _fmt(row.train_loss), _fmt(row.val_loss), _fmt(row.macro_auroc), _fmt(row.macro_auprc), _fmt(row.macro_f1)] for row in history.itertuples()],
    ))
    content.extend([
        "",
        f"Minimum validation loss was epoch {summary['minimum_val_loss_epoch']} ({_fmt(summary['minimum_val_loss'])}). Epoch 10 versus epoch 9: val loss {_fmt(summary['epoch_10_validation_degradation']['vs_epoch_9_val_loss_delta'])}; macro AUROC {_fmt(summary['epoch_10_validation_degradation']['vs_epoch_9_macro_auroc_delta'])}; macro AUPRC {_fmt(summary['epoch_10_validation_degradation']['vs_epoch_9_macro_auprc_delta'])}. This is mild overfitting / generalization degradation, so B2 was not extended.",
        "",
        "## Validation Threshold Analysis",
        "",
    ])
    content.extend(_table(
        ["Metric", "Fixed 0.5", "Validation tuned"],
        [["Macro F1", _fmt(validation['macro_F1_fixed05']), _fmt(validation['macro_F1_tuned'])], ["Micro F1", _fmt(validation['micro_F1_fixed05']), _fmt(validation['micro_F1_tuned'])], ["Sample F1", _fmt(validation['sample_F1_fixed05']), _fmt(validation['sample_F1_tuned'])], ["Macro AUROC", _fmt(validation['macro_AUROC']), _fmt(validation['macro_AUROC'])], ["Macro AUPRC", _fmt(validation['macro_AUPRC']), _fmt(validation['macro_AUPRC'])]],
    ))
    content.extend([
        "",
        "Threshold tuning uses the fixed candidate set 0.01..0.99, independently maximizes binary F1 per class, and resolves equal maxima by choosing the candidate nearest 0.5. AUROC and AUPRC are threshold-independent.",
        "",
        "## Frozen-Threshold Test",
        "",
    ])
    content.extend(_table(
        ["Metric", "Fixed 0.5", "Validation-frozen tuned"],
        [["Macro AUROC", _fmt(test_fixed['macro_AUROC']), _fmt(test_tuned['macro_AUROC'])], ["Micro AUROC", _fmt(test_fixed['micro_AUROC']), _fmt(test_tuned['micro_AUROC'])], ["Macro AUPRC", _fmt(test_fixed['macro_AUPRC']), _fmt(test_tuned['macro_AUPRC'])], ["Micro AUPRC", _fmt(test_fixed['micro_AUPRC']), _fmt(test_tuned['micro_AUPRC'])], ["Macro F1", _fmt(test_fixed['metrics']['macro_F1']), _fmt(test_tuned['metrics']['macro_F1'])], ["Micro F1", _fmt(test_fixed['metrics']['micro_F1']), _fmt(test_tuned['metrics']['micro_F1'])], ["Sample F1", _fmt(test_fixed['metrics']['sample_F1']), _fmt(test_tuned['metrics']['sample_F1'])]],
    ))
    content.extend(["", "## Per-label Validation and Frozen-Test Results", ""])
    content.extend(_table(
        ["Label", "Val threshold", "Val F1 tuned", "Test AUROC", "Test AUPRC", "Test precision", "Test recall", "Test F1"],
        [[row['label_name'], _fmt(row['best_threshold'], 2), _fmt(row['f1_tuned']), _fmt(test_per_label[row['label_name']]['auroc']), _fmt(test_per_label[row['label_name']]['auprc']), _fmt(test_per_label[row['label_name']]['tuned_precision']), _fmt(test_per_label[row['label_name']]['tuned_recall']), _fmt(test_per_label[row['label_name']]['tuned_f1'])] for row in val_per_label],
    ))
    content.extend(["", "## B1 versus B2 Validation Comparison", ""])
    content.extend(_table(
        ["Metric", "B1 BCE", "B2 sqrt-capped pos_weight", "B2 - B1"],
        [[metric, _fmt(comparison['B1'][metric]), _fmt(comparison['B2'][metric]), _fmt(comparison['B2_minus_B1'][metric])] for metric in ("macro_auroc", "macro_auprc", "macro_f1_fixed05", "macro_f1_tuned", "micro_f1_tuned", "sample_f1_tuned")],
    ))
    content.extend(["", "Both rows use identical saved validation targets and the B2 grid threshold protocol. This comparison is validation-only.", "", "## Rare-5 Validation Comparison", ""])
    rare = _rows(post / "B1_vs_B2_rare5_validation.csv")
    content.extend(_table(
        ["Label", "B1 AUPRC", "B2 AUPRC", "B1 tuned F1", "B2 tuned F1", "B2 - B1 F1"],
        [[row['label_name'], _fmt(row['b1_auprc']), _fmt(row['b2_auprc']), _fmt(row['b1_tuned_f1']), _fmt(row['b2_tuned_f1']), _fmt(row['delta_tuned_f1'])] for row in rare],
    ))
    content.extend(["", "## Hard-label Error Analysis", ""])
    hard_rows = []
    for label in ("Infiltration", "Pneumonia", "Hernia"):
        counts = _count_hard_labels(hard_dir, label)
        hard_rows.append([label, counts['TP'], counts['FP'], counts['FN'], counts['TN']])
    content.extend(_table(["Label", "TP", "FP", "FN", "TN"], hard_rows))
    content.extend(["", "Train-split top conditional co-occurrences:"])
    for label in ("Infiltration", "Pneumonia", "Hernia"):
        pairs = ", ".join(f"{name}={_fmt(prob)}" for name, prob in cooccurrence[label])
        content.append(f"- {label}: {pairs}")
    content.extend([
        "",
        "## Conclusions and B3 Hypothesis",
        "",
        "- B2 improves validation macro AUPRC and fixed-threshold F1 relative to BCE; all B1/B2 model comparisons are validation-only.",
        "- Pneumonia remains the difficult rare class: its frozen-test AUPRC and F1 are the lowest, despite a nontrivial AUROC. This is a precision-recall limitation rather than only a threshold issue.",
        "- Infiltration has high support but relatively low discrimination, so it is not explained by class rarity alone. Hernia provides a counterexample: it is rare but retains high rank discrimination.",
        "- B3 is a pre-specified ASL experiment (gamma_neg=4, gamma_pos=1, clip=0.05) under otherwise B2-matched settings. Its primary validation metric is macro AUPRC, then macro AUROC, then validation-tuned macro F1. B2 test results must not select B3 or its checkpoints.",
        "",
        "## Artifact Index",
        "",
        "- `checkpoint_inventory.csv`, `post_analysis/checkpoint_selection.json`, `post_analysis/training_curves.png`",
        "- `thresholds/y_true_val.npy`, `thresholds/y_prob_val.npy`, `thresholds/threshold_tuning_val.csv`, `thresholds/thresholds_frozen_for_test.json`, `thresholds/threshold_freeze.json`",
        "- `test/test_fixed05_metrics.json`, `test/test_tuned_metrics.json`, `test/test_per_label_comparison.csv`",
        "- `post_analysis/hard_labels/` and `post_analysis/label_cooccurrence.*`",
    ])
    text = "\n".join(content) + "\n"
    report_path = output_dir / "B2_FINAL_REPORT.md"
    if report_path.exists() and report_path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Existing B2 final report differs and will not be overwritten: {report_path}")
    if not report_path.exists():
        report_path.write_text(text, encoding="utf-8", newline="\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate B2 final report from frozen artifacts.")
    parser.add_argument("--output-dir", default="outputs/B2_densenet121_sqrt_posweight")
    args = parser.parse_args()
    print(generate_b2_final_report(args.output_dir))


if __name__ == "__main__":
    main()
