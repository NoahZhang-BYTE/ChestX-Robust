"""Generate publication-ready aggregate analysis for the frozen B4+B5 ensemble.

The script consumes only frozen validation/test arrays and the validation-derived
threshold artifact. It never refits thresholds or uses test metrics for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def _validate_inputs(targets: np.ndarray, probabilities: np.ndarray, thresholds: Sequence[float], labels: Sequence[str]) -> None:
    if targets.ndim != 2 or probabilities.shape != targets.shape:
        raise ValueError("targets and probabilities must be 2D arrays with matching shapes")
    if targets.shape[1] != len(labels) or len(thresholds) != len(labels):
        raise ValueError("one threshold and label are required per class")
    if not np.isin(targets, (0, 1)).all():
        raise ValueError("targets must contain only 0 and 1")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite values between 0 and 1")


def confusion_counts(targets: np.ndarray, probabilities: np.ndarray, thresholds: Sequence[float]) -> list[dict[str, int]]:
    """Return TP/FP/TN/FN counts in label order at frozen thresholds."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    thresholds = np.asarray(thresholds, dtype=float)
    if targets.ndim != 2 or probabilities.shape != targets.shape or thresholds.size != targets.shape[1]:
        raise ValueError("targets, probabilities, and thresholds have incompatible shapes")
    predicted = probabilities >= thresholds.reshape(1, -1)
    rows: list[dict[str, int]] = []
    for index in range(targets.shape[1]):
        actual = targets[:, index].astype(bool)
        guess = predicted[:, index]
        rows.append({
            "tp": int(np.logical_and(actual, guess).sum()),
            "fp": int(np.logical_and(~actual, guess).sum()),
            "tn": int(np.logical_and(~actual, ~guess).sum()),
            "fn": int(np.logical_and(actual, ~guess).sum()),
        })
    return rows


def top_error_rows(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float],
    labels: Sequence[str],
    top_n: int = 25,
    ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Return highest-confidence FP/FN records, preserving sample identifiers when supplied."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    thresholds = np.asarray(thresholds, dtype=float)
    _validate_inputs(targets, probabilities, thresholds, labels)
    if ids is not None and len(ids) != targets.shape[0]:
        raise ValueError("ids must have one value per sample")
    predicted = probabilities >= thresholds.reshape(1, -1)
    rows: list[dict[str, object]] = []
    for class_index, label in enumerate(labels):
        actual = targets[:, class_index].astype(bool)
        guess = predicted[:, class_index]
        fp = np.flatnonzero(~actual & guess)
        fn = np.flatnonzero(actual & ~guess)
        for error_type, indices in (("FP", fp), ("FN", fn)):
            ranked = indices[np.argsort(probabilities[indices, class_index])[::-1]][:top_n]
            for sample_index in ranked:
                rows.append({
                    "label": str(label),
                    "error_type": error_type,
                    "sample_index": int(sample_index),
                    "sample_id": str(ids[sample_index]) if ids is not None else str(sample_index),
                    "score": float(probabilities[sample_index, class_index]),
                    "threshold": float(thresholds[class_index]),
                })
    # Confidence is the useful ordering for manual review; FP/FN remains explicit.
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    return rows


def _metrics_table(targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    counts = confusion_counts(targets, probabilities, thresholds)
    for i, label in enumerate(labels):
        target = targets[:, i]
        prob = probabilities[:, i]
        pred = prob >= thresholds[i]
        rows.append({
            "label": label,
            "support": int(target.sum()),
            "prevalence": float(target.mean()),
            "threshold": float(thresholds[i]),
            "auroc": float(roc_auc_score(target, prob)) if np.unique(target).size == 2 else np.nan,
            "auprc": float(average_precision_score(target, prob)) if np.unique(target).size == 2 else np.nan,
            "tp": counts[i]["tp"], "fp": counts[i]["fp"], "tn": counts[i]["tn"], "fn": counts[i]["fn"],
            "precision": float(counts[i]["tp"] / (counts[i]["tp"] + counts[i]["fp"])) if counts[i]["tp"] + counts[i]["fp"] else 0.0,
            "recall": float(counts[i]["tp"] / (counts[i]["tp"] + counts[i]["fn"])) if counts[i]["tp"] + counts[i]["fn"] else 0.0,
        })
    table = pd.DataFrame(rows)
    table["f1"] = np.where(
        table["precision"] + table["recall"] > 0,
        2 * table["precision"] * table["recall"] / (table["precision"] + table["recall"]),
        0.0,
    )
    return table


def _save_figure(fig: Any, path: Path) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    # Matplotlib may still emit an alpha channel; normalize to an opaque RGB deliverable.
    with Image.open(png_path) as image:
        image.convert("RGB").save(png_path, dpi=(300, 300), optimize=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_roc_pr(targets: np.ndarray, probabilities: np.ndarray, labels: Sequence[str], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for i, (label, color) in enumerate(zip(labels, colors)):
        if np.unique(targets[:, i]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(targets[:, i], probabilities[:, i])
        precision, recall, _ = precision_recall_curve(targets[:, i], probabilities[:, i])
        auc = roc_auc_score(targets[:, i], probabilities[:, i])
        ap = average_precision_score(targets[:, i], probabilities[:, i])
        axes[0].plot(fpr, tpr, lw=1.5, color=color, label=f"{label} ({auc:.3f})")
        axes[1].plot(recall, precision, lw=1.5, color=color, label=f"{label} ({ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#777777", lw=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test ROC curves")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curves")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    _save_figure(fig, output)


def plot_confusions(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = table[["tp", "fp", "tn", "fn"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(np.log1p(values), cmap="viridis", aspect="auto")
    ax.set_xticks(range(4), ["TP", "FP", "TN", "FN"])
    ax.set_yticks(range(len(table)), table["label"])
    ax.set_title("Test confusion counts at frozen validation thresholds\n(color scale: log1p count)")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, str(int(values[row, col])), ha="center", va="center", fontsize=8,
                    color="white" if np.log1p(values[row, col]) > np.log1p(values.max()) * 0.55 else "black")
    fig.colorbar(image, ax=ax, label="log1p(count)")
    _save_figure(fig, output)


def plot_prevalence(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(table["prevalence"], table["f1"], c=table["auprc"], s=55, cmap="viridis", edgecolor="black", linewidth=0.3)
    for _, row in table.iterrows():
        ax.annotate(row["label"], (row["prevalence"], row["f1"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", xlabel="Test prevalence (log scale)", ylabel="F1 at frozen threshold", title="Long-tail prevalence and thresholded performance")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="AUPRC")
    _save_figure(fig, output)


def plot_per_class_performance(table: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = table.sort_values("auroc", ascending=True)
    y = np.arange(len(ordered))
    height = 0.24
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.barh(y - height, ordered["auroc"], height, label="AUROC", color="#0072B2")
    ax.barh(y, ordered["auprc"], height, label="AUPRC", color="#009E73")
    ax.barh(y + height, ordered["f1"], height, label="F1 (frozen threshold)", color="#D55E00")
    ax.set(yticks=y, yticklabels=ordered["label"], xlim=(0, 1), xlabel="Score", title="Per-class test performance")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    _save_figure(fig, output)


def run_analysis(artifact_dir: Path, output_dir: Path, labels: Sequence[str] = LABELS, top_n: int = 25) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = np.load(artifact_dir / "test_targets.npy")
    probabilities = np.load(artifact_dir / "test_probabilities.npy")
    threshold_artifact = json.loads((artifact_dir / "thresholds.json").read_text(encoding="utf-8"))
    thresholds = np.asarray(threshold_artifact["thresholds"], dtype=float)
    _validate_inputs(targets, probabilities, thresholds, labels)

    table = _metrics_table(targets, probabilities, thresholds, labels)
    table.to_csv(output_dir / "test_metrics_with_confusion.csv", index=False, float_format="%.8f")
    pd.DataFrame(confusion_counts(targets, probabilities, thresholds), index=labels).to_csv(output_dir / "test_confusion_counts.csv")

    frame = pd.read_csv(Path("data") / "labels.csv")
    test_frame = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    csv_targets = test_frame.loc[:, list(labels)].to_numpy(dtype=np.int64) if len(test_frame) == len(targets) else None
    ids = (
        test_frame["image_id"].astype(str).tolist()
        if csv_targets is not None and np.array_equal(csv_targets, targets)
        else None
    )
    errors = top_error_rows(targets, probabilities, thresholds, labels, top_n=top_n, ids=ids)
    pd.DataFrame(errors).to_csv(output_dir / "top_error_cases.csv", index=False)

    plot_roc_pr(targets, probabilities, labels, output_dir / "test_roc_pr")
    plot_confusions(table, output_dir / "test_confusion_summary")
    plot_prevalence(table, output_dir / "prevalence_vs_f1_auprc")
    plot_per_class_performance(table, output_dir / "per_class_performance")

    summary = {
        "source": str(artifact_dir),
        "threshold_source": "validation_only",
        "test_used_for_selection": False,
        "n_test": int(len(targets)),
        "macro_auroc": float(table["auroc"].mean()),
        "macro_auprc": float(table["auprc"].mean()),
        "lowest_auroc": str(table.loc[table["auroc"].idxmin(), "label"]),
        "lowest_f1": str(table.loc[table["f1"].idxmin(), "label"]),
        "image_ids_recovered": ids is not None,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    micro_auroc = float(roc_auc_score(targets.ravel(), probabilities.ravel()))
    micro_auprc = float(average_precision_score(targets.ravel(), probabilities.ravel()))
    report = (
        "# Frozen ensemble final analysis\n\n"
        f"Test set contains {len(targets):,} images. The frozen candidate is evaluated once with "
        "validation-only per-class thresholds; no test value is used for selection or fitting.\n\n"
        "## Ranking metrics\n\n"
        f"- Macro AUROC: **{table['auroc'].mean():.4f}**; micro AUROC: **{micro_auroc:.4f}**\n"
        f"- Macro AUPRC: **{table['auprc'].mean():.4f}**; micro AUPRC: **{micro_auprc:.4f}**\n\n"
        "## Thresholded metrics\n\n"
        f"- Macro F1: **{table['f1'].mean():.4f}** at frozen validation thresholds\n"
        f"- Lowest AUROC: **{summary['lowest_auroc']}**; lowest F1: **{summary['lowest_f1']}**\n\n"
        "The plots are descriptive: AUROC/AUPRC assess ranking, while confusion counts and F1 "
        "depend on the frozen decision thresholds. Rare-label precision remains the main limitation.\n"
    )
    (output_dir / "final_analysis_report.md").write_text(report, encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Final analysis artifacts\n\n"
        "All plots and tables use the frozen B4+B5 ensemble test probabilities and validation-only thresholds. "
        "ROC/AUPRC are threshold-independent; confusion counts and F1 use the frozen per-class thresholds. "
        "The error table ranks false positives and false negatives by confidence for manual review.\n",
        encoding="utf-8",
    )
    manifest = {
        "source_artifact_dir": str(artifact_dir),
        "source_files": {
            name: {"path": str(artifact_dir / name), "sha256": hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest()}
            for name in ("test_targets.npy", "test_probabilities.npy", "thresholds.json")
        },
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "transformations": ["sigmoid probabilities already persisted", "validation-only frozen per-class thresholds", "no smoothing or interpolation"],
        "image_ids": "labels.csv test rows included only when their label matrix exactly matches test_targets.npy",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/B4B5_ensemble_20260904_1300"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/B4B5_ensemble_20260904_1300/final_analysis"))
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()
    run_analysis(args.artifact_dir, args.output_dir, top_n=args.top_n)


if __name__ == "__main__":
    main()
