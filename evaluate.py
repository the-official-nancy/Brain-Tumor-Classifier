"""Evaluate a trained model on the test set.

    python evaluate.py --data_dir data --checkpoint outputs/best_model.pt

Produces (in --output_dir, default: outputs/test):
    metrics.json, classification_report.txt, confusion_matrix.png,
    roc_curves.png, misclassified.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from utils import get_device, pretty_name, save_json


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        logits = model(x.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return labels, probs.argmax(axis=1), probs


def _plot_confusion(cm: np.ndarray, classes: list[str], path: Path) -> None:
    names = [pretty_name(c) for c in classes]
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, title, fmt in (
        (axes[0], cm, "Confusion matrix (counts)", "d"),
        (axes[1], cm_norm, "Confusion matrix (row-normalised)", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        thresh = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(labels, probs, classes, path: Path) -> dict[str, float]:
    aucs: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, cls in enumerate(classes):
        y = (labels == i).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y, probs[:, i])
        auc = roc_auc_score(y, probs[:, i])
        aucs[cls] = float(auc)
        ax.plot(fpr, tpr, label=f"{pretty_name(cls)} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return aucs


def evaluate_model(model, loader, device, classes: list[str], out_dir, split_name: str = "test") -> dict:
    """Full evaluation: prints a report and writes metrics + plots to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, preds, probs = collect_predictions(model, loader, device)
    idxs = list(range(len(classes)))

    report = classification_report(labels, preds, labels=idxs, target_names=classes, digits=4, zero_division=0)
    print(f"\n=== {split_name.upper()} SET RESULTS ===")
    print(report)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(labels, preds, labels=idxs)
    _plot_confusion(cm, classes, out_dir / "confusion_matrix.png")
    aucs = _plot_roc(labels, probs, classes, out_dir / "roc_curves.png")

    p, r, f1, support = precision_recall_fscore_support(labels, preds, labels=idxs, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=idxs, average="macro", zero_division=0
    )
    metrics = {
        "split": split_name,
        "num_images": int(len(labels)),
        "accuracy": float((labels == preds).mean()),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "macro_auc_ovr": float(np.mean(list(aucs.values()))) if aucs else None,
        "per_class": {
            cls: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                  "support": int(support[i]), "auc": aucs.get(cls)}
            for i, cls in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
    }

    # Binary "tumor vs no tumor" view - the real-world screening question.
    if "notumor" in classes:
        n_idx = classes.index("notumor")
        y_true = (labels != n_idx).astype(int)
        y_pred = (preds != n_idx).astype(int)
        tumor_score = 1.0 - probs[:, n_idx]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        binary = {
            "sensitivity_recall": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "auc": float(roc_auc_score(y_true, tumor_score)) if 0 < y_true.sum() < len(y_true) else None,
        }
        metrics["binary_tumor_vs_no_tumor"] = binary
        print("--- Binary view: tumor vs. no tumor ---")
        print(f"Sensitivity (tumors caught): {binary['sensitivity_recall']:.4f}   "
              f"Specificity: {binary['specificity']:.4f}   "
              f"Missed tumors (FN): {fn}   False alarms (FP): {fp}")

    # List every misclassified image so you can inspect them.
    samples = loader.dataset.samples  # loader is not shuffled, so order matches
    with open(out_dir / "misclassified.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "predicted_label", "confidence"])
        for (path, _), t, pr, pb in zip(samples, labels, preds, probs):
            if t != pr:
                writer.writerow([path, classes[t], classes[pr], f"{pb[pr]:.4f}"])

    save_json(metrics, out_dir / "metrics.json")
    print(f"Accuracy: {metrics['accuracy']:.4f} | macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Saved metrics + plots to: {out_dir}")
    return metrics


def main() -> None:
    from dataset import build_dataloaders
    from inference import load_checkpoint

    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--output_dir", default=None, help="Default: <checkpoint folder>/test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    saved = ckpt.get("args", {})
    loaders, info = build_dataloaders(
        args.data_dir,
        img_size=ckpt["img_size"],
        batch_size=args.batch_size,
        val_split=saved.get("val_split", 0.15),
        test_split=saved.get("test_split", 0.15),
        num_workers=args.num_workers,
        seed=saved.get("seed", 42),
    )
    if info["classes"] != ckpt["classes"]:
        raise RuntimeError(f"Dataset classes {info['classes']} differ from the model's {ckpt['classes']}.")
    out_dir = args.output_dir or str(Path(args.checkpoint).parent / "test")
    evaluate_model(model, loaders["test"], device, ckpt["classes"], out_dir, "test")


if __name__ == "__main__":
    main()
