"""Small helpers shared across the project."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend: we only ever save figures to disk
import matplotlib.pyplot as plt
import numpy as np
import torch

PRETTY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}


def pretty_name(class_name: str) -> str:
    """Human-friendly class name for display."""
    return PRETTY_NAMES.get(class_name, class_name.replace("_", " ").title())


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Pick cuda -> mps -> cpu automatically unless the user forces a device."""
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_history(history: dict, path) -> None:
    """Save loss / accuracy / F1 curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_title("Accuracy")

    axes[2].plot(epochs, history["val_f1"], label="validation macro-F1", color="tab:green")
    axes[2].set_title("Validation macro-F1")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
