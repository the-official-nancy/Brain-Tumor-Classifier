"""Dataset discovery, splitting and DataLoader creation.

Two folder layouts are supported automatically:

1. Kaggle "Brain Tumor MRI Dataset" layout (recommended)::

       data/Training/<class>/*.jpg
       data/Testing/<class>/*.jpg

   The Training folder is split (stratified) into train + validation. The Testing
   folder is kept untouched and is only used for the final evaluation.

2. A single folder with one sub-folder per class (e.g. the binary yes/no dataset)::

       data/<class>/*.jpg

   The images are split (stratified) into train / validation / test.

Run ``python dataset.py --data_dir data`` to print a summary and check that all
images can be opened.
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Training transforms add mild augmentation; eval transforms are deterministic."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# Folder scanning
# --------------------------------------------------------------------------- #
def _find_subdir(root: Path, names: tuple[str, ...]) -> Path | None:
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.lower() in names:
            return d
    return None


def locate_train_test(data_dir: str | Path) -> tuple[Path | None, Path | None]:
    """Find Training/ and Testing/ folders in data_dir or one level below it."""
    data_dir = Path(data_dir)
    candidates = [data_dir] + [d for d in sorted(data_dir.iterdir()) if d.is_dir()]
    for base in candidates:
        train = _find_subdir(base, ("training", "train"))
        test = _find_subdir(base, ("testing", "test"))
        if train is not None and test is not None:
            return train, test
    return None, None


def scan_folder(root: str | Path, classes: list[str] | None = None):
    """Return (classes, samples) where samples = [(path, class_index), ...]."""
    root = Path(root)
    if classes is None:
        classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    samples: list[tuple[str, int]] = []
    for idx, cls in enumerate(classes):
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), idx))
    return classes, samples


def stratified_split(samples, fraction: float, seed: int):
    """Split samples into (kept, held_out) keeping class proportions."""
    labels = [label for _, label in samples]
    idx = np.arange(len(samples))
    keep_idx, out_idx = train_test_split(idx, test_size=fraction, stratify=labels, random_state=seed)
    return [samples[i] for i in keep_idx], [samples[i] for i in out_idx]


# --------------------------------------------------------------------------- #
# Dataset + loaders
# --------------------------------------------------------------------------- #
class MRIDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as im:
            img = im.convert("RGB")  # MRI slices are grayscale; replicate to 3 channels
        return self.transform(img), label


def _count(samples, classes) -> dict[str, int]:
    c = Counter(label for _, label in samples)
    return {cls: c.get(i, 0) for i, cls in enumerate(classes)}


def build_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
):
    """Create train / val / test DataLoaders.

    ``val_split`` is the fraction of the training data held out for validation.
    ``test_split`` is only used when the dataset has no Testing/ folder.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory '{data_dir}' does not exist. See README.md -> 'Dataset' "
            "or run: python download_dataset.py"
        )

    train_root, test_root = locate_train_test(data_dir)
    if train_root is not None:
        layout = "Training/Testing folders"
        classes, train_all = scan_folder(train_root)
        _, test_samples = scan_folder(test_root, classes)
        train_samples, val_samples = stratified_split(train_all, val_split, seed)
    else:
        layout = "single folder (random stratified split)"
        classes, all_samples = scan_folder(data_dir)
        rest, test_samples = stratified_split(all_samples, test_split, seed)
        train_samples, val_samples = stratified_split(rest, val_split, seed)

    if len(classes) < 2 or not train_samples or not test_samples:
        raise RuntimeError(
            f"Found classes={classes}, {len(train_samples)} train and {len(test_samples)} test images "
            f"in '{data_dir}'. Expected one sub-folder per class containing images."
        )

    counts = {
        "train": _count(train_samples, classes),
        "val": _count(val_samples, classes),
        "test": _count(test_samples, classes),
    }

    # Inverse-frequency class weights (handles class imbalance in the loss).
    train_counts = np.array([max(1, counts["train"][c]) for c in classes], dtype=np.float64)
    weights = train_counts.sum() / (len(classes) * train_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_ds = MRIDataset(train_samples, get_transforms(img_size, train=True))
    val_ds = MRIDataset(val_samples, get_transforms(img_size, train=False))
    test_ds = MRIDataset(test_samples, get_transforms(img_size, train=False))

    pin = torch.cuda.is_available()
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    loaders = {
        # drop_last avoids a 1-sample final batch, which breaks BatchNorm in train mode
        "train": DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > batch_size, **common),
        "val": DataLoader(val_ds, shuffle=False, **common),
        "test": DataLoader(test_ds, shuffle=False, **common),
    }
    info = {"classes": classes, "counts": counts, "class_weights": class_weights, "layout": layout}
    return loaders, info


# --------------------------------------------------------------------------- #
# CLI: python dataset.py --data_dir data
# --------------------------------------------------------------------------- #
def _main() -> None:
    parser = argparse.ArgumentParser(description="Print a dataset summary and verify images open correctly.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--no_verify", action="store_true", help="Skip opening every image.")
    args = parser.parse_args()

    loaders, info = build_dataloaders(args.data_dir, num_workers=0)
    print(f"Layout : {info['layout']}")
    print(f"Classes: {info['classes']}\n")
    header = f"{'class':<14}" + "".join(f"{s:>10}" for s in ("train", "val", "test"))
    print(header)
    print("-" * len(header))
    for cls in info["classes"]:
        print(f"{cls:<14}" + "".join(f"{info['counts'][s][cls]:>10}" for s in ("train", "val", "test")))
    totals = {s: sum(info["counts"][s].values()) for s in ("train", "val", "test")}
    print("-" * len(header))
    print(f"{'total':<14}" + "".join(f"{totals[s]:>10}" for s in ("train", "val", "test")))
    print("\nClass weights used in the loss:", [round(float(w), 3) for w in info["class_weights"]])

    if not args.no_verify:
        bad = []
        all_samples = [s for split in ("train", "val", "test") for s in loaders[split].dataset.samples]
        for path, _ in all_samples:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception as exc:  # noqa: BLE001
                bad.append((path, str(exc)))
        if bad:
            print(f"\n{len(bad)} unreadable image(s):")
            for p, e in bad[:20]:
                print(f"  {p}: {e}")
        else:
            print(f"\nAll {len(all_samples)} images opened successfully.")


if __name__ == "__main__":
    _main()
