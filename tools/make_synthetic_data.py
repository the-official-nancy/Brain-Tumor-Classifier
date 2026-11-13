"""Generate a tiny FAKE MRI-like dataset to test that the code runs end to end.

    python tools/make_synthetic_data.py --out synthetic_data

These are cartoon images (grey ellipse "heads" with bright blobs). They are ONLY for checking
that your installation works - a model trained on them says nothing about real tumors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def make_image(cls: str, rng: np.random.Generator, size: int = 256) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = size / 2 + rng.normal(0, 4)
    cy = size / 2 + rng.normal(0, 4)
    head = (((xx - cx) / (size * 0.38)) ** 2 + ((yy - cy) / (size * 0.46)) ** 2) < 1
    img = np.zeros((size, size), np.float32)
    img[head] = 0.45 + 0.1 * rng.random()
    img += rng.normal(0, 0.05, (size, size)) * head

    def blob(bx, by, radius, intensity):
        mask = ((xx - bx) ** 2 + (yy - by) ** 2) < radius**2
        img[mask & head] += intensity

    if cls == "glioma":        # large, bright, inside the brain
        blob(cx + rng.uniform(-50, 50), cy + rng.uniform(-50, 20), rng.uniform(18, 32), 0.35)
    elif cls == "meningioma":  # medium blob near the skull edge
        blob(cx + rng.choice([-1, 1]) * rng.uniform(55, 70), cy + rng.uniform(-30, 30), rng.uniform(14, 22), 0.45)
    elif cls == "pituitary":   # small blob low and central
        blob(cx + rng.uniform(-6, 6), cy + rng.uniform(25, 40), rng.uniform(8, 12), 0.5)

    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="synthetic_data")
    parser.add_argument("--train_per_class", type=int, default=40)
    parser.add_argument("--test_per_class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for split, n in (("Training", args.train_per_class), ("Testing", args.test_per_class)):
        for cls in CLASSES:
            folder = Path(args.out) / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                make_image(cls, rng).save(folder / f"{cls}_{i:03d}.jpg", quality=92)
    print(f"Synthetic dataset written to '{args.out}' (for pipeline testing only).")


if __name__ == "__main__":
    main()
