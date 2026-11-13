"""Build a small, balanced subset of the Brain Tumor MRI dataset that fits an upload limit.

Run it on the dataset you downloaded and unzipped from Kaggle:

    python tools/make_small_subset.py --src path/to/unzipped_dataset --out small_subset.zip --max_mb 28

What it does
  * finds the Training/ and Testing/ folders inside --src
  * shrinks every picked image to 224x224 greyscale JPEG (the model resizes to 224x224 anyway,
    so almost nothing the model uses is lost)
  * adds images round-robin across the classes (so the classes stay balanced) until the size
    budget is used: 80% of the budget for Training/, 20% for Testing/
  * writes a zip that has the same layout as the original (Training/<class>/..., Testing/<class>/...)
"""
from __future__ import annotations

import argparse
import io
import random
import zipfile
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_split_dirs(src: Path):
    """Return {'Training': path, 'Testing': path} searching src and one level below."""
    for base in [src] + [d for d in sorted(src.iterdir()) if d.is_dir()]:
        found = {}
        for d in base.iterdir():
            if d.is_dir() and d.name.lower() in ("training", "train"):
                found["Training"] = d
            elif d.is_dir() and d.name.lower() in ("testing", "test"):
                found["Testing"] = d
        if len(found) == 2:
            return found
    raise SystemExit(f"Could not find Training/ and Testing/ folders inside '{src}'.")


def shrink(path: Path, size: int, quality: int) -> bytes:
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def fill_split(zf: zipfile.ZipFile, split_name: str, split_dir: Path, budget: int, size: int,
               quality: int, max_per_class: int | None, rng: random.Random) -> dict[str, int]:
    classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    files = {}
    for c in classes:
        imgs = [p for p in (split_dir / c).rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        rng.shuffle(imgs)
        files[c] = imgs[:max_per_class] if max_per_class else imgs

    counts = {c: 0 for c in classes}
    used = 0
    active = list(classes)
    pos = {c: 0 for c in classes}
    while active:
        for c in list(active):
            if pos[c] >= len(files[c]):
                active.remove(c)
                continue
            src_path = files[c][pos[c]]
            pos[c] += 1
            try:
                data = shrink(src_path, size, quality)
            except Exception:  # unreadable image: skip it
                continue
            if used + len(data) > budget:
                return counts  # budget reached; stop (round-robin keeps classes balanced)
            zf.writestr(f"{split_name}/{c}/{c}_{counts[c]:04d}.jpg", data, compress_type=zipfile.ZIP_STORED)
            used += len(data)
            counts[c] += 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser(description="Make a small balanced subset zip of the MRI dataset.")
    p.add_argument("--src", required=True, help="Folder of the unzipped Kaggle dataset.")
    p.add_argument("--out", default="small_subset.zip")
    p.add_argument("--max_mb", type=float, default=28.0, help="Target zip size in MB (default 28).")
    p.add_argument("--size", type=int, default=224, help="Image side length in pixels.")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality 1-95.")
    p.add_argument("--max_per_class", type=int, default=None, help="Optional cap per class and split.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    splits = find_split_dirs(Path(args.src))
    budget_total = int(args.max_mb * 1024 * 1024) - 200_000  # leave room for zip headers
    budgets = {"Training": int(budget_total * 0.8), "Testing": int(budget_total * 0.2)}
    rng = random.Random(args.seed)

    with zipfile.ZipFile(args.out, "w") as zf:
        for name in ("Training", "Testing"):
            counts = fill_split(zf, name, splits[name], budgets[name], args.size, args.quality,
                                args.max_per_class, rng)
            print(f"{name:<9} {sum(counts.values()):>5} images  {counts}")

    size_mb = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"\nWrote {args.out}  ({size_mb:.1f} MB)")
    if size_mb > args.max_mb:
        print("Warning: larger than requested; lower --max_mb or --quality.")


if __name__ == "__main__":
    main()
