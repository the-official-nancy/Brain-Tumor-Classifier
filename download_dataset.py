"""Download the Kaggle "Brain Tumor MRI Dataset" into ./data

    pip install kagglehub
    python download_dataset.py

Kaggle may ask for credentials. Create a free account, then either run `kaggle auth login`-style
setup as described at https://www.kaggle.com/docs/api, or put your API token in
~/.kaggle/kaggle.json (or set KAGGLE_USERNAME and KAGGLE_KEY).

If this script fails, download the ZIP manually (see README.md -> Dataset) and unzip it so that
you end up with  data/Training/<class>/  and  data/Testing/<class>/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

KAGGLE_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"
PAGE = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"

MANUAL = f"""
Manual download:
  1. Open {PAGE}
  2. Click "Download" (sign in with a free Kaggle account) and unzip the file.
  3. Move the folders so you have:
         {Path('data') / 'Training'}/glioma, meningioma, notumor, pituitary
         {Path('data') / 'Testing'}/glioma, meningioma, notumor, pituitary
  4. Check it:  python dataset.py --data_dir data
"""


def find_dataset_root(path: Path) -> Path | None:
    """Locate the folder that contains both Training/ and Testing/."""
    for base in [path] + [d for d in path.rglob("*") if d.is_dir()]:
        names = {d.name.lower() for d in base.iterdir() if d.is_dir()}
        if {"training", "testing"} <= names:
            return base
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Brain Tumor MRI dataset from Kaggle.")
    parser.add_argument("--out", default="data", help="Destination folder (default: data)")
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run:  pip install kagglehub")
        print(MANUAL)
        sys.exit(1)

    try:
        print(f"Downloading {KAGGLE_SLUG} ... (about 150 MB)")
        cache_path = Path(kagglehub.dataset_download(KAGGLE_SLUG))
        root = find_dataset_root(cache_path)
        if root is None:
            raise RuntimeError(f"Could not find Training/ and Testing/ inside {cache_path}")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for d in root.iterdir():
            if d.is_dir() and d.name.lower() in ("training", "testing"):
                shutil.copytree(d, out / d.name, dirs_exist_ok=True)
        print(f"Dataset ready in '{out}'. Verify with:  python dataset.py --data_dir {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nAutomatic download failed: {exc}")
        print(MANUAL)
        sys.exit(1)


if __name__ == "__main__":
    main()
