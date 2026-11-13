"""Classify one MRI image (or a whole folder) with a trained model.

    python predict.py --image path/to/mri.jpg
    python predict.py --image path/to/folder_of_images --save_dir outputs/predictions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dataset import IMG_EXTS
from inference import load_checkpoint, predict_image
from utils import get_device, pretty_name

DISCLAIMER = "Research/educational tool - NOT a medical device. Not for diagnosis."


def save_figure(result: dict, title: str, path: Path) -> None:
    has_cam = result["overlay"] is not None
    fig, axes = plt.subplots(1, 2 if has_cam else 1, figsize=(10 if has_cam else 5, 5.2), squeeze=False)
    axes[0][0].imshow(result["original"])
    axes[0][0].set_title("MRI input")
    if has_cam:
        axes[0][1].imshow(result["overlay"])
        axes[0][1].set_title("Grad-CAM (model attention)")
    for ax in axes[0]:
        ax.axis("off")
    verdict = f"{pretty_name(result['label'])} ({result['confidence'] * 100:.1f}%)"
    fig.suptitle(f"{title}\nPrediction: {verdict}", fontsize=11)
    fig.text(0.5, 0.01, DISCLAIMER, ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict tumor class for MRI image(s).")
    parser.add_argument("--image", required=True, help="Path to an image file or a folder of images.")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--save_dir", default="outputs/predictions")
    parser.add_argument("--no_cam", action="store_true", help="Skip the Grad-CAM heat-map.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)

    src = Path(args.image)
    if src.is_dir():
        paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    elif src.is_file():
        paths = [src]
    else:
        raise FileNotFoundError(f"'{src}' does not exist.")
    if not paths:
        raise RuntimeError(f"No images found in '{src}'.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"{DISCLAIMER}\n")

    for path in paths:
        image = Image.open(path)
        result = predict_image(model, ckpt, image, device, with_cam=not args.no_cam)
        print(f"{path.name}")
        print(f"  Prediction : {pretty_name(result['label'])}  ({result['confidence'] * 100:.1f}% confidence)")
        if result["tumor_probability"] is not None:
            print(f"  Tumor prob.: {result['tumor_probability'] * 100:.1f}%")
        for cls, p in sorted(result["probs"].items(), key=lambda kv: -kv[1]):
            print(f"    {pretty_name(cls):<16}{p * 100:6.2f}%")
        out = save_dir / f"{path.stem}_prediction.png"
        save_figure(result, path.name, out)
        print(f"  Saved      : {out}\n")


if __name__ == "__main__":
    main()
