"""Load a trained checkpoint and run predictions (with optional Grad-CAM)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import get_transforms
from gradcam import GradCAM, overlay_heatmap
from model import build_model, get_target_layer


def load_checkpoint(path: str | Path, device: torch.device):
    """Return (model in eval mode, checkpoint dict)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint '{path}' not found. Train a model first:  python train.py --data_dir data"
        )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


def preprocess(image: Image.Image, img_size: int) -> torch.Tensor:
    return get_transforms(img_size, train=False)(image.convert("RGB")).unsqueeze(0)


def predict_image(model, ckpt: dict, image: Image.Image, device: torch.device, with_cam: bool = True) -> dict:
    """Classify one PIL image.

    Returns a dict with: label, confidence, probs {class: p}, tumor_probability,
    tumor_detected, original (uint8 RGB), overlay (uint8 RGB or None).
    """
    classes: list[str] = ckpt["classes"]
    img_size: int = ckpt["img_size"]
    x = preprocess(image, img_size).to(device)

    overlay = None
    if with_cam:
        with GradCAM(model, get_target_layer(model, ckpt["model_name"])) as cam_engine:
            cam, probs, idx = cam_engine(x, size=(img_size, img_size))
    else:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        cam = None

    original = np.array(image.convert("RGB").resize((img_size, img_size)))
    if cam is not None:
        overlay = overlay_heatmap(original, cam)

    result = {
        "label": classes[idx],
        "confidence": float(probs[idx]),
        "probs": {c: float(p) for c, p in zip(classes, probs)},
        "original": original,
        "overlay": overlay,
        "tumor_probability": None,
        "tumor_detected": None,
    }
    if "notumor" in classes:
        p_no = float(probs[classes.index("notumor")])
        result["tumor_probability"] = 1.0 - p_no
        result["tumor_detected"] = idx != classes.index("notumor")
    return result
