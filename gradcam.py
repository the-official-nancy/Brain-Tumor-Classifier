"""Grad-CAM: highlights the image regions that drove the model's decision.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization" (ICCV 2017). Implemented from scratch - no extra packages.

IMPORTANT: the heat-map shows what the *model* looked at. It is NOT a tumor
segmentation and must not be treated as a precise tumor location.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        # A tensor hook (instead of a module backward hook) is robust to in-place ReLUs.
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None, size: tuple[int, int] | None = None):
        """x: (1, C, H, W). Returns (cam [H,W] in 0..1, probabilities [num_classes], class_idx)."""
        if x.shape[0] != 1:
            raise ValueError("Grad-CAM expects a batch of exactly one image.")
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            logits[0, class_idx].backward()

        acts = self.activations[0]          # (C, h, w)
        grads = self.gradients[0]           # (C, h, w)
        weights = grads.mean(dim=(1, 2))    # global-average-pooled gradients
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        peak = cam.max()
        if peak > 0:  # scale-safe normalisation (no fixed epsilon)
            cam = cam / peak

        size = size or tuple(x.shape[-2:])
        cam = F.interpolate(cam[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
        return cam.cpu().numpy(), probs.detach().cpu().numpy(), class_idx


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heat-map over an RGB uint8 image of the same height/width."""
    cmap = matplotlib.colormaps["jet"]
    heat = (cmap(np.clip(cam, 0, 1))[..., :3] * 255).astype(np.float32)
    blended = (1 - alpha) * rgb.astype(np.float32) + alpha * heat
    return np.clip(blended, 0, 255).astype(np.uint8)
