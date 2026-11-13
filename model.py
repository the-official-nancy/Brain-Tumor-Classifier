"""Model definitions.

Available architectures (``--model``):

* ``resnet18``        - fast, light, a strong default (11M params, ImageNet pre-trained)
* ``resnet50``        - bigger and usually a little more accurate (25M params)
* ``efficientnet_b0`` - accurate and compact (5M params)
* ``simple_cnn``      - a small CNN trained from scratch (for learning / comparison)
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED = ("resnet18", "resnet50", "efficientnet_b0", "simple_cnn")

# Name of the classification-head attribute on each architecture (used for freezing).
HEAD_ATTR = {
    "resnet18": "fc",
    "resnet50": "fc",
    "efficientnet_b0": "classifier",
    "simple_cnn": "classifier",
}


class SimpleCNN(nn.Module):
    """Four conv blocks + global average pooling. Trains from scratch."""

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=False),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def _make_torchvision(ctor, weights_enum, pretrained: bool):
    """Build a torchvision model; fall back to random init if weights can't be downloaded."""
    if pretrained:
        try:
            return ctor(weights=weights_enum.DEFAULT)
        except Exception as exc:  # noqa: BLE001 - network / cache problems
            print(
                f"[warning] Could not load pre-trained weights ({exc}).\n"
                "          Falling back to random initialisation - expect lower accuracy.\n"
                "          Check your internet connection or use --no_pretrained to silence this."
            )
    return ctor(weights=None)


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    name = name.lower()
    if name not in SUPPORTED:
        raise ValueError(f"Unknown model '{name}'. Choose from {SUPPORTED}.")

    if name == "simple_cnn":
        return SimpleCNN(num_classes)

    if name == "resnet18":
        model = _make_torchvision(models.resnet18, models.ResNet18_Weights, pretrained)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "resnet50":
        model = _make_torchvision(models.resnet50, models.ResNet50_Weights, pretrained)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    else:  # efficientnet_b0
        model = _make_torchvision(models.efficientnet_b0, models.EfficientNet_B0_Weights, pretrained)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return model


def get_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional stage - the layer Grad-CAM explains."""
    name = name.lower()
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    return model.features[-1]  # simple_cnn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
