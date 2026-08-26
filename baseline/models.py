from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Build a torchvision classifier whose final layer emits one logit per label."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {name}. Choose resnet18 or densenet121.")
    return model
