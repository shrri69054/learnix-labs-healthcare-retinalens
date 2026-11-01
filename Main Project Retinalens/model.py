"""EfficientNet-B0 transfer-learning classifier."""
import torch.nn as nn
from torchvision import models


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, 1))  # single logit: P(RG)
    return m
