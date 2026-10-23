"""
CropXAI Fusion Mechanisms
===========================

Handles fusing multi-scale feature maps or multimodal inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.core.registry import Registry

FUSION_REGISTRY = Registry("fusion")


class BaseFusion(nn.Module):
    """Base interface for feature fusion."""
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


@FUSION_REGISTRY.register("concatenation")
class ConcatenationFusion(BaseFusion):
    """Concatenates features along the channel dimension."""
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Assuming spatial dimensions are identical
        return torch.cat(features, dim=1)


@FUSION_REGISTRY.register("addition")
class AdditionFusion(BaseFusion):
    """Adds features element-wise."""
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        res = features[0]
        for f in features[1:]:
            res = res + f
        return res


@FUSION_REGISTRY.register("weighted")
class WeightedFusion(BaseFusion):
    """Learns scalar weights for each feature map before addition."""
    def __init__(self, num_inputs: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        weights = torch.softmax(self.weights, dim=0)
        res = features[0] * weights[0]
        for i, f in enumerate(features[1:], 1):
            res = res + f * weights[i]
        return res


@FUSION_REGISTRY.register("attention")
class AttentionFusion(BaseFusion):
    """
    Basic Channel Attention Fusion (SE block style) across stacked features.
    """
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.cat(features, dim=1)
        attention = self.se(stacked)
        return stacked * attention


__all__ = ["FUSION_REGISTRY", "ConcatenationFusion", "AdditionFusion", "WeightedFusion", "AttentionFusion"]
