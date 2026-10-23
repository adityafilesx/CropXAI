"""
CropXAI Output Adapter
========================

Normalizes outputs from various backbones (tokens, hierarchical feature maps)
into standard 2D feature maps (B, C, H, W) for Decoders and Segmentation Heads.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class OutputAdapter(nn.Module):
    """
    Transforms backbone outputs into standard format for Decoders.
    """

    def __init__(self) -> None:
        super().__init__()

    def tokens_to_feature_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Reshape a sequence of ViT tokens (B, N, C) into a feature map (B, C, H, W).
        Assumes N = H * W.
        """
        b, n, c = tokens.shape
        if n != h * w:
            # Handle CLS token if present
            if n == h * w + 1:
                tokens = tokens[:, 1:, :]
            else:
                raise ValueError(f"Cannot reshape {n} tokens to {h}x{w}.")
                
        # (B, H*W, C) -> (B, C, H, W)
        x = tokens.permute(0, 2, 1).contiguous()
        x = x.view(b, c, h, w)
        return x

    def forward(self, features: Any) -> list[torch.Tensor]:
        """
        By default, we assume the backbone's `extract_multiscale` already 
        handles the heavy lifting. This module is a passthrough for CNNs,
        but can be subclassed for complex token-based outputs.
        """
        if isinstance(features, torch.Tensor):
            return [features]
        return features


__all__ = ["OutputAdapter"]
