"""
CropXAI Input Adapter
=======================

Decouples the universal dataset format (B, T, C, H, W) from 
what the specific backbone expects (e.g., 2D CNNs expect (B, C, H, W)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from src.research.models.interfaces.vision_backbone import ModelCapability


class InputAdapter(nn.Module):
    """
    Transforms universal (B, T, C, H, W) inputs into model-specific formats.
    """

    def __init__(self, capability: ModelCapability) -> None:
        super().__init__()
        self.capability = capability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is expected to be (B, T, C, H, W).
        """
        # If backbone doesn't support temporal but T=1, squeeze it.
        # If T>1 and backbone doesn't support temporal, raise error or merge.
        if not self.capability.temporal:
            if x.dim() == 5:
                if x.shape[1] == 1:
                    # (B, 1, C, H, W) -> (B, C, H, W)
                    return x.squeeze(1)
                else:
                    raise ValueError(
                        f"Backbone does not support temporal inputs, "
                        f"but received input with T={x.shape[1]} > 1."
                    )
        
        # If backbone expects temporal, pass as is (or reshape based on backbone specific requirements)
        # Note: Foundation adapters might need a specific reshape (e.g. B, C, T, H, W),
        # but that would be handled inside the Foundation Adapter itself.
        # This InputAdapter just ensures the base shape is compatible.
        return x


__all__ = ["InputAdapter"]
