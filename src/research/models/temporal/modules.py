"""
CropXAI Temporal Modules
==========================

Basic temporal aggregation functions to reduce (B, T, C, H, W) to (B, C, H, W).
Used primarily when a model expects 2D inputs but the dataset provides temporal sequences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from src.core.registry import Registry

TEMPORAL_REGISTRY = Registry("temporal")


class TemporalInterface(nn.Module, ABC):
    """Base interface for all temporal aggregation modules."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the temporal dimension.
        Input: (B, T, C, H, W)
        Output: (B, C, H, W)
        """
        ...


@TEMPORAL_REGISTRY.register("identity")
class IdentityTemporal(TemporalInterface):
    """
    Assumes T=1 and simply squeezes the temporal dimension.
    Raises an error if T > 1.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > 1:
            raise ValueError(f"IdentityTemporal expects T=1, got T={x.shape[1]}")
        return x.squeeze(1)


@TEMPORAL_REGISTRY.register("average")
class AverageTemporal(TemporalInterface):
    """Averages features across the temporal dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


@TEMPORAL_REGISTRY.register("max")
class MaxTemporal(TemporalInterface):
    """Takes the maximum feature across the temporal dimension."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1)[0]


__all__ = ["TEMPORAL_REGISTRY", "TemporalInterface", "IdentityTemporal", "AverageTemporal", "MaxTemporal"]
