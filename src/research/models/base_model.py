"""
CropXAI Model Base Interface
==============================

Abstract base for all model implementations.
All Phase 2/3 models (CNNs, ViTs, VFMs) must subclass this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all CropXAI models.

    Enforces a consistent interface for:
    - Forward pass
    - Input specification (bands, spatial resolution, temporal mode)
    - Summary and parameter counting

    All concrete models should be decorated with ``@MODEL_REGISTRY.register()``.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes: int = cfg.dataset.labels.num_classes
        self.in_channels: int = cfg.dataset.bands.num_bands
        self.model_name: str = cfg.model.name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor. Shape: (B, C, H, W) or (B, T, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits. Shape: (B, num_classes, H, W) for segmentation.
        """
        ...

    def get_intermediate_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Return intermediate feature maps for XAI methods.

        Override in subclasses to expose specific layers.

        Returns
        -------
        dict[str, torch.Tensor]
            Named intermediate activations.
        """
        return {}

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return number of model parameters."""
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)

    def model_summary(self) -> dict[str, Any]:
        """Return a summary dict for logging."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "total_parameters": self.count_parameters(trainable_only=False),
            "trainable_parameters": self.count_parameters(trainable_only=True),
        }

    def __repr__(self) -> str:
        summary = self.model_summary()
        return (
            f"{self.__class__.__name__}("
            f"name={summary['model_name']}, "
            f"classes={summary['num_classes']}, "
            f"params={summary['trainable_parameters']:,})"
        )


__all__ = ["BaseModel"]
