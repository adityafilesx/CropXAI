"""
CropXAI Universal Explainability Interface
============================================

Defines the BaseExplainer that every specific XAI method must inherit from.
Guarantees compatibility with (B, T, C, H, W) Foundation Model pipelines
and integrates directly with ExplanationCache.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseExplainer(ABC):
    """
    Abstract Base Class for all XAI Explanation Methods.
    """

    def __init__(self, model: nn.Module, cfg: Any = None) -> None:
        self.model = model
        self.cfg = cfg
        self.model.eval()

    @abstractmethod
    def attribute(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor | None = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Computes the raw attribution scores.
        
        Args:
            inputs: Tensor of shape (B, T, C, H, W) or (B, C, H, W).
            target_class: Target class index to explain. If None, explains the max prediction.
            
        Returns:
            Attribution map of the same spatial/temporal shape as the inputs.
        """
        pass

    def explain(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor | None = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Generates and normalizes the final explanation heatmap.
        """
        attributions = self.attribute(inputs, target_class, **kwargs)
        return self._normalize(attributions)

    def _normalize(self, attributions: torch.Tensor) -> torch.Tensor:
        """
        Normalizes attributions to [0, 1] range per batch item for visualization.
        """
        # Flatten spatial and temporal dims for normalization
        b = attributions.shape[0]
        flat_attr = attributions.view(b, -1)
        
        attr_min = flat_attr.min(dim=1, keepdim=True)[0]
        attr_max = flat_attr.max(dim=1, keepdim=True)[0]
        
        # Avoid division by zero
        denom = attr_max - attr_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        
        norm_attr = (flat_attr - attr_min) / denom
        return norm_attr.view_as(attributions)

    @classmethod
    @abstractmethod
    def supports_transformer(cls) -> bool:
        """Indicates if the explainer supports Vision Transformers."""
        pass

    @classmethod
    @abstractmethod
    def supports_temporal(cls) -> bool:
        """Indicates if the explainer natively handles the T dimension."""
        pass


__all__ = ["BaseExplainer"]
