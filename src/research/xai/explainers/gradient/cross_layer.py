"""
CropXAI Cross-Layer Attribution
=================================

Traces attribution evolution across deep layers to observe 
where specific conceptual representations form.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

try:
    from captum.attr import LayerIntegratedGradients
except ImportError:
    LayerIntegratedGradients = None

from src.research.xai.base.explainer import BaseExplainer

logger = logging.getLogger(__name__)


class CrossLayerExplainer(BaseExplainer):
    """Computes attribution at multiple intermediate layers."""

    def __init__(self, model: nn.Module, layers: list[nn.Module], cfg: Any = None) -> None:
        super().__init__(model, cfg)
        self.layers = layers
        if LayerIntegratedGradients is None:
            raise ImportError("captum not installed.")
            
    def attribute(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor | None = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Returns a dict of attributions keyed by layer index.
        Note: The BaseExplainer interface expects a single Tensor.
        For Cross-Layer, we return the stacked attributions over layers (L, B, ...).
        """
        if target_class is None:
            target_class = 0
            
        class SegmentationWrapper(nn.Module):
            def __init__(self, model: nn.Module, target: int):
                super().__init__()
                self.model = model
                self.target = target
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.model(x)
                if out.dim() > 2:
                    return out[:, self.target, ...].mean(dim=[-1, -2]).unsqueeze(-1)
                return out

        wrapped_model = SegmentationWrapper(self.model, target_class if isinstance(target_class, int) else target_class[0].item())
        
        layer_attributions = []
        for layer in self.layers:
            lig = LayerIntegratedGradients(wrapped_model, layer)
            attr = lig.attribute(inputs, target=0, n_steps=20)
            
            # Need to upsample or pool to match input resolution if we want to return a single stacked tensor.
            # For simplicity, we just return the raw feature attributions. 
            # Real visualizers would interpolate these.
            layer_attributions.append(attr)
            
        # Return a list (or dict) instead of tensor since shapes vary.
        # This breaks BaseExplainer's exact signature but is necessary for CrossLayer.
        return layer_attributions # type: ignore

    @classmethod
    def supports_transformer(cls) -> bool:
        return True

    @classmethod
    def supports_temporal(cls) -> bool:
        return True


__all__ = ["CrossLayerExplainer"]
