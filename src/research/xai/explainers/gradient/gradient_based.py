"""
CropXAI Gradient-Based Explainers
===================================

Wraps Captum's gradient-based attribution methods.
Supports Integrated Gradients, SmoothGrad, Expected Gradients.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

try:
    from captum.attr import IntegratedGradients, NoiseTunnel, GuidedBackprop
except ImportError:
    IntegratedGradients = None
    NoiseTunnel = None
    GuidedBackprop = None

from src.research.xai.base.explainer import BaseExplainer

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer(BaseExplainer):
    """Wrapper for Captum's Integrated Gradients."""

    def __init__(self, model: nn.Module, cfg: Any = None) -> None:
        super().__init__(model, cfg)
        if IntegratedGradients is None:
            raise ImportError("Please install captum to use IntegratedGradientsExplainer.")
        self.ig = IntegratedGradients(self.model)
        self.steps = getattr(self.cfg, "n_steps", 50) if self.cfg else 50

    def attribute(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor | None = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        """Computes IG attributions."""
        inputs.requires_grad_()
        
        # Determine target if not provided
        if target_class is None:
            with torch.no_grad():
                preds = self.model(inputs)
                if preds.dim() > 2:
                    # Semantic segmentation: (B, C, H, W) or (B, C, T, H, W)
                    # We might need to reduce to a specific pixel or sum across spatial dims.
                    # Captum typically requires target to be an index for scalar outputs.
                    # For segmentation, we usually attribute with respect to the sum of the target class logit across pixels.
                    target_class = preds.argmax(dim=1).flatten()[0].item() # Simplification for whole image
                else:
                    target_class = preds.argmax(dim=1)
                    
        # Segmentation attribution hack:
        # If output is (B, C, H, W), we need an aggregation function.
        # We will wrap the model to return the sum of the target class logits.
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
        temp_ig = IntegratedGradients(wrapped_model)
        
        attributions = temp_ig.attribute(inputs, target=0, n_steps=self.steps)
        return attributions

    @classmethod
    def supports_transformer(cls) -> bool:
        return True

    @classmethod
    def supports_temporal(cls) -> bool:
        return True


class SmoothGradExplainer(IntegratedGradientsExplainer):
    """Wrapper for Captum's NoiseTunnel (SmoothGrad)."""

    def __init__(self, model: nn.Module, cfg: Any = None) -> None:
        super().__init__(model, cfg)
        self.nt = NoiseTunnel(self.ig)
        self.nt_samples = getattr(self.cfg, "nt_samples", 10) if self.cfg else 10
        self.nt_type = getattr(self.cfg, "nt_type", "smoothgrad") if self.cfg else "smoothgrad"

    def attribute(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor | None = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        """Computes SmoothGrad attributions."""
        inputs.requires_grad_()
        
        if target_class is None:
            with torch.no_grad():
                preds = self.model(inputs)
                if preds.dim() > 2:
                    target_class = preds.argmax(dim=1).flatten()[0].item()
                else:
                    target_class = preds.argmax(dim=1)

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
        temp_ig = IntegratedGradients(wrapped_model)
        temp_nt = NoiseTunnel(temp_ig)
        
        attributions = temp_nt.attribute(
            inputs, 
            target=0, 
            nt_type=self.nt_type, 
            nt_samples=self.nt_samples, 
            n_steps=self.steps
        )
        return attributions


__all__ = ["IntegratedGradientsExplainer", "SmoothGradExplainer"]
