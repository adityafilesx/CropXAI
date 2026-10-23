"""
CropXAI Region-Based Perturbation Explainers
==============================================

Wraps Captum's LIME, Kernel SHAP, Occlusion, and Feature Ablation.
Uses SLIC superpixels internally to compress the search space for 
high-dimensional temporal data.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import numpy as np

try:
    from captum.attr import Lime, KernelShap, Occlusion
    from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
    from skimage.segmentation import slic
except ImportError:
    Lime = None
    KernelShap = None
    Occlusion = None
    slic = None
    SkLearnLinearRegression = None

from src.research.xai.base.explainer import BaseExplainer

logger = logging.getLogger(__name__)


class LimeExplainer(BaseExplainer):
    """Wrapper for Captum's LIME."""

    def __init__(self, model: nn.Module, cfg: Any = None) -> None:
        super().__init__(model, cfg)
        if Lime is None or slic is None:
            raise ImportError("Please install captum and scikit-image.")
            
        self.n_samples = getattr(self.cfg, "n_samples", 150) if self.cfg else 150

    def _generate_superpixels(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generates superpixel segmentation mask for LIME."""
        # SLIC usually expects (H, W, C) numpy arrays
        # For simplicity in this benchmark, we operate on the first batch element
        # and first timestep if temporal.
        img = inputs[0].detach().cpu().numpy()
        if img.ndim == 4: # (T, C, H, W) -> take first T
            img = img[0]
            
        # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
        
        segments = slic(img, n_segments=50, compactness=10, start_label=0)
        
        mask = torch.tensor(segments, device=inputs.device).unsqueeze(0)
        if inputs.dim() == 5:
            # Expand to (B, T, 1, H, W) or similar
            mask = mask.unsqueeze(0).unsqueeze(0).expand(inputs.shape[0], inputs.shape[1], 1, inputs.shape[3], inputs.shape[4])
        else:
            mask = mask.unsqueeze(0)
        return mask

    def attribute(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor | None = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        
        if target_class is None:
            with torch.no_grad():
                preds = self.model(inputs)
                target_class = preds.argmax(dim=1).flatten()[0].item() if preds.dim() > 2 else preds.argmax(dim=1)
                
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
        
        lime = Lime(
            wrapped_model, 
            interpretable_model=SkLearnLinearRegression()
        )
        
        mask = self._generate_superpixels(inputs)
        
        attributions = lime.attribute(
            inputs, 
            target=0, 
            feature_mask=mask, 
            n_samples=self.n_samples,
            perturbations_per_eval=16
        )
        return attributions

    @classmethod
    def supports_transformer(cls) -> bool:
        return True

    @classmethod
    def supports_temporal(cls) -> bool:
        return True


class OcclusionExplainer(BaseExplainer):
    """Wrapper for Captum's Occlusion Sensitivity."""
    
    def __init__(self, model: nn.Module, cfg: Any = None) -> None:
        super().__init__(model, cfg)
        if Occlusion is None:
            raise ImportError("captum not installed.")
        self.occlusion = Occlusion(self.model)

    def attribute(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor | None = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        
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
        temp_occ = Occlusion(wrapped_model)
        
        sliding_window = (1, 3, 16, 16) if inputs.dim() == 5 else (3, 16, 16)
        
        attributions = temp_occ.attribute(
            inputs,
            strides=sliding_window,
            target=0,
            sliding_window_shapes=sliding_window,
            baselines=0
        )
        return attributions
        
    @classmethod
    def supports_transformer(cls) -> bool:
        return True

    @classmethod
    def supports_temporal(cls) -> bool:
        return True


__all__ = ["LimeExplainer", "OcclusionExplainer"]
