"""
CropXAI Granularity Trust Analytics
=====================================

Decomposes image-level trust into Region-wise (Pixel) Trust
and Class-wise Trust.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


class GranularTrustAnalyzer:
    """Computes fine-grained trust."""

    @staticmethod
    def compute_region_trust(
        pixel_confidence: torch.Tensor, 
        pixel_uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes a spatial Trust Map for Semantic Segmentation.
        Inputs: (B, C, H, W) or (B, H, W)
        Returns: (B, H, W) Trust Score map.
        """
        # Simplified formulation for Region Trust
        if pixel_confidence.dim() == 4:
            pixel_confidence = pixel_confidence.max(dim=1)[0]
        if pixel_uncertainty.dim() == 4:
            pixel_uncertainty = pixel_uncertainty.mean(dim=1)
            
        trust_map = (pixel_confidence + (1.0 - pixel_uncertainty)) / 2.0
        return trust_map.clamp(0, 1)

    @staticmethod
    def compute_class_trust(
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        trust_scores: torch.Tensor,
        num_classes: int
    ) -> dict[int, float]:
        """
        Computes average trust score per predicted class.
        Identifies which classes the model trusts itself on vs which it doesn't.
        """
        preds = predictions.argmax(dim=1) if predictions.dim() > 1 else predictions
        if preds.dim() > 1: # Flatten spatial dims
            preds = preds.flatten()
            trust_scores = trust_scores.flatten() # Assumes trust is per-pixel here
            labels = labels.flatten()
            
        class_trust = {}
        for c in range(num_classes):
            mask = (preds == c)
            if mask.sum() > 0:
                class_trust[c] = float(trust_scores[mask].mean().item())
            else:
                class_trust[c] = 0.0
                
        return class_trust


__all__ = ["GranularTrustAnalyzer"]
