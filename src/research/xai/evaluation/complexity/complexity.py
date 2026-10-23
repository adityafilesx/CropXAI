"""
CropXAI Explanation Complexity
================================

Computes the entropy and sparsity of explanation heatmaps. 
Simpler, contiguous explanations are generally preferred over noisy ones.
"""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


class ComplexityEvaluator:
    """Computes sparsity and entropy of attributions."""

    @staticmethod
    def compute_sparsity(attr: torch.Tensor, threshold: float = 0.1) -> float:
        """
        Percentage of pixels with attribution near zero.
        Higher sparsity = simpler explanation.
        """
        a = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        sparse_pixels = (a < threshold).float().sum()
        total_pixels = attr.numel()
        return float((sparse_pixels / total_pixels).item())

    @staticmethod
    def compute_entropy(attr: torch.Tensor) -> float:
        """
        Shannon entropy of the attribution distribution.
        Lower entropy = simpler, peakier explanation.
        """
        a = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        # Normalize to probability distribution
        a_sum = a.sum()
        if a_sum == 0:
            return 0.0
            
        p = a / a_sum
        
        # Entropy
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        return float(entropy.item())


__all__ = ["ComplexityEvaluator"]
