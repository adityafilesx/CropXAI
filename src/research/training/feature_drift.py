"""
CropXAI Feature Drift Analyzer
================================

Tracks how pretrained foundation model features drift during fine-tuning.
Computes CKA (Centered Kernel Alignment) and Cosine Drift.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureDriftAnalyzer:
    """Analyzes representation changes between pretrained and fine-tuned states."""

    def __init__(self) -> None:
        self.reference_features: dict[str, torch.Tensor] = {}

    def set_reference(self, name: str, features: torch.Tensor) -> None:
        """Store the pretrained reference features for a given batch."""
        self.reference_features[name] = features.detach().clone()

    def compute_cosine_drift(self, name: str, current_features: torch.Tensor) -> float:
        """Compute mean cosine similarity between reference and current."""
        if name not in self.reference_features:
            return 0.0
            
        ref = self.reference_features[name]
        
        # Flatten spatial dims if necessary: (B, C, H, W) -> (B, C, H*W)
        if ref.dim() == 4:
            ref = ref.flatten(2)
            current = current_features.flatten(2)
            
        # Cosine similarity along channel dimension
        sim = F.cosine_similarity(ref, current, dim=1)
        return float(sim.mean().item())
        
    def compute_cka(self, name: str, current_features: torch.Tensor) -> float:
        """
        Compute Linear Centered Kernel Alignment (CKA) between reference and current.
        A measure of representation similarity invariant to orthogonal transformation.
        """
        if name not in self.reference_features:
            return 0.0
            
        X = self.reference_features[name]
        Y = current_features
        
        if X.dim() > 2:
            X = X.flatten(1)
            Y = Y.flatten(1)
            
        # Center the representations
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute dot products
        dot_XX = torch.sum(X * X)
        dot_YY = torch.sum(Y * Y)
        dot_XY = torch.sum(X * Y)
        
        if dot_XX == 0 or dot_YY == 0:
            return 0.0
            
        cka = (dot_XY ** 2) / (dot_XX * dot_YY)
        return float(cka.item())


__all__ = ["FeatureDriftAnalyzer"]
