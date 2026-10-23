"""
CropXAI Robustness Metrics
============================

Injects on-the-fly perturbations (noise, blur, etc.) and evaluates 
Explanation Drift (Heatmap Variance, Stability).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.research.xai.base.explainer import BaseExplainer

logger = logging.getLogger(__name__)


class RobustnessEvaluator:
    """Evaluates explanation stability under input perturbations."""

    def __init__(self, explainer: BaseExplainer) -> None:
        self.explainer = explainer

    def _inject_gaussian_noise(self, inputs: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        noise = torch.randn_like(inputs) * std
        return inputs + noise

    def compute_explanation_stability(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor,
        noise_std: float = 0.05,
        n_samples: int = 5
    ) -> dict[str, float]:
        """
        Computes how stable the explanation is when small noise is injected.
        Returns the mean Structural Similarity or Cosine Similarity.
        """
        base_attr = self.explainer.explain(inputs, target_class=target_class)
        base_attr_flat = base_attr.flatten(1)
        
        similarities = []
        for _ in range(n_samples):
            noisy_inputs = self._inject_gaussian_noise(inputs, std=noise_std)
            noisy_attr = self.explainer.explain(noisy_inputs, target_class=target_class)
            noisy_attr_flat = noisy_attr.flatten(1)
            
            # Cosine similarity
            sim = F.cosine_similarity(base_attr_flat, noisy_attr_flat, dim=1).mean().item()
            similarities.append(sim)
            
        return {
            "mean_stability": sum(similarities) / len(similarities),
            "variance": sum((x - sum(similarities)/len(similarities))**2 for x in similarities) / len(similarities)
        }


__all__ = ["RobustnessEvaluator"]
