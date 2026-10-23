"""
CropXAI Spectral Explainability Analyzer
==========================================

Specifically designed for Remote Sensing multispectral imagery.
Quantifies which spectral bands (e.g., NIR, SWIR, Red Edge) 
drive predictions the most.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Any

logger = logging.getLogger(__name__)


class SpectralExplainabilityAnalyzer:
    """Computes band importance and spectral sensitivity."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def compute_band_importance(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor,
        baseline_mode: str = "zero"
    ) -> torch.Tensor:
        """
        Computes how much the prediction drops if a specific band is ablated.
        inputs: (B, T, C, H, W) or (B, C, H, W)
        Returns:
            Importance scores per band: (B, C)
        """
        with torch.no_grad():
            base_preds = self.model(inputs)
            if base_preds.dim() > 2:
                # Target probability for the spatial mean
                if isinstance(target_class, torch.Tensor):
                    base_prob = base_preds[torch.arange(inputs.shape[0]), target_class.flatten(), ...].mean(dim=[-1, -2])
                else:
                    base_prob = base_preds[:, target_class, ...].mean(dim=[-1, -2])
            else:
                if isinstance(target_class, torch.Tensor):
                    base_prob = base_preds[torch.arange(inputs.shape[0]), target_class.flatten()]
                else:
                    base_prob = base_preds[:, target_class]
                
        C_dim = 2 if inputs.dim() == 5 else 1
        num_channels = inputs.shape[C_dim]
        
        importance_scores = torch.zeros((inputs.shape[0], num_channels), device=inputs.device)
        
        for c in range(num_channels):
            ablated_inputs = inputs.clone()
            
            # Ablate channel c
            if inputs.dim() == 5:
                if baseline_mode == "zero":
                    ablated_inputs[:, :, c, :, :] = 0.0
                elif baseline_mode == "mean":
                    ablated_inputs[:, :, c, :, :] = inputs[:, :, c, :, :].mean()
            else:
                if baseline_mode == "zero":
                    ablated_inputs[:, c, :, :] = 0.0
                elif baseline_mode == "mean":
                    ablated_inputs[:, c, :, :] = inputs[:, c, :, :].mean()
                    
            with torch.no_grad():
                abl_preds = self.model(ablated_inputs)
                if abl_preds.dim() > 2:
                    if isinstance(target_class, torch.Tensor):
                        abl_prob = abl_preds[torch.arange(inputs.shape[0]), target_class.flatten(), ...].mean(dim=[-1, -2])
                    else:
                        abl_prob = abl_preds[:, target_class, ...].mean(dim=[-1, -2])
                else:
                    if isinstance(target_class, torch.Tensor):
                        abl_prob = abl_preds[torch.arange(inputs.shape[0]), target_class.flatten()]
                    else:
                        abl_prob = abl_preds[:, target_class]
                    
            # Importance = Base Prob - Ablated Prob
            # High drop means high importance
            importance_scores[:, c] = base_prob - abl_prob
            
        # Normalize scores to sum to 1 per batch item
        importance_scores = torch.clamp(importance_scores, min=0)
        sums = importance_scores.sum(dim=1, keepdim=True)
        sums = torch.where(sums == 0, torch.ones_like(sums), sums)
        return importance_scores / sums


__all__ = ["SpectralExplainabilityAnalyzer"]
