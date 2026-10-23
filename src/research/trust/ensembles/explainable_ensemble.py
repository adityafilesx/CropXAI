"""
CropXAI Explainability-Aware Ensemble
=======================================

Dynamically weights models (e.g., CNN, ViT, VFM) based on their 
individual Trust Scores rather than a simple probability average.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrustAwareEnsemble(nn.Module):
    """
    Combines predictions using dynamic Trust Score weights.
    If the VFM has high uncertainty and OOD on a patch, but the CNN 
    is highly confident and well-calibrated, the CNN's prediction dominates.
    """

    def __init__(self, models: list[Any], trust_estimators: list[Any]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.trust_estimators = trust_estimators

    def forward(
        self, 
        inputs: torch.Tensor, 
        trust_features_list: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (B, C, H, W) or (B, T, C, H, W)
            trust_features_list: A list (one per model) of trust feature dicts.
            
        Returns:
            Weighted logits and the final fused Trust Score.
        """
        b = inputs.shape[0]
        device = inputs.device
        
        all_logits = []
        all_trusts = []
        
        for i, model in enumerate(self.models):
            logits = model(inputs)
            if logits.dim() > 2:
                # Assuming spatial aggregation for simplicity
                logits = logits.mean(dim=[-1, -2])
            all_logits.append(logits)
            
            # Compute Trust Score for this specific model on this input
            estimator = self.trust_estimators[i]
            features = trust_features_list[i]
            trust = estimator.estimate_trust(**features)
            
            # If Pareto Trust, take the mean across trust dimensions for scalar weighting
            if trust.dim() > 1:
                trust = trust.mean(dim=1)
                
            all_trusts.append(trust)
            
        # all_trusts: list of (B,) tensors
        # Stack into (B, num_models)
        stacked_trust = torch.stack(all_trusts, dim=1)
        
        # Softmax over trust scores to get dynamic weights
        # Temperature can be applied here to sharpen weights
        weights = torch.softmax(stacked_trust * 2.0, dim=1) # (B, num_models)
        
        # Stack logits: (B, num_models, num_classes)
        stacked_logits = torch.stack(all_logits, dim=1)
        
        # Weighted sum of logits
        # (B, 1, num_models) @ (B, num_models, num_classes) -> (B, 1, num_classes)
        weighted_logits = torch.bmm(weights.unsqueeze(1), stacked_logits).squeeze(1)
        
        # Fused Trust Score (weighted average of individual trusts)
        fused_trust = (weights * stacked_trust).sum(dim=1)
        
        return weighted_logits, fused_trust


__all__ = ["TrustAwareEnsemble"]
