"""
CropXAI Temporal Explainability Analyzer
==========================================

Specifically designed for Time-Series Earth Observation.
Quantifies Frame Importance and Temporal Drift over the T dimension.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Any

logger = logging.getLogger(__name__)


class TemporalExplainabilityAnalyzer:
    """Computes Frame Importance and Time Sensitivity."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def compute_frame_importance(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor,
        baseline_mode: str = "zero"
    ) -> torch.Tensor:
        """
        Computes how much the prediction drops if a specific timestamp is ablated.
        inputs: (B, T, C, H, W)
        Returns:
            Importance scores per timestamp: (B, T)
        """
        if inputs.dim() != 5:
            # If (B, C, H, W), temporal importance is undefined/1.0
            return torch.ones((inputs.shape[0], 1), device=inputs.device)
            
        with torch.no_grad():
            base_preds = self.model(inputs)
            if base_preds.dim() > 2:
                if isinstance(target_class, torch.Tensor):
                    base_prob = base_preds[torch.arange(inputs.shape[0]), target_class.flatten(), ...].mean(dim=[-1, -2])
                else:
                    base_prob = base_preds[:, target_class, ...].mean(dim=[-1, -2])
            else:
                if isinstance(target_class, torch.Tensor):
                    base_prob = base_preds[torch.arange(inputs.shape[0]), target_class.flatten()]
                else:
                    base_prob = base_preds[:, target_class]
                
        num_frames = inputs.shape[1]
        importance_scores = torch.zeros((inputs.shape[0], num_frames), device=inputs.device)
        
        for t in range(num_frames):
            ablated_inputs = inputs.clone()
            
            # Ablate frame t
            if baseline_mode == "zero":
                ablated_inputs[:, t, :, :, :] = 0.0
            elif baseline_mode == "mean":
                ablated_inputs[:, t, :, :, :] = inputs[:, t, :, :, :].mean(dim=[2, 3, 4], keepdim=True)
                
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
                    
            importance_scores[:, t] = base_prob - abl_prob
            
        importance_scores = torch.clamp(importance_scores, min=0)
        sums = importance_scores.sum(dim=1, keepdim=True)
        sums = torch.where(sums == 0, torch.ones_like(sums), sums)
        return importance_scores / sums


__all__ = ["TemporalExplainabilityAnalyzer"]
