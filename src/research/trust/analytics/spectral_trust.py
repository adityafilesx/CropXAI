"""
CropXAI Spectral Trust Analytics
==================================

Computes Spectral Trust by ablating bands and observing the drop in
the final Trust Score. Unique remote sensing contribution.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.research.trust.estimation.estimator import BaseTrustEstimator

logger = logging.getLogger(__name__)


class SpectralTrustAnalyzer:
    """Computes band contributions to Trust Score."""

    def __init__(self, trust_engine: Any) -> None:
        # trust_engine orchestrates the full pipeline from raw input to Trust Score
        self.trust_engine = trust_engine

    def compute_band_trust_contribution(
        self, 
        inputs: torch.Tensor, 
        baseline_mode: str = "zero"
    ) -> torch.Tensor:
        """
        Ablates spectral bands and computes the change in the final Trust Score.
        Returns (B, C) attribution scores.
        """
        with torch.no_grad():
            base_trust = self.trust_engine.estimate_end_to_end(inputs)
            
        C_dim = 2 if inputs.dim() == 5 else 1
        num_channels = inputs.shape[C_dim]
        
        trust_contributions = torch.zeros((inputs.shape[0], num_channels), device=inputs.device)
        
        for c in range(num_channels):
            ablated_inputs = inputs.clone()
            
            if inputs.dim() == 5:
                if baseline_mode == "zero":
                    ablated_inputs[:, :, c, :, :] = 0.0
                else:
                    ablated_inputs[:, :, c, :, :] = inputs[:, :, c, :, :].mean()
            else:
                if baseline_mode == "zero":
                    ablated_inputs[:, c, :, :] = 0.0
                else:
                    ablated_inputs[:, c, :, :] = inputs[:, c, :, :].mean()
                    
            with torch.no_grad():
                abl_trust = self.trust_engine.estimate_end_to_end(ablated_inputs)
                
            # If trust drops when band is ablated, the band positively contributes to trust
            trust_contributions[:, c] = base_trust - abl_trust
            
        return trust_contributions


__all__ = ["SpectralTrustAnalyzer"]
