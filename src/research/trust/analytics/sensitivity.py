"""
CropXAI Trust Sensitivity & Drift
===================================

Computes how sensitive the Trust Score is to changes in underlying modules
and tracks Trust Drift over training epochs.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class TrustSensitivityAnalyzer:
    """Analyzes the robustness and stability of the Trust Estimator itself."""

    @staticmethod
    def compute_sensitivity(
        estimator: Any, 
        test_features: dict[str, torch.Tensor],
        perturbation_std: float = 0.1
    ) -> dict[str, float]:
        """
        Injects noise into individual features (e.g. Uncertainty) and measures
        how much the final Trust Score shifts.
        """
        base_trust = estimator.estimate_trust(**test_features)
        
        sensitivities = {}
        for key, val in test_features.items():
            perturbed_features = test_features.copy()
            noise = torch.randn_like(val) * perturbation_std
            perturbed_features[key] = torch.clamp(val + noise, 0, 1)
            
            perturbed_trust = estimator.estimate_trust(**perturbed_features)
            
            # Mean absolute difference
            mad = torch.abs(base_trust - perturbed_trust).mean().item()
            sensitivities[key] = mad
            
        return sensitivities

    @staticmethod
    def compute_trust_drift(epoch_trust_scores: list[torch.Tensor]) -> list[float]:
        """
        Computes the variance of the Trust Score across training epochs.
        Used to plot the Trust Evolution curve.
        """
        if len(epoch_trust_scores) < 2:
            return [0.0]
            
        drifts = [0.0]
        for i in range(1, len(epoch_trust_scores)):
            prev = epoch_trust_scores[i-1]
            curr = epoch_trust_scores[i]
            drift = torch.abs(curr - prev).mean().item()
            drifts.append(drift)
            
        return drifts


__all__ = ["TrustSensitivityAnalyzer"]
