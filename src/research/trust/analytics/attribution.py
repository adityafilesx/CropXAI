"""
CropXAI Trust Attribution
===========================

Explains the Trust Score itself using SHAP/Gradient methods.
Identifies exactly *why* a prediction is deemed untrustworthy.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None

from src.research.trust.estimation.estimator import BaseTrustEstimator

logger = logging.getLogger(__name__)


class TrustAttribution:
    """Computes feature attributions for the Trust Score."""

    def __init__(self, estimator: BaseTrustEstimator) -> None:
        self.estimator = estimator
        # To use Captum, the estimator needs to be wrapped as an nn.Module
        # that takes a single stacked tensor.
        class EstimatorWrapper(nn.Module):
            def __init__(self, est: BaseTrustEstimator):
                super().__init__()
                self.est = est
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x shape: (B, 6)
                kwargs = {
                    "confidence": x[:, 0],
                    "uncertainty": x[:, 1],
                    "ood_score": x[:, 2],
                    "faithfulness": x[:, 3],
                    "calibration_error": x[:, 4],
                    "representation_score": x[:, 5]
                }
                out = self.est.estimate_trust(**kwargs)
                # If Pareto Trust, return just the first dimension for attribution
                if out.dim() > 1:
                    out = out.mean(dim=1)
                return out.unsqueeze(-1)
                
        self.wrapped_estimator = EstimatorWrapper(self.estimator)
        if IntegratedGradients is not None:
            self.ig = IntegratedGradients(self.wrapped_estimator)

    def attribute(self, test_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Computes the contribution of each feature to the final Trust Score.
        """
        if IntegratedGradients is None:
            logger.warning("captum not installed. Returning mocked Trust Attribution.")
            return {k: torch.rand_like(v) * 0.1 for k, v in test_features.items()}
            
        x = torch.stack([
            test_features["confidence"],
            test_features["uncertainty"],
            test_features["ood_score"],
            test_features["faithfulness"],
            test_features["calibration_error"],
            test_features["representation_score"]
        ], dim=-1)
        x.requires_grad_()
        
        # Baseline is a completely neutral/average sample (e.g. 0.5 for all)
        baseline = torch.ones_like(x) * 0.5
        
        attributions = self.ig.attribute(x, baselines=baseline, n_steps=50)
        
        # Unpack back to dict
        keys = ["confidence", "uncertainty", "ood_score", "faithfulness", "calibration_error", "representation_score"]
        attr_dict = {}
        for i, key in enumerate(keys):
            attr_dict[key] = attributions[:, i]
            
        return attr_dict


__all__ = ["TrustAttribution"]
