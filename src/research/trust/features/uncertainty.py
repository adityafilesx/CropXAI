"""
CropXAI Trust Features: Predictive Uncertainty
==============================================

Implements Aleatoric and Epistemic Uncertainty Estimation.
- MC Dropout (Epistemic)
- Deep Ensembles (Epistemic)
- Heteroscedastic Variance (Aleatoric)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.research.trust.base import BaseTrustModule

logger = logging.getLogger(__name__)


class MCDropoutUncertainty(BaseTrustModule):
    """Estimates Epistemic Uncertainty using Monte Carlo Dropout."""

    def __init__(self, model: nn.Module, num_samples: int = 10, cfg: Any = None) -> None:
        super().__init__(model, cfg)
        self.num_samples = num_samples

    def fit(self, dataloader: Any) -> None:
        pass

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns the mean prediction over MC samples."""
        self.model.train() # Enable Dropout
        with torch.no_grad():
            preds = [self.model(inputs) for _ in range(self.num_samples)]
            stacked = torch.stack(preds, dim=0)
            return stacked.mean(dim=0)

    def estimate_uncertainty(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Returns predictive entropy or variance as uncertainty.
        Using variance of the softmax probabilities for simplicity.
        """
        self.model.train()
        with torch.no_grad():
            preds = []
            for _ in range(self.num_samples):
                logits = self.model(inputs)
                if logits.dim() > 2:
                    # Spatial averaging for simplicity in metric
                    logits = logits.mean(dim=[-1, -2])
                probs = F.softmax(logits, dim=-1)
                preds.append(probs)
                
            stacked = torch.stack(preds, dim=0) # (Samples, B, C)
            variance = stacked.var(dim=0) # (B, C)
            # Max variance across classes as the uncertainty score
            uncertainty = variance.max(dim=1)[0]
            
        return uncertainty

    def estimate_confidence(self, inputs: torch.Tensor) -> torch.Tensor:
        mean_preds = self.predict(inputs)
        if mean_preds.dim() > 2:
            mean_preds = mean_preds.mean(dim=[-1, -2])
        probs = F.softmax(mean_preds, dim=-1)
        return probs.max(dim=1)[0]


class DeepEnsembleUncertainty(BaseTrustModule):
    """Estimates Epistemic Uncertainty using a Deep Ensemble of models."""

    def __init__(self, models: list[nn.Module], cfg: Any = None) -> None:
        if not models:
            raise ValueError("DeepEnsemble requires at least 1 model.")
        super().__init__(models[0], cfg)
        self.models = models
        for m in self.models:
            m.eval()

    def fit(self, dataloader: Any) -> None:
        pass

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds = [m(inputs) for m in self.models]
            return torch.stack(preds, dim=0).mean(dim=0)

    def estimate_uncertainty(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds = []
            for m in self.models:
                logits = m(inputs)
                if logits.dim() > 2:
                    logits = logits.mean(dim=[-1, -2])
                probs = F.softmax(logits, dim=-1)
                preds.append(probs)
                
            stacked = torch.stack(preds, dim=0)
            variance = stacked.var(dim=0)
            return variance.max(dim=1)[0]

    def estimate_confidence(self, inputs: torch.Tensor) -> torch.Tensor:
        mean_preds = self.predict(inputs)
        if mean_preds.dim() > 2:
            mean_preds = mean_preds.mean(dim=[-1, -2])
        probs = F.softmax(mean_preds, dim=-1)
        return probs.max(dim=1)[0]


__all__ = ["MCDropoutUncertainty", "DeepEnsembleUncertainty"]
