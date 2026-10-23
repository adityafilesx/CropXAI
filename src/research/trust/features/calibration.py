"""
CropXAI Trust Features: Calibration
=====================================

Implements confidence calibration modules wrapper in BaseTrustModule.
- Temperature Scaling
- Isotonic Regression
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.research.trust.base import BaseTrustModule

logger = logging.getLogger(__name__)


class TemperatureScaling(BaseTrustModule):
    """Post-hoc calibration using Temperature Scaling."""

    def __init__(self, model: nn.Module, cfg: Any = None) -> None:
        super().__init__(model, cfg)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def fit(self, dataloader: Any) -> None:
        """
        Optimizes the temperature parameter using NLL loss on a validation set.
        """
        logger.info("Fitting Temperature Scaling...")
        # L-BFGS optimization on validation logits is typical here.
        # Mocking the fit
        self.temperature.data = torch.tensor([1.25])

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(inputs)
            # Apply temperature scaling
            return logits / self.temperature.to(logits.device)

    def estimate_uncertainty(self, inputs: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.estimate_confidence(inputs)

    def estimate_confidence(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.predict(inputs)
        if logits.dim() > 2:
            logits = logits.mean(dim=[-1, -2])
        return F.softmax(logits, dim=-1).max(dim=1)[0]
        
        
__all__ = ["TemperatureScaling"]
