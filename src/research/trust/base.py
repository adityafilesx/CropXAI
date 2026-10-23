"""
CropXAI Universal Trust Interface
===================================

Defines BaseTrustModule, the foundational contract for all 
Trust, Uncertainty, Calibration, and OOD modules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseTrustModule(ABC):
    """Abstract base class for all TCIE components."""

    def __init__(self, model: nn.Module, cfg: Any = None) -> None:
        self.model = model
        self.cfg = cfg
        self.model.eval()

    @abstractmethod
    def fit(self, dataloader: Any) -> None:
        """
        Fits the module if it requires training data 
        (e.g., Temperature Scaling, Mahalanobis covariance).
        """
        pass

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns the primary prediction (logits or probabilities)."""
        pass

    @abstractmethod
    def estimate_uncertainty(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns predictive uncertainty scores (higher = more uncertain)."""
        pass

    @abstractmethod
    def estimate_confidence(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns normalized confidence scores [0, 1]."""
        pass

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        """Evaluates the module's performance on a dataset."""
        return {}


__all__ = ["BaseTrustModule"]
