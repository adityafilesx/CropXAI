"""
CropXAI Base Uncertainty Interface
===================================

Abstract base class for Uncertainty Estimation methods.
All methods (MC Dropout, Deep Ensembles, Evidential DL)
must implement this interface in Phase 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseUncertaintyMethod(ABC):
    """
    Abstract base class for Uncertainty methods.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to estimate uncertainty for.
    cfg : Any
        Uncertainty configuration node.
    """

    def __init__(self, model: nn.Module, cfg: Any) -> None:
        self.model = model
        self.cfg = cfg
        self.method_name: str = cfg.uncertainty.method

    @abstractmethod
    def estimate(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """
        Estimate predictive uncertainty for the input.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor. Shape: (B, C, H, W).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing uncertainty metrics. Standard keys:
            - "predictive_entropy"
            - "aleatoric_uncertainty"
            - "epistemic_uncertainty"
            - "variance"
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.method_name!r})"


__all__ = ["BaseUncertaintyMethod"]
