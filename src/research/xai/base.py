"""
CropXAI Base XAI Interface
===========================

Abstract base class for Explainable AI (XAI) methods.
All methods (Grad-CAM, LIME, SHAP, Integrated Gradients, etc.)
must implement this interface in Phase 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseXAIMethod(ABC):
    """
    Abstract base class for XAI methods.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to explain.
    cfg : Any
        XAI configuration node.
    """

    def __init__(self, model: nn.Module, cfg: Any) -> None:
        self.model = model
        self.cfg = cfg
        self.method_name: str = cfg.xai.method

    @abstractmethod
    def explain(self, x: torch.Tensor, target: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """
        Generate an explanation (e.g., saliency map) for the input.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor. Shape: (B, C, H, W).
        target : torch.Tensor, optional
            Target class indices. If None, uses the model's top prediction.
            Shape: (B,) or (B, H, W) depending on task.

        Returns
        -------
        torch.Tensor
            Explanation map. Shape usually (B, H, W) or (B, C, H, W).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.method_name!r})"


__all__ = ["BaseXAIMethod"]
