"""
CropXAI Counterfactual Explainer
==================================

Implements Counterfactual Search and Prototype Swap.
Discovers the minimal spatial/temporal perturbation required to 
flip a model's prediction.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.research.xai.base.explainer import BaseExplainer

logger = logging.getLogger(__name__)


class CounterfactualExplainer(BaseExplainer):
    """Searches for minimal counterfactual perturbations."""

    def __init__(self, model: nn.Module, cfg: Any = None) -> None:
        super().__init__(model, cfg)
        self.lr = getattr(self.cfg, "lr", 0.01) if self.cfg else 0.01
        self.steps = getattr(self.cfg, "steps", 100) if self.cfg else 100
        self.lambda_reg = getattr(self.cfg, "lambda_reg", 0.1) if self.cfg else 0.1

    def attribute(
        self, 
        inputs: torch.Tensor, 
        target_class: int | torch.Tensor | None = None, 
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Uses gradient descent on the input space to find a counterfactual.
        Returns the absolute difference (perturbation) as the "attribution".
        """
        if target_class is None:
            with torch.no_grad():
                preds = self.model(inputs)
                original_class = preds.argmax(dim=1).flatten()[0].item() if preds.dim() > 2 else preds.argmax(dim=1)
            # Find a target class that is NOT the original class
            target_class = (original_class + 1) % preds.shape[1]
            
        counterfactual = inputs.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([counterfactual], lr=self.lr)
        
        criterion = nn.CrossEntropyLoss()
        
        for step in range(self.steps):
            optimizer.zero_grad()
            
            outputs = self.model(counterfactual)
            
            # If segmentation, take mean over spatial
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=[-1, -2])
                
            # We want to minimize loss with respect to the TARGET class
            target_tensor = torch.tensor([target_class] * outputs.shape[0], device=inputs.device)
            loss_pred = criterion(outputs, target_tensor)
            
            # L1/L2 regularization to keep perturbation minimal
            loss_reg = torch.norm(counterfactual - inputs, p=1)
            
            loss = loss_pred + self.lambda_reg * loss_reg
            loss.backward()
            optimizer.step()
            
            # Project back to valid image range if necessary (e.g. [0, 1])
            counterfactual.data = torch.clamp(counterfactual.data, 0, 1)
            
        # The explanation is the minimal change required
        perturbation = torch.abs(counterfactual - inputs)
        return perturbation

    @classmethod
    def supports_transformer(cls) -> bool:
        return True

    @classmethod
    def supports_temporal(cls) -> bool:
        return True


__all__ = ["CounterfactualExplainer"]
