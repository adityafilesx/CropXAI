"""
CropXAI Faithfulness Metrics
==============================

Computes Deletion AUC, Insertion AUC, and Infidelity to determine 
if the explanation actually aligns with the model's predictive reasoning.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class FaithfulnessEvaluator:
    """Evaluates explanation faithfulness."""

    def __init__(self, model: nn.Module, steps: int = 10) -> None:
        self.model = model
        self.model.eval()
        self.steps = steps

    def compute_deletion_auc(
        self, 
        inputs: torch.Tensor, 
        attributions: torch.Tensor, 
        target_class: int | torch.Tensor
    ) -> float:
        """
        Sequentially deletes top features and tracks the drop in probability.
        Lower AUC means higher faithfulness (the model quickly drops in confidence 
        when important features are removed).
        """
        # Flatten for sorting
        b = inputs.shape[0]
        flat_attr = attributions.view(b, -1)
        
        # Sort indices descending (most important first)
        sorted_indices = torch.argsort(flat_attr, dim=1, descending=True)
        
        step_size = flat_attr.shape[1] // self.steps
        
        probabilities = []
        current_inputs = inputs.clone()
        
        for i in range(self.steps + 1):
            with torch.no_grad():
                preds = self.model(current_inputs)
                if preds.dim() > 2:
                    if isinstance(target_class, torch.Tensor):
                        prob = preds[torch.arange(b), target_class.flatten(), ...].mean()
                    else:
                        prob = preds[:, target_class, ...].mean()
                else:
                    if isinstance(target_class, torch.Tensor):
                        prob = preds[torch.arange(b), target_class.flatten()].mean()
                    else:
                        prob = preds[:, target_class].mean()
                        
                probabilities.append(prob.item())
                
            if i < self.steps:
                # Ablate the next chunk of features (set to 0 for simplicity)
                start_idx = i * step_size
                end_idx = min((i + 1) * step_size, flat_attr.shape[1])
                indices_to_ablate = sorted_indices[:, start_idx:end_idx]
                
                # Expand indices to match input channels if necessary
                # This depends heavily on whether attribution is per-pixel or per-channel-pixel
                # For this benchmark, assume spatial attribution.
                for batch_idx in range(b):
                    flat_inputs = current_inputs[batch_idx].view(-1)
                    flat_inputs[indices_to_ablate[batch_idx]] = 0.0
                    current_inputs[batch_idx] = flat_inputs.view(current_inputs[batch_idx].shape)
                    
        auc = np.trapz(probabilities, dx=1.0 / self.steps)
        return float(auc)


__all__ = ["FaithfulnessEvaluator"]
