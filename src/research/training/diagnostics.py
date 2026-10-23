"""
CropXAI Training Diagnostics
==============================

Monitors gradient health, dead layers, and NaN activations to prevent 
silent failure modes during deep VFM fine-tuning.
"""

from __future__ import annotations

import logging
from typing import Any
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrainingDiagnostics:
    """Monitors model health during training."""

    def __init__(self, abort_on_explode: bool = True, max_grad_norm: float = 100.0) -> None:
        self.abort_on_explode = abort_on_explode
        self.max_grad_norm = max_grad_norm

    def check_gradients(self, model: nn.Module) -> dict[str, float]:
        """Calculates gradient norms and checks for explosions."""
        total_norm = 0.0
        dead_layers = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                
                if param_norm.item() == 0.0:
                    dead_layers += 1
                
                if math.isnan(param_norm.item()) or math.isinf(param_norm.item()):
                    msg = f"NaN/Inf gradient detected in layer {name}!"
                    logger.error(msg)
                    if self.abort_on_explode:
                        raise ValueError(msg)
                        
        total_norm = total_norm ** 0.5
        
        if total_norm > self.max_grad_norm:
            msg = f"Gradient explosion detected! Norm: {total_norm:.2f} > {self.max_grad_norm}"
            logger.error(msg)
            if self.abort_on_explode:
                raise ValueError(msg)
                
        return {
            "grad_norm_total": total_norm,
            "dead_layers": dead_layers
        }

    def check_weights(self, model: nn.Module) -> float:
        """Calculates total weight norm."""
        total_norm = 0.0
        for param in model.parameters():
            param_norm = param.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5


__all__ = ["TrainingDiagnostics"]
