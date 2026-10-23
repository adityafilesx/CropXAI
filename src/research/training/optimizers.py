"""
CropXAI Optimizer Registry
============================

Wraps PyTorch optimizers into a registry for dynamic configuration.
"""

from __future__ import annotations

import torch.optim as optim
from typing import Any, Iterable

from src.core.registry import Registry

OPTIMIZER_REGISTRY = Registry("optimizer")


def build_optimizer(cfg: Any, params: Iterable) -> optim.Optimizer:
    """Build optimizer from config."""
    name = cfg.optimizer.name
    lr = cfg.optimizer.lr
    weight_decay = cfg.optimizer.get("weight_decay", 0.0)
    
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = cfg.optimizer.get("momentum", 0.9)
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


__all__ = ["OPTIMIZER_REGISTRY", "build_optimizer"]
