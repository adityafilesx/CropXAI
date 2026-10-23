"""
CropXAI Scheduler Registry
============================

Wraps PyTorch learning rate schedulers.
"""

from __future__ import annotations

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Optimizer
from typing import Any

from src.core.registry import Registry

SCHEDULER_REGISTRY = Registry("scheduler")


def build_scheduler(cfg: Any, optimizer: Optimizer) -> dict[str, Any] | None:
    """Build scheduler from config and format it for PyTorch Lightning."""
    if not cfg.get("scheduler") or not cfg.scheduler.get("name"):
        return None
        
    name = cfg.scheduler.name
    
    if name == "cosine":
        t_max = cfg.scheduler.get("t_max", cfg.training.epochs)
        eta_min = cfg.scheduler.get("eta_min", 1e-6)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        return {"scheduler": scheduler, "interval": "epoch"}
        
    elif name == "onecycle":
        max_lr = cfg.scheduler.get("max_lr", cfg.optimizer.lr * 10)
        total_steps = cfg.scheduler.get("total_steps", cfg.training.epochs)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
        return {"scheduler": scheduler, "interval": "step"}
        
    elif name == "plateau":
        factor = cfg.scheduler.get("factor", 0.5)
        patience = cfg.scheduler.get("patience", 10)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)
        return {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}
        
    elif name == "step":
        step_size = cfg.scheduler.get("step_size", 30)
        gamma = cfg.scheduler.get("gamma", 0.1)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return {"scheduler": scheduler, "interval": "epoch"}
        
    else:
        raise ValueError(f"Unknown scheduler: {name}")


__all__ = ["SCHEDULER_REGISTRY", "build_scheduler"]
