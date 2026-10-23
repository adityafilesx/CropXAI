"""
CropXAI Loss Functions & Registry
==================================

Research-oriented semantic segmentation losses.
Registered in LOSS_REGISTRY.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import Registry

LOSS_REGISTRY = Registry("loss")


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg: Any = None, weight: Optional[torch.Tensor] = None, ignore_index: int = 255) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


@LOSS_REGISTRY.register("dice")
class DiceLoss(nn.Module):
    def __init__(self, cfg: Any = None, smooth: float = 1.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        # Mask out ignore index
        mask = targets != self.ignore_index
        targets = torch.where(mask, targets, torch.zeros_like(targets))
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Apply mask
        mask = mask.unsqueeze(1).float()
        probs = probs * mask
        targets_one_hot = targets_one_hot * mask
        
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)


@LOSS_REGISTRY.register("focal")
class FocalLoss(nn.Module):
    def __init__(self, cfg: Any = None, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Only average over valid pixels
        mask = (targets != self.ignore_index).float()
        return focal_loss.sum() / mask.sum().clamp(min=1.0)


@LOSS_REGISTRY.register("dice_ce")
class DiceCELoss(nn.Module):
    """Combination of Dice and Cross-Entropy Loss."""
    def __init__(self, cfg: Any = None, ce_weight: float = 0.5, dice_weight: float = 0.5, ignore_index: int = 255) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)


__all__ = ["LOSS_REGISTRY", "CrossEntropyLoss", "DiceLoss", "FocalLoss", "DiceCELoss"]
