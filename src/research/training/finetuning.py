"""
CropXAI Fine-Tuning Strategy Engine
=====================================

Abstraction for applying principled fine-tuning strategies 
to Vision Foundation Models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


class FineTuningStrategy(ABC):
    """Base interface for all fine-tuning strategies."""
    
    @abstractmethod
    def apply(self, backbone: nn.Module, head: nn.Module) -> None:
        """Applies freezing and unfreezing rules."""
        ...


class FrozenStrategy(FineTuningStrategy):
    """Freezes both backbone and head (for feature extraction / representation analysis)."""
    def apply(self, backbone: nn.Module, head: nn.Module) -> None:
        for param in backbone.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = False
        logger.info("Strategy applied: Frozen")


class LinearProbeStrategy(FineTuningStrategy):
    """Freezes backbone, trains head."""
    def apply(self, backbone: nn.Module, head: nn.Module) -> None:
        for param in backbone.parameters():
            param.requires_grad = False
        for param in head.parameters():
            param.requires_grad = True
        logger.info("Strategy applied: Linear Probe")


class FullFineTuningStrategy(FineTuningStrategy):
    """Trains everything."""
    def apply(self, backbone: nn.Module, head: nn.Module) -> None:
        for param in backbone.parameters():
            param.requires_grad = True
        for param in head.parameters():
            param.requires_grad = True
        logger.info("Strategy applied: Full Fine-Tuning")


class PartialFineTuningStrategy(FineTuningStrategy):
    """
    Unfreezes the last N blocks/stages of the backbone.
    """
    def __init__(self, last_n_blocks: int = 1) -> None:
        self.last_n_blocks = last_n_blocks

    def apply(self, backbone: nn.Module, head: nn.Module) -> None:
        # Implementation depends on standardizing block naming in backbones,
        # or recursively walking modules.
        # For this framework, we expect backbones to expose a `unfreeze_last_n_blocks()` 
        # or we manually handle known architectures.
        
        # Default: Freeze all, then unfreeze last N if supported
        for param in backbone.parameters():
            param.requires_grad = False
            
        if hasattr(backbone, "unfreeze_last_n_blocks"):
            backbone.unfreeze_last_n_blocks(self.last_n_blocks)
        else:
            logger.warning("Backbone doesn't support unfreeze_last_n_blocks. Remaining fully frozen.")
            
        for param in head.parameters():
            param.requires_grad = True
            
        logger.info("Strategy applied: Partial Fine-Tuning (last %d blocks)", self.last_n_blocks)


class StrategyFactory:
    @staticmethod
    def get(name: str, **kwargs: Any) -> FineTuningStrategy:
        name = name.lower()
        if name == "frozen":
            return FrozenStrategy()
        elif name == "linear_probe":
            return LinearProbeStrategy()
        elif name == "full":
            return FullFineTuningStrategy()
        elif name == "partial":
            return PartialFineTuningStrategy(**kwargs)
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {name}")


__all__ = ["FineTuningStrategy", "StrategyFactory"]
