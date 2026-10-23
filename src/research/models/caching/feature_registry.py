"""
CropXAI Feature Registry
==========================

Registry for naming and hooking into specific feature layers.
Essential for Explainable AI (XAI) methods like GradCAM and SHAP.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Hooks into model layers to extract and store intermediate features by name.
    """

    def __init__(self) -> None:
        self.features: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []

    def _get_hook(self, name: str) -> Callable:
        def hook(model: nn.Module, input: Any, output: torch.Tensor) -> None:
            self.features[name] = output
        return hook

    def register_layer(self, name: str, layer: nn.Module) -> None:
        """Register a forward hook on a specific layer."""
        handle = layer.register_forward_hook(self._get_hook(name))
        self.hooks.append(handle)
        logger.debug("Registered hook for layer: %s", name)

    def get_feature(self, name: str) -> torch.Tensor | None:
        """Retrieve the captured feature by name."""
        return self.features.get(name)

    def clear(self) -> None:
        """Clear captured features."""
        self.features.clear()

    def remove_hooks(self) -> None:
        """Remove all hooks from the model."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()


__all__ = ["FeatureRegistry"]
