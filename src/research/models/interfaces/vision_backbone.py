"""
CropXAI Universal Vision Backbone Interface
=============================================

Defines the core contract that all backbones (CNNs, Transformers, 
Foundation Models) must adhere to for benchmark compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ModelCapability:
    """Descriptor defining what a specific backbone architecture natively supports."""
    temporal: bool = False
    multispectral: bool = False
    rgb: bool = True
    variable_resolution: bool = False
    pretraining: bool = True
    masking: bool = False
    embeddings: bool = True
    decoder: bool = True
    segmentation: bool = True
    embedding_dim: int = 768


class BaseVisionBackbone(nn.Module, ABC):
    """
    Abstract Base Class for all Vision Architectures in CropXAI.
    Replaces the legacy BaseEncoder.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self._capability = self._define_capabilities()

    @abstractmethod
    def _define_capabilities(self) -> ModelCapability:
        """Returns the capabilities supported by this architecture."""
        ...

    @property
    def capability(self) -> ModelCapability:
        return self._capability

    @abstractmethod
    def load_pretrained(self, checkpoint_path: str | None = None) -> None:
        """Load pretrained weights."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning the deepest feature map or token sequence."""
        ...

    @abstractmethod
    def extract_multiscale(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale feature maps required for standard Decoders.
        Should return a list of tensors from fine to coarse resolution.
        """
        ...

    @abstractmethod
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the flattened sequence of embeddings/tokens (if applicable).
        """
        ...

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True

    @property
    @abstractmethod
    def feature_dimensions(self) -> list[int]:
        """Return the channel dimensions for the multi-scale features."""
        ...

    @property
    @abstractmethod
    def output_stride(self) -> int:
        """Return the maximum spatial reduction factor (usually 32)."""
        ...

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["BaseVisionBackbone", "ModelCapability"]
