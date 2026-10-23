"""
CropXAI TIMM CNN Encoder Wrapper
==================================

Wraps `timm` backbones to comply with BaseVisionBackbone.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

from src.core.registry import MODEL_REGISTRY
from src.research.models.interfaces.vision_backbone import BaseVisionBackbone, ModelCapability

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("timm_encoder")
class TimmEncoder(BaseVisionBackbone):
    """
    Wrapper for TIMM CNNs.
    """

    def __init__(
        self,
        cfg: Any,
        in_channels: int,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        output_stride: int = 32,
    ) -> None:
        super().__init__(cfg)
        if timm is None:
            raise ImportError("Please install timm: pip install timm")

        self.backbone_name = backbone_name
        self.output_stride = output_stride
        self.pretrained = pretrained

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        self._feature_info = self.backbone.feature_info

    def _define_capabilities(self) -> ModelCapability:
        return ModelCapability(
            temporal=False,
            multispectral=True,
            rgb=True,
            variable_resolution=True,
            pretraining=True,
            masking=False,
            embeddings=False,
            decoder=True,
            segmentation=True
        )

    def load_pretrained(self, checkpoint_path: str | None = None) -> None:
        pass  # Handled by timm.create_model for standard weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns deepest feature map
        return self.backbone(x)[-1]

    def extract_multiscale(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("CNNs do not natively support sequence embeddings.")

    @property
    def feature_dimensions(self) -> list[int]:
        return self._feature_info.channels()

    @property
    def output_stride(self) -> int:
        return max(self._feature_info.reduction())


__all__ = ["TimmEncoder"]
