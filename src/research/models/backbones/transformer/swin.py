"""
CropXAI Swin Transformer Backbone (Real Integration)
====================================================

Uses the official TIMM implementation.
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


@MODEL_REGISTRY.register("swin")
class SwinBackbone(BaseVisionBackbone):
    """Real implementation of Swin Transformer utilizing TIMM."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        if timm is None:
            raise ImportError("Please install timm.")
            
        self.pretrained_name = cfg.model.get("pretrained_name", "swin_tiny_patch4_window7_224")
        
        self.backbone = timm.create_model(
            self.pretrained_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self._feature_info = self.backbone.feature_info

    def _define_capabilities(self) -> ModelCapability:
        return ModelCapability(
            temporal=False,
            multispectral=False,
            rgb=True,
            variable_resolution=True,
            pretraining=True,
            masking=False,
            embeddings=True,
            decoder=True,
            segmentation=True,
            embedding_dim=768 # For Swin-T
        )

    def load_pretrained(self, checkpoint_path: str | None = None) -> None:
        pass
        
    def unfreeze_last_n_blocks(self, n: int) -> None:
        stages = self.backbone.layers
        for i in range(len(stages) - n, len(stages)):
            for param in stages[i].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_multiscale(x)
        return features[-1]

    def extract_multiscale(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(x)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        f4 = self.forward(x)
        b, c, h, w = f4.shape
        return f4.view(b, c, -1).transpose(1, 2)

    @property
    def feature_dimensions(self) -> list[int]:
        return self._feature_info.channels()

    @property
    def output_stride(self) -> int:
        return max(self._feature_info.reduction())


__all__ = ["SwinBackbone"]
