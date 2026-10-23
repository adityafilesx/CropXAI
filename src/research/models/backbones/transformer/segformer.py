"""
CropXAI SegFormer Backbone (Real Integration)
==============================================

Uses the official HuggingFace transformers implementation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

try:
    from transformers import SegformerModel
except ImportError:
    SegformerModel = None

from src.core.registry import MODEL_REGISTRY
from src.research.models.interfaces.vision_backbone import BaseVisionBackbone, ModelCapability

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("segformer")
class SegFormerBackbone(BaseVisionBackbone):
    """Real implementation of SegFormer leveraging transformers library."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        if SegformerModel is None:
            raise ImportError("Please install transformers: pip install transformers")
            
        self.pretrained_name = cfg.model.get("pretrained_name", "nvidia/mit-b2")
        self.backbone = SegformerModel.from_pretrained(self.pretrained_name)

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
            embedding_dim=512  # For mit-b2
        )

    def load_pretrained(self, checkpoint_path: str | None = None) -> None:
        # HuggingFace handles pretraining in __init__ via from_pretrained.
        pass
        
    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Unfreeze the last n stages of SegFormer."""
        stages = self.backbone.encoder.block
        for i in range(len(stages) - n, len(stages)):
            for param in stages[i].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_multiscale(x)
        return features[-1]

    def extract_multiscale(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.backbone(x, output_hidden_states=True)
        # SegFormer returns features of shape (B, C, H, W) natively in hidden_states
        return list(outputs.hidden_states)

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        f4 = self.forward(x)  # (B, C, H, W)
        b, c, h, w = f4.shape
        return f4.view(b, c, -1).transpose(1, 2)  # (B, N, C)

    @property
    def feature_dimensions(self) -> list[int]:
        # Dimensions for mit-b2
        return [64, 128, 320, 512]

    @property
    def output_stride(self) -> int:
        return 32


__all__ = ["SegFormerBackbone"]
