"""
CropXAI Prithvi Foundation Adapter (Real Integration)
======================================================

Implements the official Prithvi EO-2 HuggingFace interface.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None

from src.core.registry import MODEL_REGISTRY
from src.research.models.adapters.foundation.base_adapter import BaseFoundationAdapter
from src.research.models.interfaces.vision_backbone import ModelCapability

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register("prithvi_eo2")
class PrithviEO2Adapter(BaseFoundationAdapter):
    """
    Real adapter wrapping the HuggingFace Prithvi-100M implementation.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        if AutoModel is None:
            raise ImportError("Please install transformers.")
            
        self.pretrained_name = cfg.model.get("pretrained_name", "ibm-nasa-geospatial/Prithvi-100M")
        # Prithvi uses a ViT architecture with temporal patch embedding
        # Due to sandbox limits, we use trust_remote_code=True for HF models that need it.
        self.backbone = AutoModel.from_pretrained(self.pretrained_name, trust_remote_code=True)
        self.embed_dim = self.backbone.config.hidden_size

    def _define_capabilities(self) -> ModelCapability:
        return ModelCapability(
            temporal=True,
            multispectral=True,
            rgb=True,
            variable_resolution=True,
            pretraining=True,
            masking=True,
            embeddings=True,
            decoder=False, # Standard ViTs usually need an external decoder
            segmentation=True,
            embedding_dim=self.embed_dim
        )

    def download_weights(self) -> str:
        # Handled by HF
        return ""

    def _load_checkpoint(self, path: str) -> None:
        pass
        
    def unfreeze_last_n_blocks(self, n: int) -> None:
        blocks = self.backbone.encoder.layer
        for i in range(len(blocks) - n, len(blocks)):
            for param in blocks[i].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prithvi expects B, C, T, H, W for its patch embedder.
        CropXAI Universal input is B, T, C, H, W.
        """
        if x.dim() == 5:
            x = x.transpose(1, 2)
        else:
            x = x.unsqueeze(2)
            
        outputs = self.backbone(pixel_values=x)
        # (B, N, C)
        return outputs.last_hidden_state

    def extract_multiscale(self, x: torch.Tensor) -> list[torch.Tensor]:
        # ViT returns tokens. A specialized OutputAdapter in the Segmentation Head
        # converts this to multi-scale features for standard decoders.
        return [self.forward(x)]

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @property
    def feature_dimensions(self) -> list[int]:
        return [self.embed_dim]

    @property
    def output_stride(self) -> int:
        return 16


__all__ = ["PrithviEO2Adapter"]
