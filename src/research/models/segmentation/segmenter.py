"""
CropXAI Semantic Segmenter
===========================

The master assembly module that connects an Encoder, Decoder, and Head.
Inherits from BaseModel to fulfill all abstract hooks required for
parameter counting, XAI feature extraction, and standardized forward passes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import MODEL_REGISTRY
from src.research.models.base_model import BaseModel
from src.research.models.backbones.base import BaseEncoder
from src.research.models.decoders.base import BaseDecoder
from src.research.models.heads.segmentation_head import SegmentationHead


@MODEL_REGISTRY.register("semantic_segmenter")
class SemanticSegmenter(BaseModel):
    """
    Modular Semantic Segmentation Model.
    Assembles interchangeable Encoders and Decoders into a unified architecture.

    Parameters
    ----------
    cfg : Any
        Root configuration node.
    encoder : BaseEncoder
        The initialized backbone (e.g. TimmEncoder).
    decoder : BaseDecoder
        The initialized decoder (e.g. UNetDecoder, ASPPDecoder).
    upsampling : int
        Final upsampling factor to match target image size.
        (DeepLab often requires upsampling=4 or 8).
    """

    def __init__(
        self,
        cfg: Any,
        encoder: BaseEncoder,
        decoder: BaseDecoder,
        upsampling: int = 1,
    ) -> None:
        super().__init__(cfg)
        self.encoder = encoder
        self.decoder = decoder
        
        # Instantiate standard head based on decoder output channels
        self.head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            num_classes=self.num_classes,
            kernel_size=3,
            upsampling=upsampling,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, C, H, W).
            
        Returns
        -------
        torch.Tensor
            Class logits (B, num_classes, H, W).
        """
        input_shape = x.shape[-2:]
        
        # 1. Extract multi-scale features
        features = self.encoder(x)
        
        # 2. Decode features
        decoded = self.decoder(features)
        
        # 3. Project to logits
        logits = self.head(decoded)
        
        # Ensure exact shape match (handle rounding errors from striding)
        if logits.shape[-2:] != input_shape:
            logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
            
        return logits

    # ─── XAI HOOKS (Phase 4 Preparedness) ─────────────────────────────────

    def get_intermediate_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose all intermediate features for Explainability."""
        features = self.encoder(x)
        decoded = self.decoder(features)
        return {
            "encoder_features": features,
            "decoder_features": decoded,
        }

    def extract_encoder(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Explicitly run only the encoder."""
        return self.encoder(x)
        
    def extract_decoder(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Explicitly run only the decoder."""
        return self.decoder(features)

    def freeze_encoder(self) -> None:
        """Freeze the backbone for transfer learning/finetuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze the backbone."""
        for param in self.encoder.parameters():
            param.requires_grad = True


__all__ = ["SemanticSegmenter"]
