"""
CropXAI FPN Decoder
====================

Feature Pyramid Network Decoder for semantic segmentation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import MODEL_REGISTRY
from src.research.models.decoders.base import BaseDecoder


class FPNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        skip = self.skip_conv(skip)
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        return x + skip


class FPNSmoothing(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


@MODEL_REGISTRY.register("fpn_decoder")
class FPNDecoder(BaseDecoder):
    """
    Feature Pyramid Network (FPN) Decoder.
    Takes 4 feature maps from the encoder (usually strides 4, 8, 16, 32).
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: int = 256,
        dropout: float = 0.2,
        cfg: Any = None,
    ) -> None:
        super().__init__(encoder_channels, decoder_channels)
        
        # We usually use the last 4 feature maps
        if len(encoder_channels) < 4:
            raise ValueError("FPNDecoder requires at least 4 feature maps from the encoder.")
            
        enc_ch = encoder_channels[-4:]
        
        self.p5_conv = nn.Conv2d(enc_ch[3], decoder_channels, kernel_size=1)
        self.p4_block = FPNBlock(enc_ch[2], decoder_channels)
        self.p3_block = FPNBlock(enc_ch[1], decoder_channels)
        self.p2_block = FPNBlock(enc_ch[0], decoder_channels)
        
        self.smooth = nn.ModuleList([FPNSmoothing(decoder_channels, decoder_channels) for _ in range(4)])
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5 = features[-4:]
        
        p5 = self.p5_conv(c5)
        p4 = self.p4_block(p5, c4)
        p3 = self.p3_block(p4, c3)
        p2 = self.p2_block(p3, c2)
        
        p5 = self.smooth[3](p5)
        p4 = self.smooth[2](p4)
        p3 = self.smooth[1](p3)
        p2 = self.smooth[0](p2)
        
        # Upsample all to the largest resolution (p2)
        size = p2.shape[2:]
        p5 = F.interpolate(p5, size=size, mode="bilinear", align_corners=False)
        p4 = F.interpolate(p4, size=size, mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=size, mode="bilinear", align_corners=False)
        
        # Aggregate
        out = p5 + p4 + p3 + p2
        return self.dropout(out)

    @property
    def out_channels(self) -> int:
        return self.decoder_channels


__all__ = ["FPNDecoder"]
