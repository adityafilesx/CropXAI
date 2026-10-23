"""
CropXAI U-Net Decoder
======================

Standard U-Net decoder with skip connections.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import MODEL_REGISTRY
from src.research.models.decoders.base import BaseDecoder


class DecoderBlock(nn.Module):
    """
    A single U-Net decoder block.
    Upsamples the previous feature map, concatenates with the skip connection,
    and applies convolutions.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        # Upsample by factor of 2
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle potential size mismatches (e.g. odd shapes)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


@MODEL_REGISTRY.register("unet_decoder")
class UNetDecoder(BaseDecoder):
    """
    U-Net Decoder processing features from fine to coarse.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int] = (256, 128, 64, 32, 16),
        use_batchnorm: bool = True,
        cfg: Any = None,
    ) -> None:
        super().__init__(encoder_channels, decoder_channels)
        
        # encoder_channels are typically ordered fine to coarse:
        # [c_stage0, c_stage1, c_stage2, c_stage3, c_stage4]
        # We need to process them from coarse to fine in the decoder
        enc_ch = encoder_channels[::-1]
        
        # The first decoder block takes the coarsest feature map
        head_channels = enc_ch[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(enc_ch[1:]) + [0]  # The last block might not have a skip connection
        
        blocks = []
        for i, (in_ch, skip_ch, out_ch) in enumerate(zip(in_channels, skip_channels, decoder_channels)):
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        # Features ordered fine to coarse. Reverse them.
        features = features[::-1]
        x = features[0]
        skips = features[1:]
        
        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
            
        return x

    @property
    def out_channels(self) -> int:
        return self.decoder_channels[-1]


__all__ = ["UNetDecoder"]
