"""
CropXAI DeepLabV3+ ASPP Decoder
================================

Atrous Spatial Pyramid Pooling Decoder.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.registry import MODEL_REGISTRY
from src.research.models.decoders.base import BaseDecoder


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: tuple[int, int, int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


@MODEL_REGISTRY.register("aspp_decoder")
class ASPPDecoder(BaseDecoder):
    """
    DeepLabV3+ ASPP Decoder.
    Takes the deepest feature map for ASPP, and a low-level feature map for skip connection.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: int = 256,
        atrous_rates: tuple[int, int, int] = (12, 24, 36),
        cfg: Any = None,
    ) -> None:
        super().__init__(encoder_channels, decoder_channels)
        
        # Deepest feature map goes to ASPP
        high_level_channels = encoder_channels[-1]
        # Skip connection from a lower level (usually stride 4, index 1 in a 5-stage encoder)
        low_level_channels = encoder_channels[1]

        self.aspp = ASPP(high_level_channels, atrous_rates, decoder_channels)

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(decoder_channels + 48, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        high_level_feat = features[-1]
        low_level_feat = features[1]

        x = self.aspp(high_level_feat)
        
        # Upsample to match low-level feature map
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode="bilinear", align_corners=False)
        
        low_level = self.shortcut_conv(low_level_feat)
        
        x = torch.cat([x, low_level], dim=1)
        x = self.cat_conv(x)
        
        # Usually upsampled to original image size by the head, so we return it here
        return x

    @property
    def out_channels(self) -> int:
        return self.decoder_channels


__all__ = ["ASPPDecoder"]
