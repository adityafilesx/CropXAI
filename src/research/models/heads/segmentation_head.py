"""
CropXAI Segmentation Head
==========================

Projects decoder features to class logits and upsamples to original resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Standard Segmentation Head.

    Parameters
    ----------
    in_channels : int
        Number of channels coming from the decoder.
    num_classes : int
        Number of segmentation classes.
    kernel_size : int
        Kernel size for the projection convolution. Default 3.
    upsampling : int
        Factor by which to upsample the output to match target resolution.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernel_size: int = 3,
        upsampling: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, num_classes, kernel_size=kernel_size, padding=padding)
        self.upsampling = upsampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        if self.upsampling > 1:
            x = F.interpolate(x, scale_factor=self.upsampling, mode="bilinear", align_corners=False)
        return x


__all__ = ["SegmentationHead"]
