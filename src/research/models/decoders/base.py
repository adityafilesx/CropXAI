"""
CropXAI Base Decoder
======================

Abstract base class for all semantic segmentation decoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for Decoders.

    Parameters
    ----------
    encoder_channels : list[int]
        List of channel dimensions from the encoder feature maps.
    decoder_channels : int | list[int]
        Channel dimension(s) for the decoder layers.
    """

    def __init__(self, encoder_channels: list[int], decoder_channels: Any) -> None:
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

    @abstractmethod
    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Process the encoder feature maps into a decoded representation.

        Parameters
        ----------
        features : list[torch.Tensor]
            Multi-scale feature maps from the encoder.
            [ (B, C1, H/2, W/2), ... , (B, C_n, H/32, W/32) ]

        Returns
        -------
        torch.Tensor
            Decoded feature map. Typically shape (B, decoder_channels, H, W)
            or (B, decoder_channels, H/4, W/4).
        """
        ...

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Return the number of output channels from the decoder."""
        ...


__all__ = ["BaseDecoder"]
