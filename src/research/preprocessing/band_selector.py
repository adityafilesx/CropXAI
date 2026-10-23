"""
CropXAI Band Selector
======================

Select and reorder spectral bands from multi-band imagery.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BandSelector:
    """
    Select a subset of bands from a multi-band image tensor.

    Parameters
    ----------
    band_indices : list[int]
        0-based indices of bands to select.
    """

    def __init__(self, band_indices: list[int]) -> None:
        self.band_indices = band_indices

    @classmethod
    def from_cfg(cls, cfg: Any) -> "BandSelector":
        """Build from config using dataset band names."""
        all_bands = list(cfg.dataset.bands.all)
        selected = list(cfg.dataset.bands.selected)
        indices = [all_bands.index(b) for b in selected if b in all_bands]
        return cls(indices)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Select bands from image tensor.

        Parameters
        ----------
        image : torch.Tensor
            Shape (C, H, W) or (T, C, H, W).
        """
        if image.dim() == 3:
            return image[self.band_indices]
        elif image.dim() == 4:
            return image[:, self.band_indices]
        raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")


__all__ = ["BandSelector"]
