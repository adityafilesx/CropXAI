"""
CropXAI Missing Value Handler
================================

Handles NaN, NoData, and invalid pixel values in multi-band imagery.
"""

from __future__ import annotations
import logging
from typing import Any
import numpy as np
import torch

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """
    Handle missing (NaN / NoData) values in image tensors.

    Parameters
    ----------
    strategy : str
        "interpolate" | "mask" | "zero" | "mean"
    nodata_value : float | None
        Sentinel value representing NoData (e.g. 0 or -9999).
    max_missing_fraction : float
        Fraction of missing pixels above which the sample is flagged.
    """

    def __init__(
        self,
        strategy: str = "zero",
        nodata_value: float | None = None,
        max_missing_fraction: float = 0.1,
    ) -> None:
        self.strategy = strategy
        self.nodata_value = nodata_value
        self.max_missing_fraction = max_missing_fraction

    @classmethod
    def from_cfg(cls, cfg: Any) -> "MissingValueHandler":
        mv_cfg = cfg.preprocessing.missing_values
        return cls(
            strategy=mv_cfg.get("strategy", "zero"),
            max_missing_fraction=mv_cfg.get("max_missing_fraction", 0.1),
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Handle missing values in a (C, H, W) tensor."""
        mask = torch.isnan(image)

        if self.nodata_value is not None:
            mask = mask | (image == self.nodata_value)

        missing_frac = float(mask.float().mean())
        if missing_frac > self.max_missing_fraction:
            logger.debug(
                "Sample has high missing fraction: %.2f%% (threshold: %.2f%%)",
                missing_frac * 100,
                self.max_missing_fraction * 100,
            )

        if not mask.any():
            return image

        if self.strategy == "zero":
            return image.masked_fill(mask, 0.0)
        elif self.strategy == "mean":
            per_band_mean = image.view(image.shape[0], -1).nanmean(dim=1)
            out = image.clone()
            for c in range(image.shape[0]):
                out[c][mask[c]] = per_band_mean[c].item()
            return out
        else:
            return image.masked_fill(mask, 0.0)


__all__ = ["MissingValueHandler"]
