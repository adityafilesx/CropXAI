"""
CropXAI Normalization
======================

Implements per-band normalization strategies:
  - Min-Max scaling to [0, 1]
  - Z-score standardization (μ=0, σ=1)
  - Percentile-based robust normalization
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class Normalizer:
    """
    Base class for all normalizers.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Normalization config node.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def __call__(self, image: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Normalize a (C, H, W) tensor or array."""
        raise NotImplementedError


class MinMaxNormalizer(Normalizer):
    """Scale pixel values to [0, 1] using global min/max."""

    def __init__(self, cfg: Any, min_val: float = 0.0, max_val: float = 10000.0) -> None:
        super().__init__(cfg)
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, image: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32))
        image = image.float()
        if self.cfg.get("clip", True):
            image = torch.clamp(image, self.min_val, self.max_val)
        normalized = (image - self.min_val) / max(self.max_val - self.min_val, 1e-8)
        return normalized.clamp(0.0, 1.0)


class ZScoreNormalizer(Normalizer):
    """Standardize using per-band mean and std."""

    def __init__(self, cfg: Any, mean: list[float], std: list[float]) -> None:
        super().__init__(cfg)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32))
        image = image.float()
        return (image - self.mean) / (self.std + 1e-8)


class PercentileNormalizer(Normalizer):
    """Robust normalization using per-image percentile clipping."""

    def __init__(self, cfg: Any, low: float = 2.0, high: float = 98.0) -> None:
        super().__init__(cfg)
        self.low = low
        self.high = high

    def __call__(self, image: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.astype(np.float32))
        arr = image.float()
        flat = arr.view(arr.shape[0], -1)
        p_low = torch.quantile(flat, self.low / 100.0, dim=1).view(-1, 1, 1)
        p_high = torch.quantile(flat, self.high / 100.0, dim=1).view(-1, 1, 1)
        normalized = (arr - p_low) / (p_high - p_low + 1e-8)
        return normalized.clamp(0.0, 1.0)


def build_normalizer(cfg: Any) -> Normalizer | None:
    """Factory function to build a normalizer from config."""
    norm_cfg = cfg.preprocessing.normalization
    if not norm_cfg.get("enabled", True):
        return None

    method = norm_cfg.get("method", "minmax")

    if method == "minmax":
        return MinMaxNormalizer(
            norm_cfg,
            min_val=norm_cfg.get("min_val", 0.0),
            max_val=norm_cfg.get("max_val", 10000.0),
        )
    elif method == "zscore":
        mean = norm_cfg.get("mean") or cfg.dataset.statistics.get("mean") or [0.0]
        std = norm_cfg.get("std") or cfg.dataset.statistics.get("std") or [1.0]
        return ZScoreNormalizer(norm_cfg, mean=mean, std=std)
    elif method == "percentile":
        return PercentileNormalizer(norm_cfg)
    elif method == "none":
        return None
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")


__all__ = [
    "Normalizer",
    "MinMaxNormalizer",
    "ZScoreNormalizer",
    "PercentileNormalizer",
    "build_normalizer",
]
