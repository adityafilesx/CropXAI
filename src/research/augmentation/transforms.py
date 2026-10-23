"""
CropXAI Custom Transforms
==========================

Custom Albumentations-compatible transforms for multi-band remote sensing data.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class SpectralJitter:
    """
    Randomly perturb per-band radiometric values (spectral jitter).
    Simulates atmospheric variation across acquisitions.

    Compatible with Albumentations-style ``__call__(image=..., ...)`` API.
    """

    def __init__(self, scale: float = 0.05, p: float = 0.3) -> None:
        self.scale = scale
        self.p = p

    def __call__(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        if np.random.random() < self.p:
            # Apply random per-band multiplicative factor
            c = image.shape[-1]  # (H, W, C) in Albumentations
            factors = np.random.normal(1.0, self.scale, size=c).astype(np.float32)
            image = np.clip(image * factors, 0, 1)
        return {"image": image, **kwargs}


class TemporalDropout:
    """
    Randomly zero out one temporal frame in a multi-temporal sequence.
    Simulates missing acquisitions due to cloud cover.
    """

    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        sequence : np.ndarray
            Shape (T, H, W, C).
        """
        if np.random.random() < self.p and len(sequence) > 1:
            drop_idx = np.random.randint(0, len(sequence))
            sequence = sequence.copy()
            sequence[drop_idx] = 0.0
        return sequence


__all__ = ["SpectralJitter", "TemporalDropout"]
