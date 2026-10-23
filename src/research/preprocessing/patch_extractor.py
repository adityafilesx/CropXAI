"""CropXAI Patch Extractor — wraps PatchGenerator as a callable transform."""

from __future__ import annotations
from typing import Any
import torch
import numpy as np
from src.research.dataset.patch_generator import PatchGenerator


class PatchExtractor:
    """Extract a random patch from an image tensor during transform."""

    def __init__(self, patch_size: int = 224, strategy: str = "random") -> None:
        self.generator = PatchGenerator(patch_size=patch_size, strategy=strategy, n_random_patches=1)

    @classmethod
    def from_cfg(cls, cfg: Any) -> "PatchExtractor":
        return cls(
            patch_size=cfg.dataset.spatial.patch_size,
            strategy=cfg.preprocessing.patch_extraction.get("strategy", "random"),
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        arr = image.numpy()  # (C, H, W)
        patches = self.generator.extract_from_array(arr)
        if patches:
            return torch.from_numpy(patches[0].data)
        return image


__all__ = ["PatchExtractor"]
