"""
CropXAI Patch Generator
=========================

Extracts spatial patches from large GeoTIFF images using either:
  - Random sampling (training)
  - Sliding window grid (inference / full coverage)
  - Center crop (evaluation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Patch:
    """A single extracted patch."""
    data: np.ndarray        # Shape: (C, H, W) or (T, C, H, W) for temporal
    label: np.ndarray | None  # Shape: (H, W)
    row_start: int
    col_start: int
    source_file: str
    temporal_index: int | None = None


class PatchGenerator:
    """
    Generates patches from large GeoTIFF images.

    Parameters
    ----------
    patch_size : int
        Output patch size (pixels). Default 224.
    overlap : int
        Overlap between adjacent patches for sliding window. Default 0.
    strategy : str
        "random" | "sliding_window" | "center"
    n_random_patches : int
        Number of patches per image when strategy="random".
    """

    def __init__(
        self,
        patch_size: int = 224,
        overlap: int = 0,
        strategy: str = "random",
        n_random_patches: int = 16,
        seed: int = 42,
    ) -> None:
        self.patch_size = patch_size
        self.overlap = overlap
        self.strategy = strategy
        self.n_random_patches = n_random_patches
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_cfg(cls, cfg: Any) -> "PatchGenerator":
        """Build from Hydra config."""
        spatial_cfg = cfg.dataset.spatial
        preproc_cfg = cfg.preprocessing.patch_extraction
        return cls(
            patch_size=spatial_cfg.patch_size,
            overlap=spatial_cfg.get("overlap", 0),
            strategy=preproc_cfg.get("strategy", "random"),
        )

    def extract_from_array(
        self, image: np.ndarray, label: np.ndarray | None = None
    ) -> list[Patch]:
        """
        Extract patches from an in-memory array.

        Parameters
        ----------
        image : np.ndarray
            Shape: (C, H, W).
        label : np.ndarray | None
            Shape: (H, W).

        Returns
        -------
        list[Patch]
        """
        _, h, w = image.shape
        if self.strategy == "random":
            return self._random_patches(image, label, h, w)
        elif self.strategy == "sliding_window":
            return list(self._sliding_window_patches(image, label, h, w))
        elif self.strategy == "center":
            return [self._center_patch(image, label, h, w)]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def extract_from_file(self, path: str | Path, label_path: str | Path | None = None) -> list[Patch]:
        """Read a GeoTIFF file and extract patches."""
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for patch extraction from files.")

        p = Path(path)
        with rasterio.open(p) as src:
            image = src.read().astype(np.float32)  # (C, H, W)

        label = None
        if label_path is not None:
            with rasterio.open(label_path) as src:
                label = src.read(1)

        patches = self.extract_from_array(image, label)
        for patch in patches:
            patch.source_file = str(p)
        return patches

    def _random_patches(
        self, image: np.ndarray, label: np.ndarray | None, h: int, w: int
    ) -> list[Patch]:
        ps = self.patch_size
        patches = []
        max_row = max(h - ps, 0)
        max_col = max(w - ps, 0)

        for _ in range(self.n_random_patches):
            r = int(self.rng.integers(0, max_row + 1)) if max_row > 0 else 0
            c = int(self.rng.integers(0, max_col + 1)) if max_col > 0 else 0
            img_patch = image[:, r:r + ps, c:c + ps]
            lbl_patch = label[r:r + ps, c:c + ps] if label is not None else None

            if img_patch.shape[1] == ps and img_patch.shape[2] == ps:
                patches.append(Patch(
                    data=img_patch,
                    label=lbl_patch,
                    row_start=r,
                    col_start=c,
                    source_file="",
                ))
        return patches

    def _sliding_window_patches(
        self, image: np.ndarray, label: np.ndarray | None, h: int, w: int
    ) -> Generator[Patch, None, None]:
        ps = self.patch_size
        stride = ps - self.overlap
        for r in range(0, h - ps + 1, stride):
            for c in range(0, w - ps + 1, stride):
                img_patch = image[:, r:r + ps, c:c + ps]
                lbl_patch = label[r:r + ps, c:c + ps] if label is not None else None
                yield Patch(
                    data=img_patch,
                    label=lbl_patch,
                    row_start=r,
                    col_start=c,
                    source_file="",
                )

    def _center_patch(
        self, image: np.ndarray, label: np.ndarray | None, h: int, w: int
    ) -> Patch:
        ps = self.patch_size
        r = max((h - ps) // 2, 0)
        c = max((w - ps) // 2, 0)
        img_patch = image[:, r:r + ps, c:c + ps]
        lbl_patch = label[r:r + ps, c:c + ps] if label is not None else None
        return Patch(data=img_patch, label=lbl_patch, row_start=r, col_start=c, source_file="")


__all__ = ["PatchGenerator", "Patch"]
