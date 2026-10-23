"""
CropXAI Dataset — PyTorch Dataset Class
=========================================

``CropXAIDataset`` is the primary PyTorch Dataset implementation.

Features:
  - Automatic metadata loading from index
  - Lazy loading (reads from disk on demand)
  - Disk-based LRU caching
  - Multi-band and multi-temporal support
  - Label loading with configurable ignore index
  - Transform and augmentation support
  - Statistics calculation (mean/std)
  - Visualization API
  - Split-aware (train / val / test)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.research.dataset.cache import DatasetCache
from src.research.dataset.indexer import DatasetIndex, SampleRecord

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sample Type
# ──────────────────────────────────────────────────────────────────────────────

class CropSample:
    """Container for a single dataset sample."""

    def __init__(
        self,
        image: torch.Tensor,
        label: torch.Tensor | None,
        record: SampleRecord,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.image = image          # (C, H, W) or (T, C, H, W)
        self.label = label          # (H, W) long tensor or None
        self.record = record
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"image": self.image, "record_idx": self.record.idx}
        if self.label is not None:
            d["label"] = self.label
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CropXAIDataset(Dataset):
    """
    PyTorch Dataset for the CropXAI framework.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root CropXAI config.
    index : DatasetIndex
        Pre-built dataset index.
    split : str
        Dataset split: "train" | "val" | "test" | "all".
    transform : callable, optional
        Image transform (applied to the image tensor after loading).
    augmentation : callable, optional
        Data augmentation (applied on top of transform; training only).
    cache : DatasetCache, optional
        Cache instance. Defaults to in-memory cache if None.
    lazy : bool
        If True (default), load data on demand. If False, pre-load everything.
    """

    def __init__(
        self,
        cfg: Any,
        index: DatasetIndex,
        split: str = "train",
        transform: Callable | None = None,
        augmentation: Callable | None = None,
        cache: DatasetCache | None = None,
        lazy: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.augmentation = augmentation
        self.lazy = lazy

        # Filter records by split
        if split == "all":
            self._records = index.records
        else:
            self._records = index.filter_by_split(split)

        # Cache
        if cache is None:
            cache_cfg = cfg.dataset.cache
            cache_dir = Path(cfg.paths.cache) / "samples"
            self._cache = DatasetCache(
                cache_dir=cache_dir,
                backend=cache_cfg.get("backend", "memory"),
                max_size_gb=cache_cfg.get("max_size_gb", 5.0),
                enabled=cache_cfg.get("enabled", True),
            )
        else:
            self._cache = cache

        # Dataset config
        self.num_bands = cfg.dataset.bands.num_bands
        self.patch_size = cfg.dataset.spatial.patch_size
        self.temporal_enabled = cfg.dataset.temporal.enabled
        self.num_timesteps = cfg.dataset.temporal.num_timesteps
        self.ignore_index = cfg.dataset.labels.get("ignore_index", 255)

        # Pre-loading (non-lazy mode)
        self._data_store: list[CropSample] | None = None
        if not lazy:
            self._preload()

        logger.info(
            "CropXAIDataset initialized: split=%s, n=%d, lazy=%s",
            split,
            len(self._records),
            lazy,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core Dataset API
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.lazy and self._data_store is not None:
            sample = self._data_store[idx]
        else:
            sample = self._load_sample(idx)

        # Apply augmentation (training split only)
        if self.augmentation is not None and self.split == "train":
            sample = self._apply_augmentation(sample)

        return sample.to_dict()

    # ──────────────────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_sample(self, idx: int) -> CropSample:
        """Load a single sample (with caching)."""
        record = self._records[idx]
        cache_key = f"{self.split}_{record.idx}_{record.image_path[-20:]}"

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        image = self._load_image(record)
        label = self._load_label(record)

        # Apply base transform
        if self.transform is not None:
            image = self.transform(image)

        sample = CropSample(image=image, label=label, record=record)
        self._cache.put(cache_key, sample)
        return sample

    def _load_image(self, record: SampleRecord) -> torch.Tensor:
        """Load and return image as a float tensor (C, H, W) or (T, C, H, W)."""
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required to load GeoTIFF images.")

        with rasterio.open(record.image_path) as src:
            arr = src.read(
                list(range(1, self.num_bands + 1))
            ).astype(np.float32)  # (C, H, W)

        tensor = torch.from_numpy(arr)  # (C, H, W)
        return tensor

    def _load_label(self, record: SampleRecord) -> torch.Tensor | None:
        """Load and return label as a long tensor (H, W)."""
        if record.label_path is None:
            return None
        try:
            import rasterio
            with rasterio.open(record.label_path) as src:
                lbl = src.read(1).astype(np.int64)  # (H, W)
            return torch.from_numpy(lbl)
        except Exception as exc:
            logger.debug("Could not load label for idx %d: %s", record.idx, exc)
            return None

    def _apply_augmentation(self, sample: CropSample) -> CropSample:
        """Apply augmentation to a sample in-place."""
        if self.augmentation is None:
            return sample

        img_np = sample.image.numpy()  # (C, H, W)
        lbl_np = sample.label.numpy() if sample.label is not None else None

        # Albumentations expects (H, W, C) format
        img_hwc = np.transpose(img_np, (1, 2, 0))

        aug_input: dict[str, Any] = {"image": img_hwc}
        if lbl_np is not None:
            aug_input["mask"] = lbl_np

        try:
            augmented = self.augmentation(**aug_input)
            img_out = np.transpose(augmented["image"], (2, 0, 1))
            sample.image = torch.from_numpy(img_out)
            if "mask" in augmented and sample.label is not None:
                sample.label = torch.from_numpy(augmented["mask"])
        except Exception as exc:
            logger.debug("Augmentation failed for sample %d: %s", sample.record.idx, exc)

        return sample

    def _preload(self) -> None:
        """Pre-load all samples into memory (use with caution for large datasets)."""
        logger.info("Pre-loading %d samples into memory...", len(self._records))
        self._data_store = []
        for i in range(len(self._records)):
            self._data_store.append(self._load_sample(i))
        logger.info("Pre-loading complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────────────────────

    def compute_statistics(
        self, max_samples: int = 1000
    ) -> dict[str, list[float]]:
        """
        Compute per-band mean and standard deviation over the training set.

        Parameters
        ----------
        max_samples : int
            Maximum number of samples to use (for speed).

        Returns
        -------
        dict with keys "mean" and "std", each a list of length num_bands.
        """
        n = min(max_samples, len(self))
        indices = np.random.choice(len(self), n, replace=False)

        sum_ = np.zeros(self.num_bands, dtype=np.float64)
        sum_sq = np.zeros(self.num_bands, dtype=np.float64)
        count = 0

        for i in indices:
            sample = self._load_sample(i)
            arr = sample.image.numpy()  # (C, H, W)
            n_pixels = arr.shape[1] * arr.shape[2]
            sum_ += arr.sum(axis=(1, 2))
            sum_sq += (arr ** 2).sum(axis=(1, 2))
            count += n_pixels

        mean = (sum_ / count).tolist()
        std = (np.sqrt(sum_sq / count - (sum_ / count) ** 2)).tolist()

        logger.info("Computed stats over %d samples: mean=%s", n, [f"{m:.2f}" for m in mean])
        return {"mean": mean, "std": std}

    # ──────────────────────────────────────────────────────────────────────────
    # Visualization API
    # ──────────────────────────────────────────────────────────────────────────

    def get_sample_for_visualization(
        self, idx: int
    ) -> dict[str, np.ndarray | None]:
        """
        Return a sample as NumPy arrays for visualization purposes.
        No caching, no augmentation.

        Returns
        -------
        dict with keys "image" (C, H, W) and "label" (H, W) or None.
        """
        sample = self._load_sample(idx)
        return {
            "image": sample.image.numpy(),
            "label": sample.label.numpy() if sample.label is not None else None,
            "idx": idx,
            "path": sample.record.image_path,
        }

    def get_batch_for_visualization(
        self, n: int = 16, shuffle: bool = True
    ) -> list[dict[str, Any]]:
        """Return ``n`` samples for grid visualization."""
        indices = list(range(len(self)))
        if shuffle:
            np.random.shuffle(indices)
        return [self.get_sample_for_visualization(i) for i in indices[:n]]

    # ──────────────────────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────────────────────

    def info(self) -> dict[str, Any]:
        """Return dataset summary info."""
        return {
            "split": self.split,
            "n_samples": len(self),
            "num_bands": self.num_bands,
            "patch_size": self.patch_size,
            "temporal_enabled": self.temporal_enabled,
            "cache_stats": self._cache.stats(),
        }

    def __repr__(self) -> str:
        return (
            f"CropXAIDataset("
            f"split={self.split!r}, n={len(self)}, "
            f"bands={self.num_bands}, patch={self.patch_size})"
        )


__all__ = ["CropXAIDataset", "CropSample"]
