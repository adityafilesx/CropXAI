"""
CropXAI Dataset Analyzer
==========================

Computes comprehensive dataset statistics:
  - Class distribution (pixel-level and sample-level)
  - Per-band spectral statistics (mean, std, min, max, percentiles)
  - Temporal information (acquisition dates, gap analysis)
  - Missing value analysis
  - Spatial coverage statistics
  - Inter-class imbalance metrics
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.core.pipeline import PipelineStage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BandStatistics:
    """Per-band statistics computed over the full dataset."""
    band_index: int
    band_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    p1: float       # 1st percentile
    p99: float      # 99th percentile
    missing_fraction: float


@dataclass
class ClassStatistics:
    """Statistics for a single class."""
    class_id: int
    class_name: str
    pixel_count: int
    sample_count: int
    pixel_fraction: float
    sample_fraction: float


@dataclass
class DatasetStatistics:
    """Full dataset statistics summary."""
    dataset_name: str
    total_samples: int
    total_pixels: int
    num_classes: int
    class_stats: list[ClassStatistics] = field(default_factory=list)
    band_stats: list[BandStatistics] = field(default_factory=list)
    missing_fraction_overall: float = 0.0
    imbalance_ratio: float = 1.0        # max_class / min_class pixel count
    temporal_dates: list[str] = field(default_factory=list)
    temporal_gaps_days: list[float] = field(default_factory=list)

    def get_class_weights(self) -> list[float]:
        """Compute inverse-frequency class weights for loss reweighting."""
        if not self.class_stats:
            return []
        total = sum(c.pixel_count for c in self.class_stats)
        weights = []
        for c in self.class_stats:
            freq = c.pixel_count / total if total > 0 else 1.0
            weights.append(1.0 / (freq + 1e-8))
        # Normalize
        w_sum = sum(weights)
        return [w / w_sum * len(weights) for w in weights]


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class DatasetAnalyzer(PipelineStage):
    """
    Stage 3: Analyze dataset statistics.

    Computes class distributions, band statistics, temporal info,
    and missing value analysis.
    """

    def __init__(self, cfg: Any, max_samples: int | None = 500) -> None:
        super().__init__(name="Analyze")
        self.cfg = cfg
        self.max_samples = max_samples  # Limit for expensive per-pixel stats

    def process(self, data: Any) -> Any:
        from src.research.dataset.downloader import DownloadContext

        ctx = data if isinstance(data, DownloadContext) else data

        if not hasattr(ctx, "dataset_dir") or not ctx.dataset_dir.exists():
            logger.warning("Analyzer: dataset_dir not found, skipping analysis.")
            return ctx

        from src.utils.filesystem.io import list_files
        image_files = list_files(ctx.dataset_dir, pattern="*.tif")
        label_files = list_files(ctx.dataset_dir, pattern="*label*.tif")

        logger.info(
            "Analyzing %d image files, %d label files...",
            len(image_files),
            len(label_files),
        )

        stats = self._compute_statistics(image_files, label_files)
        ctx.metadata["dataset_statistics"] = stats
        logger.info(
            "Analysis complete. %d samples, %d classes, imbalance ratio: %.2f",
            stats.total_samples,
            stats.num_classes,
            stats.imbalance_ratio,
        )
        return ctx

    def _compute_statistics(
        self,
        image_files: list[Path],
        label_files: list[Path],
    ) -> DatasetStatistics:
        """Compute full dataset statistics."""
        cfg = self.cfg

        stats = DatasetStatistics(
            dataset_name=cfg.dataset.name,
            total_samples=len(image_files),
            total_pixels=0,
            num_classes=cfg.dataset.labels.num_classes,
        )

        # Analyze band statistics (sample up to max_samples)
        sample_images = image_files[: self.max_samples] if self.max_samples else image_files
        band_stats = self._compute_band_statistics(sample_images)
        stats.band_stats = band_stats

        # Analyze class distribution
        if label_files:
            class_stats = self._compute_class_distribution(label_files)
            stats.class_stats = class_stats
            stats.total_pixels = sum(c.pixel_count for c in class_stats)
            if class_stats:
                max_count = max(c.pixel_count for c in class_stats)
                min_count = min(c.pixel_count for c in class_stats if c.pixel_count > 0)
                stats.imbalance_ratio = max_count / max(min_count, 1)

        # Temporal analysis
        stats.temporal_dates = self._extract_temporal_dates(image_files)
        stats.temporal_gaps_days = self._compute_temporal_gaps(stats.temporal_dates)

        return stats

    def _compute_band_statistics(self, files: list[Path]) -> list[BandStatistics]:
        """Compute per-band statistics from a sample of files."""
        try:
            import rasterio
        except ImportError:
            logger.warning("rasterio not available — skipping band statistics.")
            return []

        band_names = list(self.cfg.dataset.bands.selected or [])
        num_bands = self.cfg.dataset.bands.num_bands

        # Accumulate per-band values
        band_values: dict[int, list[np.ndarray]] = {i: [] for i in range(num_bands)}

        for f in files:
            try:
                with rasterio.open(f) as src:
                    for b in range(min(src.count, num_bands)):
                        arr = src.read(b + 1).astype(np.float32)
                        # Subsample large arrays
                        if arr.size > 10000:
                            flat = arr.ravel()
                            idx = np.random.choice(len(flat), 10000, replace=False)
                            arr = flat[idx]
                        band_values[b].append(arr.ravel())
            except Exception as exc:
                logger.debug("Skipping file in band stats: %s", exc)

        band_stats_list = []
        for b_idx in range(num_bands):
            vals = band_values.get(b_idx, [])
            if not vals:
                continue
            all_vals = np.concatenate(vals).astype(np.float32)
            missing = np.isnan(all_vals).mean()
            all_vals = all_vals[~np.isnan(all_vals)]

            if len(all_vals) == 0:
                continue

            band_name = band_names[b_idx] if b_idx < len(band_names) else f"Band_{b_idx + 1}"
            band_stats_list.append(BandStatistics(
                band_index=b_idx,
                band_name=band_name,
                mean=float(np.mean(all_vals)),
                std=float(np.std(all_vals)),
                min_val=float(np.min(all_vals)),
                max_val=float(np.max(all_vals)),
                p1=float(np.percentile(all_vals, 1)),
                p99=float(np.percentile(all_vals, 99)),
                missing_fraction=float(missing),
            ))

        return band_stats_list

    def _compute_class_distribution(self, label_files: list[Path]) -> list[ClassStatistics]:
        """Count pixel occurrences for each class across all label files."""
        try:
            import rasterio
        except ImportError:
            return []

        class_names = list(self.cfg.dataset.labels.class_names)
        num_classes = self.cfg.dataset.labels.num_classes
        pixel_counts: Counter[int] = Counter()
        sample_counts: Counter[int] = Counter()
        ignore_index = 255

        sample_files = label_files[: self.max_samples] if self.max_samples else label_files

        for f in sample_files:
            try:
                with rasterio.open(f) as src:
                    arr = src.read(1)
                    valid = arr[arr != ignore_index]
                    classes_in_sample = set(np.unique(valid))
                    for cls in classes_in_sample:
                        if 0 <= cls < num_classes:
                            pixel_counts[int(cls)] += int(np.sum(valid == cls))
                            sample_counts[int(cls)] += 1
            except Exception as exc:
                logger.debug("Skipping label file: %s", exc)

        total_pixels = max(sum(pixel_counts.values()), 1)
        total_samples = max(sum(sample_counts.values()), 1)

        class_stats = []
        for cls_id in range(num_classes):
            name = class_names[cls_id] if cls_id < len(class_names) else f"Class_{cls_id}"
            px = pixel_counts.get(cls_id, 0)
            sc = sample_counts.get(cls_id, 0)
            class_stats.append(ClassStatistics(
                class_id=cls_id,
                class_name=name,
                pixel_count=px,
                sample_count=sc,
                pixel_fraction=px / total_pixels,
                sample_fraction=sc / total_samples,
            ))

        return class_stats

    def _extract_temporal_dates(self, files: list[Path]) -> list[str]:
        """Extract and sort unique acquisition dates from filenames."""
        import re
        pattern = re.compile(r"(\d{7}T\d{6}|\d{8})")
        dates: set[str] = set()
        for f in files:
            m = pattern.search(f.name)
            if m:
                dates.add(m.group(0))
        return sorted(dates)

    def _compute_temporal_gaps(self, dates: list[str]) -> list[float]:
        """Compute day-gaps between consecutive temporal acquisitions."""
        if len(dates) < 2:
            return []

        from datetime import datetime
        parsed = []
        for d in dates:
            for fmt in ("%Y%j", "%Y%m%d"):
                try:
                    parsed.append(datetime.strptime(d[:len(fmt.replace("%", "").replace("j", "000"))], fmt))
                    break
                except ValueError:
                    pass

        parsed.sort()
        gaps = []
        for i in range(1, len(parsed)):
            delta = (parsed[i] - parsed[i - 1]).days
            gaps.append(float(delta))
        return gaps


__all__ = [
    "DatasetAnalyzer",
    "DatasetStatistics",
    "ClassStatistics",
    "BandStatistics",
]
