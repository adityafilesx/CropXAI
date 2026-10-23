"""
CropXAI Dataset Metadata Reader
=================================

Reads and parses GeoTIFF metadata using rasterio:
  - Coordinate Reference System (CRS)
  - Spatial resolution (meters/pixel)
  - Bounding box (geographic extent)
  - Band descriptions and data types
  - Temporal information (from filename patterns)
  - File size and compression info
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BandMetadata:
    """Metadata for a single spectral band."""
    index: int
    name: str
    dtype: str
    nodata: float | None
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None


@dataclass
class FileMetadata:
    """Complete metadata for a single GeoTIFF file."""
    path: str
    filename: str
    width: int
    height: int
    num_bands: int
    dtype: str
    crs: str
    crs_epsg: int | None
    transform: list[float]          # Affine transform coefficients
    resolution_x: float             # Meters per pixel (x)
    resolution_y: float             # Meters per pixel (y)
    bounds: dict[str, float]        # left, bottom, right, top
    nodata: float | None
    compression: str | None
    file_size_mb: float
    bands: list[BandMetadata] = field(default_factory=list)
    temporal_date: str | None = None  # Parsed from filename if available

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetMetadata:
    """Aggregated metadata for the entire dataset."""
    dataset_name: str
    total_files: int
    total_size_gb: float
    unique_crs: list[str]
    unique_resolutions: list[float]
    bounds_union: dict[str, float]
    num_bands: int
    dtype: str
    temporal_range: tuple[str | None, str | None]
    files: list[FileMetadata] = field(default_factory=list)
    class_distribution: dict[str, int] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────────────────────────────────────

class MetadataReader:
    """
    Reads and aggregates metadata from a collection of GeoTIFF files.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root CropXAI config.
    max_files : int | None
        If set, read at most this many files (useful for quick checks).
    """

    # Regex to extract dates from common HLS/Sentinel filename patterns
    # e.g. HLS.S30.T19TBG.2022001T145729.v2.0.B02.tif
    DATE_PATTERN = re.compile(r"(\d{7}T\d{6}|\d{8})")

    def __init__(self, cfg: Any, max_files: int | None = None) -> None:
        self.cfg = cfg
        self.max_files = max_files

    def read_file(self, path: str | Path) -> FileMetadata | None:
        """Read metadata from a single GeoTIFF file."""
        try:
            import rasterio
        except ImportError:
            logger.error("rasterio is required for metadata reading.")
            return None

        p = Path(path)
        try:
            with rasterio.open(p) as src:
                bounds = src.bounds
                transform = list(src.transform)[:6]

                # Parse date from filename
                date_match = self.DATE_PATTERN.search(p.name)
                temporal_date = date_match.group(0) if date_match else None

                # Band metadata (quick — no data read)
                bands = []
                for i in range(1, src.count + 1):
                    bands.append(BandMetadata(
                        index=i,
                        name=src.descriptions[i - 1] or f"Band_{i}",
                        dtype=str(src.dtypes[i - 1]),
                        nodata=src.nodata,
                    ))

                return FileMetadata(
                    path=str(p),
                    filename=p.name,
                    width=src.width,
                    height=src.height,
                    num_bands=src.count,
                    dtype=str(src.dtypes[0]),
                    crs=str(src.crs) if src.crs else "Unknown",
                    crs_epsg=src.crs.to_epsg() if src.crs else None,
                    transform=transform,
                    resolution_x=abs(src.transform.a),
                    resolution_y=abs(src.transform.e),
                    bounds={
                        "left": bounds.left,
                        "bottom": bounds.bottom,
                        "right": bounds.right,
                        "top": bounds.top,
                    },
                    nodata=src.nodata,
                    compression=src.compression.value if src.compression else None,
                    file_size_mb=p.stat().st_size / 1e6,
                    bands=bands,
                    temporal_date=temporal_date,
                )
        except Exception as exc:
            logger.warning("Could not read metadata from %s: %s", p.name, exc)
            return None

    def read_dataset(self, dataset_dir: str | Path) -> DatasetMetadata:
        """
        Read and aggregate metadata from all GeoTIFF files in a directory.

        Parameters
        ----------
        dataset_dir : str | Path
            Directory to scan for .tif files.

        Returns
        -------
        DatasetMetadata
        """
        from src.utils.filesystem.io import list_files

        files = list_files(dataset_dir, pattern="*.tif")
        if self.max_files:
            files = files[: self.max_files]

        logger.info("Reading metadata from %d files...", len(files))
        file_metas: list[FileMetadata] = []

        for f in files:
            meta = self.read_file(f)
            if meta is not None:
                file_metas.append(meta)

        return self._aggregate(file_metas)

    def _aggregate(self, file_metas: list[FileMetadata]) -> DatasetMetadata:
        """Compute dataset-level statistics from individual file metadata."""
        if not file_metas:
            return DatasetMetadata(
                dataset_name=self.cfg.dataset.name,
                total_files=0,
                total_size_gb=0.0,
                unique_crs=[],
                unique_resolutions=[],
                bounds_union={},
                num_bands=0,
                dtype="unknown",
                temporal_range=(None, None),
            )

        crss = sorted({m.crs for m in file_metas})
        resolutions = sorted({round(m.resolution_x, 2) for m in file_metas})
        total_size = sum(m.file_size_mb for m in file_metas) / 1024.0  # GB

        # Compute union bounds
        lefts = [m.bounds["left"] for m in file_metas]
        bottoms = [m.bounds["bottom"] for m in file_metas]
        rights = [m.bounds["right"] for m in file_metas]
        tops = [m.bounds["top"] for m in file_metas]

        bounds_union = {
            "left": min(lefts),
            "bottom": min(bottoms),
            "right": max(rights),
            "top": max(tops),
        }

        # Temporal range
        dates = sorted(
            m.temporal_date for m in file_metas if m.temporal_date is not None
        )
        temporal_range = (dates[0], dates[-1]) if dates else (None, None)

        return DatasetMetadata(
            dataset_name=self.cfg.dataset.name,
            total_files=len(file_metas),
            total_size_gb=round(total_size, 3),
            unique_crs=crss,
            unique_resolutions=resolutions,
            bounds_union=bounds_union,
            num_bands=file_metas[0].num_bands,
            dtype=file_metas[0].dtype,
            temporal_range=temporal_range,
            files=file_metas,
        )


__all__ = [
    "MetadataReader",
    "DatasetMetadata",
    "FileMetadata",
    "BandMetadata",
]
