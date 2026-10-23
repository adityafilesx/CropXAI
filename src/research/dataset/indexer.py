"""
CropXAI Dataset Indexer
========================

Builds and maintains a fast index mapping sample indices to:
  - File paths (image + label)
  - Temporal group (multiple timestamps for same location)
  - Spatial tile identifier
  - Class labels (if pre-computed)

The index is persisted as JSON and loaded at dataset init time
to avoid re-scanning the filesystem on every run.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from src.utils.filesystem.io import ensure_dir, save_json, load_json, list_files

logger = logging.getLogger(__name__)


@dataclass
class SampleRecord:
    """Index record for a single dataset sample."""
    idx: int
    image_path: str
    label_path: str | None = None
    tile_id: str | None = None
    temporal_group: str | None = None
    date: str | None = None
    split: str | None = None            # "train" | "val" | "test"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetIndex:
    """
    Fast lookup index for the dataset.

    Parameters
    ----------
    index_path : str | Path
        Path to store/load the index JSON.
    """

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self._records: list[SampleRecord] = []
        self._index_hash: str | None = None

    @property
    def records(self) -> list[SampleRecord]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> SampleRecord:
        return self._records[idx]

    # ──────────────────────────────────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, dataset_dir: str | Path, cfg: Any) -> None:
        """
        Scan ``dataset_dir`` and build the index.

        Parameters
        ----------
        dataset_dir : str | Path
            Directory containing image and label files.
        cfg : OmegaConf DictConfig
            Dataset config.
        """
        import re
        dataset_dir = Path(dataset_dir)

        image_files = sorted(list_files(dataset_dir, pattern="*[!label]*.tif"))
        label_files = sorted(list_files(dataset_dir, pattern="*label*.tif"))

        # Build label lookup: filename pattern matching
        label_lookup = {lf.stem.replace("_label", ""): lf for lf in label_files}

        date_pattern = re.compile(r"(\d{7}T\d{6}|\d{8})")

        self._records = []
        for idx, img_path in enumerate(image_files):
            stem = img_path.stem

            # Find matching label
            label_path = None
            for key, lp in label_lookup.items():
                if key in stem:
                    label_path = lp
                    break

            # Extract date
            date_match = date_pattern.search(img_path.name)
            date_str = date_match.group(0) if date_match else None

            # Extract tile ID (common pattern: T{tile_id})
            tile_match = re.search(r"T(\d{2}[A-Z]{3})", img_path.name)
            tile_id = tile_match.group(1) if tile_match else None

            record = SampleRecord(
                idx=idx,
                image_path=str(img_path),
                label_path=str(label_path) if label_path else None,
                tile_id=tile_id,
                temporal_group=tile_id,   # Samples with same tile_id are temporal pairs
                date=date_str,
            )
            self._records.append(record)

        self._index_hash = self._compute_hash()
        logger.info("Built index with %d samples.", len(self._records))

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the index to disk."""
        ensure_dir(self.index_path.parent)
        data = {
            "version": 1,
            "hash": self._index_hash,
            "records": [r.to_dict() for r in self._records],
        }
        save_json(self.index_path, data)
        logger.info("Index saved → %s (%d records)", self.index_path, len(self._records))

    def load(self) -> bool:
        """
        Load index from disk.

        Returns
        -------
        bool
            True if loaded successfully, False if not found.
        """
        if not self.index_path.exists():
            return False
        try:
            data = load_json(self.index_path)
            self._records = [SampleRecord(**r) for r in data["records"]]
            self._index_hash = data.get("hash")
            logger.info("Loaded index from %s (%d records)", self.index_path, len(self._records))
            return True
        except Exception as exc:
            logger.warning("Failed to load index: %s — rebuilding.", exc)
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Filtering
    # ──────────────────────────────────────────────────────────────────────────

    def filter_by_split(self, split: str) -> list[SampleRecord]:
        """Return records for a specific split."""
        return [r for r in self._records if r.split == split]

    def filter_by_tile(self, tile_id: str) -> list[SampleRecord]:
        """Return records for a specific spatial tile."""
        return [r for r in self._records if r.tile_id == tile_id]

    def assign_splits(self, split_indices: dict[str, list[int]]) -> None:
        """Assign split labels to records from split index files."""
        idx_to_split: dict[int, str] = {}
        for split_name, indices in split_indices.items():
            for idx in indices:
                idx_to_split[idx] = split_name
        for record in self._records:
            record.split = idx_to_split.get(record.idx)

    def _compute_hash(self) -> str:
        paths = "".join(r.image_path for r in self._records)
        return hashlib.sha256(paths.encode()).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Indexer (pipeline stage)
# ──────────────────────────────────────────────────────────────────────────────

class IndexerStage:
    """Build or load the dataset index."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.index_path = Path(cfg.paths.processed) / "dataset_index.json"

    def build_or_load(self, dataset_dir: str | Path) -> DatasetIndex:
        """Build a fresh index or load from cache."""
        index = DatasetIndex(self.index_path)
        if not index.load():
            index.build(dataset_dir, self.cfg)
            index.save()
        return index


__all__ = ["DatasetIndex", "SampleRecord", "IndexerStage"]
