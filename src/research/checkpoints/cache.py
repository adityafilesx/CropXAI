"""
CropXAI Checkpoint Cache
==========================

Handles local caching of remote foundation model checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
import hashlib
import os
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)


class CheckpointCache:
    """Manages the local storage and verification of downloaded weights."""

    def __init__(self, cache_dir: str | Path = ".cropxai/checkpoints") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.pt"

    def is_cached(self, name: str) -> bool:
        return self._get_path(name).exists()

    def get_path(self, name: str) -> Path:
        return self._get_path(name)

    def verify_sha256(self, path: Path, expected_hash: str) -> bool:
        """Verify file integrity."""
        logger.info("Verifying SHA-256 for %s...", path.name)
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        actual = sha256.hexdigest()
        if actual != expected_hash:
            logger.error("Hash mismatch! Expected %s, got %s", expected_hash, actual)
            return False
        return True


__all__ = ["CheckpointCache"]
