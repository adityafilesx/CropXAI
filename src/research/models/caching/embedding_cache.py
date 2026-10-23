"""
CropXAI Embedding Cache
=========================

Disk/Memory cache for Foundation Model embeddings to accelerate downstream tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Caches expensive Foundation Model embeddings.
    """

    def __init__(self, cache_dir: str | Path, in_memory: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.in_memory = in_memory
        self.memory_cache: dict[str, torch.Tensor] = {}

    def _get_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pt"

    def put(self, key: str, embeddings: torch.Tensor) -> None:
        """Store embeddings in cache."""
        if self.in_memory:
            self.memory_cache[key] = embeddings.cpu()
        else:
            torch.save(embeddings.cpu(), self._get_path(key))

    def get(self, key: str) -> torch.Tensor | None:
        """Retrieve embeddings from cache."""
        if self.in_memory:
            return self.memory_cache.get(key)
        
        path = self._get_path(key)
        if path.exists():
            return torch.load(path, weights_only=True)
        return None

    def clear(self) -> None:
        """Clear the cache."""
        self.memory_cache.clear()
        for p in self.cache_dir.glob("*.pt"):
            p.unlink()


__all__ = ["EmbeddingCache"]
