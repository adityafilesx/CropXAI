"""
CropXAI Dataset Cache
======================

Disk-based LRU cache for preprocessed dataset samples.
Avoids redundant re-preprocessing of large GeoTIFF files.

Backend options:
  - ``joblib``: Fast, numpy-optimized (default)
  - ``shelve``: Pure Python, no dependencies
  - ``memory``: In-memory (no persistence)
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class DatasetCache:
    """
    Disk-based cache for preprocessed dataset samples.

    Parameters
    ----------
    cache_dir : str | Path
        Directory to store cached files.
    backend : str
        Cache backend: "joblib" | "shelve" | "memory".
    max_size_gb : float
        Maximum cache size in gigabytes. Oldest entries evicted when exceeded.
    enabled : bool
        If False, all operations are no-ops.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        backend: str = "joblib",
        max_size_gb: float = 20.0,
        enabled: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.max_size_gb = max_size_gb
        self.enabled = enabled
        self._memory_cache: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """Retrieve a cached item. Returns None if not in cache."""
        if not self.enabled:
            return None

        try:
            result = self._get(key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result
        except Exception as exc:
            logger.debug("Cache get failed for key '%s': %s", key, exc)
            self._misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Store an item in the cache."""
        if not self.enabled:
            return

        try:
            self._check_and_evict()
            self._put(key, value)
        except Exception as exc:
            logger.debug("Cache put failed for key '%s': %s", key, exc)

    def clear(self) -> None:
        """Clear the entire cache."""
        if self.backend == "joblib":
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        elif self.backend == "memory":
            self._memory_cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    @property
    def size_gb(self) -> float:
        if self.backend == "memory":
            return 0.0
        return sum(f.stat().st_size for f in self.cache_dir.rglob("*") if f.is_file()) / 1e9

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "size_gb": round(self.size_gb, 3),
            "enabled": self.enabled,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Backend Implementations
    # ──────────────────────────────────────────────────────────────────────────

    def _get(self, key: str) -> Any | None:
        if self.backend == "joblib":
            return self._joblib_get(key)
        elif self.backend == "memory":
            return self._memory_cache.get(key)
        else:
            return None

    def _put(self, key: str, value: Any) -> None:
        if self.backend == "joblib":
            self._joblib_put(key, value)
        elif self.backend == "memory":
            self._memory_cache[key] = value

    def _joblib_get(self, key: str) -> Any | None:
        import joblib
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            return joblib.load(cache_path)
        return None

    def _joblib_put(self, key: str, value: Any) -> None:
        import joblib
        cache_path = self.cache_dir / f"{key}.pkl"
        joblib.dump(value, cache_path, compress=3)

    def _check_and_evict(self) -> None:
        """Evict oldest cache entries if max size exceeded."""
        if self.backend == "memory":
            return
        if self.size_gb > self.max_size_gb:
            files = sorted(
                (f for f in self.cache_dir.rglob("*.pkl") if f.is_file()),
                key=lambda f: f.stat().st_mtime,
            )
            removed = 0
            for f in files:
                if self.size_gb <= self.max_size_gb * 0.9:
                    break
                f.unlink()
                removed += 1
            if removed > 0:
                logger.info("Cache evicted %d entries (size limit: %.1f GB).", removed, self.max_size_gb)


def cached(cache: DatasetCache, key_fn: Callable[..., str]) -> Callable[[F], F]:
    """
    Decorator that caches the return value of a function in ``cache``.

    Parameters
    ----------
    cache : DatasetCache
        The cache instance to use.
    key_fn : Callable
        Function that generates a cache key from the decorated function's args.
    """
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = key_fn(*args, **kwargs)
            result = cache.get(key)
            if result is not None:
                return result
            result = fn(*args, **kwargs)
            cache.put(key, result)
            return result
        return wrapper  # type: ignore[return-value]
    return decorator


__all__ = ["DatasetCache", "cached"]
