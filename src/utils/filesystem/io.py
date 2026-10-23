"""
CropXAI Filesystem Utilities
==============================

Safe file I/O helpers: atomic writes, directory creation,
checksum computation, and structured path management.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Directory Helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and all parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dirs(*paths: str | Path) -> list[Path]:
    """Create multiple directories at once."""
    return [ensure_dir(p) for p in paths]


def list_files(directory: str | Path, pattern: str = "*", recursive: bool = True) -> list[Path]:
    """
    List all files matching ``pattern`` in ``directory``.

    Parameters
    ----------
    directory : str | Path
        Root directory to search.
    pattern : str
        Glob pattern (e.g. "*.tif", "*.npy").
    recursive : bool
        If True, search recursively using ``rglob``.
    """
    d = Path(directory)
    if not d.exists():
        return []
    fn = d.rglob if recursive else d.glob
    return sorted(f for f in fn(pattern) if f.is_file())


# ──────────────────────────────────────────────────────────────────────────────
# Atomic Write
# ──────────────────────────────────────────────────────────────────────────────

def atomic_write(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """
    Write text to ``path`` atomically (write to temp file, then rename).
    Prevents partial writes from corrupting files.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory (ensures same filesystem)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        shutil.move(tmp_path, str(target))
    except Exception:
        os.unlink(tmp_path)
        raise


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write a JSON-serialisable object atomically."""
    atomic_write(path, json.dumps(data, indent=indent, default=str))


# ──────────────────────────────────────────────────────────────────────────────
# JSON Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path: str | Path) -> Any:
    """Load and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Save data as a formatted JSON file."""
    atomic_write_json(path, data, indent=indent)


# ──────────────────────────────────────────────────────────────────────────────
# Checksum
# ──────────────────────────────────────────────────────────────────────────────

def compute_checksum(
    path: str | Path,
    algorithm: str = "sha256",
    chunk_size: int = 1 << 20,  # 1 MB
) -> str:
    """
    Compute the hex-encoded checksum of a file.

    Parameters
    ----------
    path : str | Path
        File to hash.
    algorithm : str
        Hash algorithm name (sha256, md5, sha1, etc.).
    chunk_size : int
        Read chunk size in bytes (default 1 MB).
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_directory_checksum(
    directory: str | Path,
    pattern: str = "*",
    algorithm: str = "sha256",
) -> str:
    """
    Compute a stable checksum for an entire directory by hashing
    the concatenation of all file checksums (sorted by path).
    """
    files = list_files(directory, pattern=pattern)
    combined = "".join(compute_checksum(f, algorithm) for f in files)
    return hashlib.new(algorithm, combined.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Path Utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_file_size_mb(path: str | Path) -> float:
    """Return file size in megabytes."""
    return Path(path).stat().st_size / (1024 ** 2)


def get_directory_size_gb(directory: str | Path) -> float:
    """Return total size of all files under ``directory`` in gigabytes."""
    total = sum(f.stat().st_size for f in Path(directory).rglob("*") if f.is_file())
    return total / (1024 ** 3)


def copy_with_metadata(src: str | Path, dst: str | Path) -> Path:
    """Copy a file preserving metadata (timestamps, permissions)."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst_path))
    return dst_path


__all__ = [
    "ensure_dir",
    "ensure_dirs",
    "list_files",
    "atomic_write",
    "atomic_write_json",
    "load_json",
    "save_json",
    "compute_checksum",
    "compute_directory_checksum",
    "get_file_size_mb",
    "get_directory_size_gb",
    "copy_with_metadata",
]
