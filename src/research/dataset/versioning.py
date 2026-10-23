"""
CropXAI Dataset Versioning System
====================================

Implements immutable dataset version snapshots with SHA-256 checksums.

Every preprocessing run writes to a versioned directory:
  ``data/processed/v1.0/``

A ``manifest.json`` is written alongside containing:
  - Version tag
  - Creation timestamp
  - Config hash
  - Per-file checksums
  - Lineage (which raw version it was derived from)

The version registry at ``data/version/manifest.json`` tracks all versions.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.filesystem.io import (
    ensure_dir,
    save_json,
    load_json,
    list_files,
    compute_checksum,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Version Manifest
# ──────────────────────────────────────────────────────────────────────────────

class VersionManifest:
    """
    Manages the manifest file for a single dataset version.

    Parameters
    ----------
    version_dir : Path
        Directory for this dataset version (e.g. ``data/processed/v1.0/``).
    """

    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, version_dir: str | Path) -> None:
        self.version_dir = Path(version_dir)
        self.manifest_path = self.version_dir / self.MANIFEST_FILENAME

    def create(
        self,
        version_tag: str,
        config_hash: str,
        description: str = "",
        parent_version: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new manifest for this version.

        Parameters
        ----------
        version_tag : str
            Version identifier (e.g. "v1.0").
        config_hash : str
            SHA-256 hash of the preprocessing config used.
        description : str
            Human-readable description of what changed in this version.
        parent_version : str | None
            Version tag this was derived from (for lineage tracking).
        """
        manifest: dict[str, Any] = {
            "version": version_tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": config_hash,
            "description": description,
            "parent_version": parent_version,
            "checksums": {},
            "stats": {},
        }
        save_json(self.manifest_path, manifest)
        logger.info("Created manifest for version '%s' at %s", version_tag, self.manifest_path)
        return manifest

    def add_file(self, path: str | Path) -> str:
        """
        Add a file to the manifest (compute and store its checksum).

        Parameters
        ----------
        path : str | Path
            Absolute path to the file.

        Returns
        -------
        str
            SHA-256 checksum of the file.
        """
        manifest = self.load()
        p = Path(path)
        rel_path = str(p.relative_to(self.version_dir))
        checksum = compute_checksum(p)
        manifest.setdefault("checksums", {})[rel_path] = checksum
        save_json(self.manifest_path, manifest)
        return checksum

    def finalize(self, stats: dict[str, Any] | None = None) -> str:
        """
        Finalize the manifest: compute the overall version checksum and store stats.

        Returns
        -------
        str
            SHA-256 of all file checksums combined (the "version fingerprint").
        """
        manifest = self.load()
        checksums = manifest.get("checksums", {})
        combined = "".join(sorted(checksums.values()))
        version_hash = hashlib.sha256(combined.encode()).hexdigest()

        manifest["version_hash"] = version_hash
        manifest["finalized_at"] = datetime.now(timezone.utc).isoformat()
        manifest["num_files"] = len(checksums)
        if stats:
            manifest["stats"] = stats

        save_json(self.manifest_path, manifest)
        logger.info("Version finalized. Hash: %s...", version_hash[:12])
        return version_hash

    def load(self) -> dict[str, Any]:
        """Load the manifest from disk."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        return load_json(self.manifest_path)

    def verify(self) -> tuple[bool, list[str]]:
        """
        Verify all files in the manifest against their stored checksums.

        Returns
        -------
        tuple[bool, list[str]]
            (all_valid, list_of_failed_relative_paths)
        """
        manifest = self.load()
        failed = []
        for rel_path, expected_hash in manifest.get("checksums", {}).items():
            abs_path = self.version_dir / rel_path
            if not abs_path.exists():
                failed.append(f"MISSING: {rel_path}")
                continue
            actual = compute_checksum(abs_path)
            if actual != expected_hash:
                failed.append(f"CORRUPT: {rel_path}")
        return len(failed) == 0, failed


# ──────────────────────────────────────────────────────────────────────────────
# Version Registry
# ──────────────────────────────────────────────────────────────────────────────

class DatasetVersionRegistry:
    """
    Maintains the master registry of all dataset versions.

    Stored at ``data/version/manifest.json``.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)

    def register(self, version_tag: str, version_dir: str | Path, metadata: dict[str, Any]) -> None:
        """Register a new version in the global registry."""
        registry = self._load()
        registry["versions"][version_tag] = {
            "path": str(version_dir),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        save_json(self.registry_path, registry)
        logger.info("Registered dataset version '%s' in global registry.", version_tag)

    def get(self, version_tag: str) -> dict[str, Any] | None:
        """Get registry entry for a version tag."""
        return self._load()["versions"].get(version_tag)

    def list_versions(self) -> list[str]:
        """Return sorted list of all registered version tags."""
        return sorted(self._load()["versions"].keys())

    def latest(self) -> str | None:
        """Return the most recently registered version tag."""
        versions = self.list_versions()
        return versions[-1] if versions else None

    def _load(self) -> dict[str, Any]:
        ensure_dir(self.registry_path.parent)
        if not self.registry_path.exists():
            default: dict[str, Any] = {"versions": {}}
            save_json(self.registry_path, default)
            return default
        return load_json(self.registry_path)


# ──────────────────────────────────────────────────────────────────────────────
# Versioning Stage
# ──────────────────────────────────────────────────────────────────────────────

class VersioningStage:
    """
    Creates a new immutable dataset version snapshot.

    Called at the start of every preprocessing run to stamp the output.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def initialize_version(self, config_hash: str) -> tuple[Path, VersionManifest]:
        """
        Set up the versioned output directory and its manifest.

        Returns
        -------
        tuple[Path, VersionManifest]
            (version_dir, manifest)
        """
        version_cfg = self.cfg.dataset.versioning
        version_tag = version_cfg.version_tag
        version_dir = ensure_dir(Path(version_cfg.processed_dir))

        manifest = VersionManifest(version_dir)
        manifest.create(
            version_tag=version_tag,
            config_hash=config_hash,
            description=f"Auto-generated by CropXAI preprocessing pipeline.",
        )

        # Register in global registry
        registry = DatasetVersionRegistry(
            Path(self.cfg.paths.data_root) / "version" / "manifest.json"
        )
        registry.register(version_tag, version_dir, {"config_hash": config_hash})

        logger.info("Initialized dataset version '%s' at %s", version_tag, version_dir)
        return version_dir, manifest


__all__ = [
    "VersionManifest",
    "DatasetVersionRegistry",
    "VersioningStage",
]
