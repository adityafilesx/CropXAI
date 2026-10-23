"""
CropXAI Dataset Downloader
============================

Downloads the IBM–NASA Crop Classification Dataset from available sources
(HuggingFace Hub, TorchGeo registry, or a direct URL).

Implements the :class:`PipelineStage` interface so it slots into the
:class:`DatasetPipeline`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from src.core.pipeline import PipelineStage, StageResult
from src.utils.filesystem.io import ensure_dir, compute_checksum

logger = logging.getLogger(__name__)


class DownloadContext:
    """Shared context object passed through the dataset pipeline."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.raw_dir = Path(cfg.paths.raw)
        self.dataset_dir = Path(cfg.paths.raw) / cfg.dataset.name
        self.files: list[Path] = []
        self.metadata: dict[str, Any] = {}


class DownloadStage(PipelineStage):
    """
    Stage 1: Download the dataset from the configured source.

    Supported sources:
      - ``huggingface``: Download from HuggingFace Hub
      - ``torchgeo``: Use TorchGeo dataset registry
      - ``local``: No download; just verify the path exists
      - ``url``: Download from a direct HTTP/HTTPS URL

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root CropXAI config.
    """

    def __init__(self, cfg: Any) -> None:
        download_cfg = cfg.dataset.get("download", {})
        super().__init__(
            name="Download",
            enabled=download_cfg.get("auto_download", False),
        )
        self.cfg = cfg
        self.download_cfg = download_cfg

    def process(self, data: DownloadContext | None) -> DownloadContext:
        if data is None:
            data = DownloadContext(self.cfg)

        source = self.download_cfg.get("source", "local")
        logger.info("Dataset source: %s", source)

        if source == "local":
            self._verify_local(data)
        elif source == "huggingface":
            self._download_huggingface(data)
        elif source == "torchgeo":
            self._download_torchgeo(data)
        elif source == "url":
            self._download_url(data)
        else:
            raise ValueError(f"Unknown dataset source: {source!r}")

        return data

    def _verify_local(self, data: DownloadContext) -> None:
        """Verify that locally specified data path exists."""
        dataset_dir = data.dataset_dir
        if not dataset_dir.exists():
            logger.warning(
                "Local dataset directory not found: %s\n"
                "Run `cropxai download` after placing data there, or set "
                "`dataset.download.source=huggingface`.",
                dataset_dir,
            )
        else:
            from src.utils.filesystem.io import list_files
            data.files = list_files(dataset_dir, pattern="*.tif")
            logger.info(
                "Local dataset found: %d .tif files in %s",
                len(data.files),
                dataset_dir,
            )

    def _download_huggingface(self, data: DownloadContext) -> None:
        """Download from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HuggingFace downloads. "
                "Install with: pip install huggingface_hub"
            )

        repo_id = self.download_cfg.get("repo_id")
        if not repo_id:
            raise ValueError("dataset.download.repo_id must be set for HuggingFace downloads.")

        cache_dir = ensure_dir(self.download_cfg.get("cache_dir", data.raw_dir / "hf_cache"))
        logger.info("Downloading from HuggingFace Hub: %s → %s", repo_id, cache_dir)

        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        data.dataset_dir = Path(local_dir)
        logger.info("✓ HuggingFace download complete: %s", local_dir)

    def _download_torchgeo(self, data: DownloadContext) -> None:
        """Use TorchGeo's built-in download mechanism."""
        logger.info("Using TorchGeo download (see dataset config for details).")
        ensure_dir(data.dataset_dir)
        # TorchGeo datasets handle their own download via `download=True` in __init__
        # This is a hook for custom TorchGeo-based datasets.
        data.metadata["download_method"] = "torchgeo"

    def _download_url(self, data: DownloadContext) -> None:
        """Download from a direct URL using urllib."""
        import urllib.request

        url = self.download_cfg.get("url")
        if not url:
            raise ValueError("dataset.download.url must be set for URL downloads.")

        filename = Path(url).name
        dest = ensure_dir(data.dataset_dir) / filename

        logger.info("Downloading from URL: %s → %s", url, dest)

        def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            pct = min(downloaded / total_size * 100, 100.0) if total_size > 0 else 0.0
            logger.debug("  %.1f%% (%d / %d bytes)", pct, downloaded, total_size)

        urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
        logger.info("✓ URL download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


__all__ = ["DownloadStage", "DownloadContext"]
