"""
CropXAI Checkpoint Manager
============================

Orchestrates downloading, caching, and loading pretrained foundation models.
"""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

import torch
import torch.nn as nn

from src.research.checkpoints.registry import PretrainedRegistry
from src.research.checkpoints.cache import CheckpointCache
from src.research.checkpoints.validation import CheckpointValidator

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    High-level manager for pretrained weights.
    Supports HuggingFace, TIMM, and local manual fallbacks.
    """

    def __init__(self, cache_dir: str | Path = ".cropxai/checkpoints") -> None:
        self.cache = CheckpointCache(cache_dir)
        
    def resolve_and_load(self, model: nn.Module, name: str, manual_path: str | None = None) -> None:
        """
        Resolves the checkpoint location (downloading if needed), 
        validates the weights, and loads them into the model.
        """
        if manual_path is not None:
            logger.info("Using manual fallback path for %s: %s", name, manual_path)
            self._load_local(model, manual_path)
            return
            
        try:
            meta = PretrainedRegistry.get(name)
        except KeyError:
            logger.warning("No official metadata found for %s. Attempting load without validation.", name)
            # Cannot do much here without a path
            return

        # If it's HuggingFace or Timm, the underlying library (transformers/timm) 
        # usually handles downloading and caching natively via their own APIs.
        # We wrap that mechanism in the respective Adapters, but we log and validate here.
        logger.info(
            "Resolving checkpoint for %s (Source: %s, License: %s)", 
            meta.name, meta.source, meta.license
        )
        
        # If it's a strictly local model (e.g. SatMAE custom weights)
        if meta.source == "local":
            if not self.cache.is_cached(meta.name):
                logger.error("Local checkpoint %s not found in cache. Please provide manual_path.", meta.name)
                raise FileNotFoundError(f"Missing local checkpoint for {meta.name}")
            self._load_local(model, str(self.cache.get_path(meta.name)))

    def _load_local(self, model: nn.Module, path: str) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        state_dict = torch.load(path, map_location="cpu")
        # Handle cases where state_dict is nested under 'model' or 'state_dict'
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        CheckpointValidator.validate_keys(model, state_dict, strict=False)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded local checkpoint from %s", path)


__all__ = ["CheckpointManager"]
