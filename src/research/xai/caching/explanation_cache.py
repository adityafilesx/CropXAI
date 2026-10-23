"""
CropXAI Explanation Cache
===========================

Stores computed heatmaps, masks, and attributions indexed by robust hashes
to prevent redundant computation of expensive perturbation methods.
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

import torch
import numpy as np

logger = logging.getLogger(__name__)


class ExplanationCache:
    """Manages disk-based caching of heavy XAI computations."""

    def __init__(self, cache_dir: str | Path = ".cropxai/xai_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_hash(
        self, 
        model_name: str, 
        method_name: str, 
        inputs: torch.Tensor, 
        target_class: Any
    ) -> str:
        """Generates a deterministic hash for an explanation query."""
        # Simple input hash based on mean/std/sum to avoid massive hashing overhead
        input_sig = f"{inputs.shape}_{inputs.sum().item():.4f}_{inputs.mean().item():.4f}"
        target_sig = str(target_class)
        
        payload = f"{model_name}-{method_name}-{input_sig}-{target_sig}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get_path(self, hash_key: str) -> Path:
        return self.cache_dir / f"{hash_key}.pt"

    def get(
        self, 
        model_name: str, 
        method_name: str, 
        inputs: torch.Tensor, 
        target_class: Any
    ) -> torch.Tensor | None:
        """Retrieves cached attribution if it exists."""
        h = self._generate_hash(model_name, method_name, inputs, target_class)
        path = self.get_path(h)
        if path.exists():
            logger.debug("ExplanationCache Hit: %s (%s)", method_name, h)
            return torch.load(path, map_location="cpu")
        return None

    def put(
        self, 
        model_name: str, 
        method_name: str, 
        inputs: torch.Tensor, 
        target_class: Any, 
        attributions: torch.Tensor
    ) -> None:
        """Caches computed attributions."""
        h = self._generate_hash(model_name, method_name, inputs, target_class)
        path = self.get_path(h)
        torch.save(attributions.cpu(), path)
        logger.debug("ExplanationCache Miss -> Saved: %s (%s)", method_name, h)


__all__ = ["ExplanationCache"]
