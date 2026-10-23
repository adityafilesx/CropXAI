"""
CropXAI Base Foundation Adapter
=================================

Base class for wrapping large Earth Observation Foundation Models 
so they seamlessly integrate with the Universal Benchmark pipeline.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
from src.research.models.interfaces.vision_backbone import BaseVisionBackbone, ModelCapability


class BaseFoundationAdapter(BaseVisionBackbone):
    """
    Adapter bridging Foundation Models (e.g. Prithvi, SatMAE) to the VisionBackbone interface.
    Handles downloading weights and adapting specific I/O expectations.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.checkpoint_path = cfg.model.get("checkpoint_path", None)

    @abstractmethod
    def download_weights(self) -> str:
        """Download remote weights if not cached locally."""
        ...

    def load_pretrained(self, checkpoint_path: str | None = None) -> None:
        """Loads weights from the path returned by download_weights."""
        path = checkpoint_path or self.checkpoint_path or self.download_weights()
        self._load_checkpoint(path)

    @abstractmethod
    def _load_checkpoint(self, path: str) -> None:
        """Implementation specific loading logic."""
        ...


__all__ = ["BaseFoundationAdapter"]
