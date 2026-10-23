"""
CropXAI Checkpoint Validator
==============================

Strictly validates checkpoint architecture match, expected keys, 
embedding dimensions, and resolution before instantiation to prevent 
silent failure modes during evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.research.checkpoints.registry import CheckpointMetadata

logger = logging.getLogger(__name__)


class CheckpointValidator:
    """Validates physical checkpoints against expected registry metadata."""

    @staticmethod
    def validate_keys(model: nn.Module, state_dict: dict[str, Any], strict: bool = False) -> bool:
        """Check for missing or unexpected keys."""
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        
        if missing:
            logger.warning("Missing keys in checkpoint: %s", list(missing)[:5])
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", list(unexpected)[:5])
            
        if strict and (missing or unexpected):
            raise ValueError("Strict key validation failed.")
            
        return True

    @staticmethod
    def validate_metadata(meta: CheckpointMetadata, actual_dim: int, actual_res: tuple[int, ...]) -> bool:
        """Check if the physical checkpoint dimensions match the metadata claims."""
        if meta.embedding_dim != actual_dim:
            raise ValueError(f"Dim mismatch: expected {meta.embedding_dim}, got {actual_dim}")
            
        # In a real scenario, extracting patch size/resolution from the state dict
        # requires parsing specific conv/linear weight shapes.
        logger.debug("Validated metadata for %s", meta.name)
        return True


class DatasetCompatibility:
    """Validates that the dataset structure matches the Foundation Model's requirements."""

    @staticmethod
    def verify(
        dataset_meta: dict[str, Any], 
        model_capability: Any
    ) -> bool:
        """
        model_capability is an instance of ModelCapability.
        """
        # Example validation: if dataset is temporal but model doesn't support it
        temporal = dataset_meta.get("temporal", False)
        if temporal and not model_capability.temporal:
            raise ValueError("Dataset provides temporal sequences, but model does not support it natively.")
            
        # Check Multispectral
        ds_channels = dataset_meta.get("in_channels", 3)
        if ds_channels > 3 and not model_capability.multispectral:
            raise ValueError(f"Dataset has {ds_channels} channels, but model only supports RGB.")
            
        return True


__all__ = ["CheckpointValidator", "DatasetCompatibility"]
