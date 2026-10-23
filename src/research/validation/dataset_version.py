"""
CropXAI Dataset Version Validation
====================================

Ensures that the exact dataset version, preprocessing, augmentation,
and label versions match the frozen protocol. Prevents silent data drift.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DatasetVersionValidator:
    """Validates the dataset provenance against the frozen protocol."""

    def __init__(self, protocol: dict[str, Any]) -> None:
        self.expected = protocol.get("dataset", {})

    def validate(self, current_dataset_meta: dict[str, str]) -> bool:
        """Compares current dataset metadata against expected versions."""
        logger.info("Validating Dataset Provenance...")
        
        keys = ["version", "preprocessing", "augmentations", "label_version"]
        is_valid = True
        
        for key in keys:
            expected_val = self.expected.get(key)
            actual_val = current_dataset_meta.get(key)
            
            if expected_val != actual_val:
                logger.error(
                    "Dataset Drift Detected: %s expected %s, found %s",
                    key, expected_val, actual_val
                )
                is_valid = False
                
        if is_valid:
            logger.info("Dataset Versioning verified. Protocol integrity maintained.")
            
        return is_valid


__all__ = ["DatasetVersionValidator"]
