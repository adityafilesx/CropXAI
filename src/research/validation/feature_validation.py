"""
CropXAI Feature Validation Pipeline
=====================================

Validates architectural compliance, ensuring embedding shapes and 
feature pyramids behave exactly as advertised by ModelCapabilities.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from src.research.models.interfaces.vision_backbone import BaseVisionBackbone

logger = logging.getLogger(__name__)


class ArchitectureValidator:
    """Strictly validates backbone outputs against its capabilities."""

    @staticmethod
    def validate(model: BaseVisionBackbone, device: torch.device = torch.device("cpu")) -> bool:
        """Run all architectural validations."""
        model.eval()
        cap = model.capability
        
        # Dummy Input
        # (B, T, C, H, W)
        if cap.temporal:
            x = torch.randn(2, 3, 3, 224, 224, device=device)
        else:
            x = torch.randn(2, 1, 3, 224, 224, device=device)
            # Simulating InputAdapter squeeze
            x = x.squeeze(1)

        # 1. Forward Pass
        try:
            with torch.no_grad():
                out = model(x)
        except Exception as e:
            logger.error("Validation failed during standard forward pass: %s", e)
            raise

        # 2. Extract Multiscale
        try:
            with torch.no_grad():
                features = model.extract_multiscale(x)
            
            # Check dimensions match feature_dimensions property
            actual_dims = [f.shape[1] for f in features]
            if actual_dims != model.feature_dimensions:
                raise ValueError(
                    f"Multiscale channels {actual_dims} do not match "
                    f"declared feature_dimensions {model.feature_dimensions}"
                )
                
            # Check spatial resolution decreases correctly (for CNNs/Hierarchical ViTs)
            if len(features) > 1 and cap.variable_resolution:
                h1, w1 = features[0].shape[-2:]
                h2, w2 = features[-1].shape[-2:]
                if h1 <= h2 or w1 <= w2:
                    logger.warning("Feature pyramid spatial dims do not decrease as expected.")
                    
        except Exception as e:
            logger.error("Validation failed during extract_multiscale: %s", e)
            raise
            
        # 3. Extract Embeddings
        if cap.embeddings:
            try:
                with torch.no_grad():
                    embeds = model.extract_embeddings(x)
                if embeds.shape[-1] != cap.embedding_dim:
                    raise ValueError(
                        f"Extracted embedding dim {embeds.shape[-1]} does not match "
                        f"capability descriptor {cap.embedding_dim}"
                    )
            except Exception as e:
                logger.error("Validation failed during extract_embeddings: %s", e)
                raise
                
        logger.info("Architecture Validation Passed for %s", model.__class__.__name__)
        return True


__all__ = ["ArchitectureValidator"]
