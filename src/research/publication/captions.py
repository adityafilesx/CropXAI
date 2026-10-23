"""
CropXAI Caption Generator
===========================

Autogenerates detailed scientific captions based on the statistical evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Generates markdown captions for figures and tables."""

    @staticmethod
    def generate_caption(claim_id: str, evidence: dict[str, Any]) -> str:
        """
        Translates raw statistical evidence into a human-readable caption.
        """
        p_val = evidence.get('p_value', 1.0)
        marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        caption = (
            f"**Figure/Table corresponding to {claim_id}.**\n"
            f"Statistical significance marked as {marker} (p={p_val:.4f}). "
            f"Effect size (Cohen's d): {evidence.get('effect_size', 0.0):.2f}. "
            "Data generated autonomously by ASCOPE."
        )
        return caption


__all__ = ["CaptionGenerator"]
