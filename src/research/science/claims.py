"""
CropXAI Scientific Claim Validator
====================================

Maps English textual claims in the manuscript directly to the 
EvidenceRegistry, ensuring no unsupported statements are published.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ScientificClaimValidator:
    """Validates claims against the EvidenceRegistry."""

    def __init__(self, evidence_registry: Any) -> None:
        self.registry = evidence_registry

    def validate_claim(self, claim_text: str, evidence_id: str, threshold_p: float = 0.05) -> bool:
        """Checks if a claim is statistically supported."""
        evidence = self.registry.get_evidence(evidence_id)
        if not evidence:
            logger.error("Claim Validation Failed: No evidence found for ID %s", evidence_id)
            return False
            
        p_val = evidence.get("p_value", 1.0)
        if p_val > threshold_p:
            logger.warning(
                "Claim Validation Failed: '%s' is not statistically significant (p=%.3f > %.3f)",
                claim_text, p_val, threshold_p
            )
            return False
            
        logger.info("Claim Validated: '%s' (p=%.3f)", claim_text, p_val)
        return True


__all__ = ["ScientificClaimValidator"]
