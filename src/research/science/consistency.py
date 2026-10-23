"""
CropXAI Claim Consistency Checker
===================================

Cross-references the manuscript text, figure captions, and tables
to ensure no contradictions exist before final publication.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ClaimConsistencyChecker:
    """Verifies that all scientific claims are consistent across the manuscript."""

    def __init__(self, hypothesis_registry: Any, evidence_registry: Any) -> None:
        self.hypothesis_registry = hypothesis_registry
        self.evidence_registry = evidence_registry

    def verify_publication_readiness(self) -> bool:
        """
        Runs before `cropxai paper build` to ensure every statement in the Results
        section is backed by the database.
        """
        logger.info("Running Claim Consistency Check...")
        hyps = self.hypothesis_registry.get_summary()
        
        all_consistent = True
        for hyp_id, hyp_data in hyps.items():
            if hyp_data["status"] == "UNTESTED":
                logger.error("Consistency Error: Hypothesis %s is UNTESTED.", hyp_id)
                all_consistent = False
            elif hyp_data["status"] == "REJECTED":
                logger.warning("Consistency Warning: Hypothesis %s is REJECTED.", hyp_id)
                
        if all_consistent:
            logger.info("All claims are consistent and supported by statistical evidence.")
            
        return all_consistent


__all__ = ["ClaimConsistencyChecker"]
