"""
CropXAI Experiment Lineage
============================

Tracks the complete scientific provenance of an experiment, allowing 
it to be traced back to its parent or ablation origin.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentLineage:
    """Tracks the origin and derivation of an experiment."""

    def __init__(self) -> None:
        self.lineage_map: dict[str, dict[str, str]] = {}

    def register_lineage(self, child_id: str, parent_id: str, derivation_type: str = "ablation") -> None:
        """
        Records that child_id was derived from parent_id.
        Types: ablation, fine-tuning, cross_dataset, continuation
        """
        self.lineage_map[child_id] = {
            "parent": parent_id,
            "type": derivation_type
        }
        logger.info("Registered Lineage: %s derives from %s (%s)", child_id, parent_id, derivation_type)

    def get_ancestry(self, exp_id: str) -> list[str]:
        """Traces back the full ancestry of an experiment."""
        ancestry = []
        curr = exp_id
        while curr in self.lineage_map:
            parent = self.lineage_map[curr]["parent"]
            ancestry.append(parent)
            curr = parent
        return ancestry


__all__ = ["ExperimentLineage"]
