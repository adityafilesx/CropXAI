"""
CropXAI Experiment Dependency Graph
=====================================

Represents campaigns as a Directed Acyclic Graph (DAG) for 
incremental recomputation and cache reuse.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentGraph:
    """DAG representation of the experimental workflow."""

    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, list[str]] = {}

    def add_node(self, exp_id: str, metadata: dict[str, Any]) -> None:
        self.nodes[exp_id] = metadata
        if exp_id not in self.edges:
            self.edges[exp_id] = []

    def add_dependency(self, target_id: str, prerequisite_id: str) -> None:
        """target_id cannot execute until prerequisite_id finishes."""
        if prerequisite_id not in self.edges:
            self.edges[prerequisite_id] = []
        self.edges[prerequisite_id].append(target_id)
        logger.info("Added dependency: %s -> %s", prerequisite_id, target_id)

    def topological_sort(self) -> list[str]:
        """Returns the optimal execution schedule."""
        visited = set()
        stack = []

        def dfs(node: str) -> None:
            visited.add(node)
            for neighbor in self.edges.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
            stack.insert(0, node)

        for n in self.nodes:
            if n not in visited:
                dfs(n)

        return stack


__all__ = ["ExperimentGraph"]
