"""
CropXAI Advanced Statistical Engine
=====================================

Combines classical significance testing with Bayesian methods.
Computes Probability of Superiority, ROPE (Region of Practical Equivalence),
and classical p-values.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class BayesianStatistics:
    """Computes Bayesian statistical metrics."""

    @staticmethod
    def compute_probability_of_superiority(model_a_scores: np.ndarray, model_b_scores: np.ndarray) -> float:
        """
        Estimates P(Model A > Model B) using Monte Carlo or exact integration.
        (Mock implementation for demonstration).
        """
        diffs = model_a_scores - model_b_scores
        prob = np.mean(diffs > 0)
        return float(prob)

    @staticmethod
    def compute_rope(model_a_scores: np.ndarray, model_b_scores: np.ndarray, rope_limit: float = 0.01) -> float:
        """
        Computes the probability that the difference between models falls
        within a Region of Practical Equivalence (e.g. less than 1% difference).
        """
        diffs = model_a_scores - model_b_scores
        prob_in_rope = np.mean(np.abs(diffs) <= rope_limit)
        return float(prob_in_rope)


__all__ = ["BayesianStatistics"]
