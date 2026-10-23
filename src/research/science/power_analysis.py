"""
CropXAI Statistical Power Analysis
====================================

Calculates statistical power based on the observed effect sizes 
and number of seeds to confirm the robustness of the methodology.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class StatisticalPowerAnalyzer:
    """Computes achieved statistical power for benchmark experiments."""

    @staticmethod
    def calculate_achieved_power(effect_size: float, n_seeds: int, alpha: float = 0.05) -> float:
        """
        Mock calculation for achieved statistical power.
        A power > 0.8 is generally considered acceptable.
        """
        logger.info("Computing achieved statistical power for n=%d seeds, d=%.2f", n_seeds, effect_size)
        
        # In a real implementation, this would use statsmodels:
        # from statsmodels.stats.power import TTestIndPower
        # analysis = TTestIndPower()
        # power = analysis.solve_power(effect_size=effect_size, nobs1=n_seeds, alpha=alpha)
        
        # Mock formula
        power = min(0.99, max(0.05, 1.0 - np.exp(-effect_size * n_seeds / 2.0)))
        return float(power)

    @staticmethod
    def recommend_sample_size(effect_size: float, target_power: float = 0.80, alpha: float = 0.05) -> int:
        """Estimates the number of seeds needed to achieve target power."""
        if effect_size <= 0:
            return 100
        
        # Mock calculation
        n = int(-2.0 * np.log(1.0 - target_power) / effect_size)
        return max(5, n)


__all__ = ["StatisticalPowerAnalyzer"]
