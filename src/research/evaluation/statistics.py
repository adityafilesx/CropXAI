"""
CropXAI Statistical Tests
===========================

Computes rigorous statistical significance between model distributions 
using Wilcoxon Signed-Rank, Paired t-tests, and Bootstrap CIs.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

logger = logging.getLogger(__name__)


class StatisticalTests:
    """Calculates significance of benchmark improvements."""

    @staticmethod
    def compute_effect_size(baseline: list[float], experimental: list[float]) -> float:
        """Computes Cohen's d effect size."""
        b_arr = np.array(baseline)
        e_arr = np.array(experimental)
        
        pool_sd = np.sqrt(((len(b_arr) - 1) * np.var(b_arr) + (len(e_arr) - 1) * np.var(e_arr)) / (len(b_arr) + len(e_arr) - 2))
        if pool_sd == 0:
            return 0.0
        return float((np.mean(e_arr) - np.mean(b_arr)) / pool_sd)

    @staticmethod
    def compare_models(baseline: list[float], experimental: list[float]) -> dict[str, Any]:
        """Run statistical comparisons between two sets of seed results."""
        if stats is None:
            logger.warning("scipy not installed. Skipping statistical tests.")
            return {}
            
        b_arr = np.array(baseline)
        e_arr = np.array(experimental)
        
        if len(b_arr) != len(e_arr) or len(b_arr) < 2:
            return {"error": "Requires paired samples of length >= 2"}
            
        # Paired t-test
        t_stat, t_p = stats.ttest_rel(b_arr, e_arr)
        
        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_p = stats.wilcoxon(b_arr, e_arr)
        except ValueError:
            w_stat, w_p = 0.0, 1.0 # E.g., if all differences are zero
            
        effect_size = StatisticalTests.compute_effect_size(baseline, experimental)
        
        return {
            "paired_t_pvalue": float(t_p),
            "wilcoxon_pvalue": float(w_p),
            "cohens_d": effect_size,
            "significant_at_05": float(w_p) < 0.05
        }

    @staticmethod
    def bootstrap_ci(data: list[float], n_bootstraps: int = 1000, ci: float = 95.0) -> tuple[float, float]:
        """Compute empirical bootstrap confidence intervals."""
        arr = np.array(data)
        bootstrapped_means = []
        for _ in range(n_bootstraps):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            bootstrapped_means.append(np.mean(sample))
            
        lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
        upper = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
        return float(lower), float(upper)


__all__ = ["StatisticalTests"]
