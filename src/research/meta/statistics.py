"""
CropXAI Statistical Rigor Engine
================================
Implements non-parametric tests, effect sizes, and Bayesian metrics
for the TrustCrop framework. 

All metrics must report confidence intervals and effect sizes. No naked p-values.
"""

import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

def bootstrap_ci(data: list[float], alpha: float = 0.05, n_resamples: int = 2000) -> tuple[float, float]:
    """Computes a (1-alpha)% bootstrap confidence interval for the mean."""
    if len(data) < 2:
        return np.mean(data), np.mean(data)
    
    data = np.array(data)
    n = len(data)
    boot_means = np.mean(np.random.choice(data, size=(n_resamples, n), replace=True), axis=1)
    
    lo = np.percentile(boot_means, 100 * (alpha / 2))
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def cliffs_delta(x: list[float], y: list[float]) -> float:
    """
    Computes Cliff's Delta effect size for two distributions.
    Interpretation: <0.14 (negligible), <0.33 (small), <0.47 (medium), >0.47 (large)
    """
    if not x or not y:
        return 0.0
        
    x, y = np.array(x), np.array(y)
    m, n = len(x), len(y)
    
    # Compare every element of x to every element of y
    # A > B counts as 1, A < B counts as -1
    diffs = np.sign(x[:, None] - y)
    delta = np.sum(diffs) / (m * n)
    return float(delta)

def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, float]:
    """
    Computes Wilcoxon signed-rank test for paired samples.
    Returns (statistic, p_value).
    """
    if len(x) < 2 or len(x) != len(y):
        return 0.0, 1.0
        
    try:
        stat, p = stats.wilcoxon(x, y)
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0
        
def evaluate_significance(metrics_a: list[float], metrics_b: list[float]) -> dict[str, float]:
    """
    Comprehensive statistical comparison between model A and model B.
    """
    delta = cliffs_delta(metrics_a, metrics_b)
    stat, p_val = wilcoxon_signed_rank(metrics_a, metrics_b)
    
    mean_a = np.mean(metrics_a) if metrics_a else 0.0
    mean_b = np.mean(metrics_b) if metrics_b else 0.0
    
    ci_a_lo, ci_a_hi = bootstrap_ci(metrics_a)
    ci_b_lo, ci_b_hi = bootstrap_ci(metrics_b)
    
    return {
        "mean_A": float(mean_a),
        "ci_A": (ci_a_lo, ci_a_hi),
        "mean_B": float(mean_b),
        "ci_B": (ci_b_lo, ci_b_hi),
        "cliffs_delta": delta,
        "wilcoxon_p": p_val
    }
