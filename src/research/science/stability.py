"""
CropXAI Result Stability Analyzer
===================================

Analyzes variance decomposition to attribute performance variance
to Seed Sensitivity vs Hyperparameter Sensitivity vs Dataset Shift.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """Analyzes the robustness of experimental results."""

    @staticmethod
    def compute_variance_decomposition(results_df: pd.DataFrame) -> dict[str, float]:
        """
        Decomposes the variance of a metric (e.g. Accuracy) into 
        factors: Seed, Dataset Split, and Hyperparameters.
        (Mocked ANOVA logic).
        """
        logger.info("Computing Variance Decomposition...")
        
        # If we had actual categorical columns, we would run a linear mixed model
        # or ANOVA to extract sum of squares.
        
        total_var = 1.0
        seed_var = 0.15
        dataset_var = 0.80
        hyper_var = 0.05
        
        return {
            "Total_Variance": total_var,
            "Seed_Contribution": seed_var,
            "Dataset_Contribution": dataset_var,
            "Hyperparameter_Contribution": hyper_var
        }

    @staticmethod
    def compute_seed_sensitivity(metrics_across_seeds: np.ndarray) -> float:
        """
        Returns the coefficient of variation (CV) across random seeds.
        CV = std / mean
        """
        if len(metrics_across_seeds) < 2:
            return 0.0
        
        mean = np.mean(metrics_across_seeds)
        std = np.std(metrics_across_seeds, ddof=1)
        
        if mean == 0:
            return 0.0
            
        cv = std / mean
        return float(cv)


__all__ = ["StabilityAnalyzer"]
