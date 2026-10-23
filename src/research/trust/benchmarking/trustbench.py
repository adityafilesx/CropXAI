"""
CropXAI TrustBench
====================

Scientific benchmarking suite to compare multiple Trust Estimators 
(Weighted vs Learned vs Bayesian) using the MetaTrustEvaluator.
"""

from __future__ import annotations

import logging
from typing import Any
import pandas as pd
import torch

from src.research.trust.evaluation.meta_evaluator import MetaTrustEvaluator

logger = logging.getLogger(__name__)


class TrustBench:
    """Benchmarks multiple TrustEstimators against each other."""

    def __init__(self, estimators: list[Any], cfg: Any = None) -> None:
        self.estimators = estimators
        self.cfg = cfg
        self.results: list[dict[str, Any]] = []

    def run(self, test_features: dict[str, torch.Tensor], correctness_labels: torch.Tensor) -> pd.DataFrame:
        """
        Evaluates each estimator on the test set.
        test_features must contain: confidence, uncertainty, ood_score, 
        faithfulness, calibration_error, representation_score.
        """
        for estimator in self.estimators:
            name = estimator.__class__.__name__
            logger.info("Evaluating Trust Estimator: %s", name)
            
            # Predict Trust
            trust_scores = estimator.estimate_trust(**test_features)
            
            # Meta-Evaluate
            metrics = MetaTrustEvaluator.evaluate_all(trust_scores, correctness_labels)
            
            metrics["Estimator"] = name
            self.results.append(metrics)
            
        return pd.DataFrame(self.results)


__all__ = ["TrustBench"]
