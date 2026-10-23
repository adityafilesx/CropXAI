"""
CropXAI Meta-Trust Evaluator
==============================

Validates the Trust Score itself. Computes Trust Accuracy, Trust AUC,
Trust Calibration, and Expected Trust Error to scientifically prove 
if a trust estimator is reliable.
"""

from __future__ import annotations

import logging
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


class MetaTrustEvaluator:
    """Evaluates how good the Trust Score is at predicting correctness."""

    @staticmethod
    def compute_trust_auc(trust_scores: torch.Tensor, correctness_labels: torch.Tensor) -> float:
        """
        Computes ROC-AUC of the Trust Score predicting prediction correctness.
        """
        trust_np = trust_scores.detach().cpu().numpy()
        labels_np = correctness_labels.detach().cpu().numpy()
        
        # If all predictions are correct or incorrect, AUC is undefined
        if len(np.unique(labels_np)) < 2:
            return 1.0
            
        return float(roc_auc_score(labels_np, trust_np))

    @staticmethod
    def compute_trust_pr(trust_scores: torch.Tensor, correctness_labels: torch.Tensor) -> float:
        """
        Computes Precision-Recall AUC of the Trust Score.
        Particularly important when failures are rare (high accuracy models).
        """
        trust_np = trust_scores.detach().cpu().numpy()
        labels_np = correctness_labels.detach().cpu().numpy()
        
        if len(np.unique(labels_np)) < 2:
            return 1.0
            
        return float(average_precision_score(labels_np, trust_np))

    @staticmethod
    def compute_expected_trust_error(trust_scores: torch.Tensor, correctness_labels: torch.Tensor, n_bins: int = 10) -> float:
        """
        Measures Trust Calibration (Expected Trust Error).
        Similar to ECE, but for the Trust Score instead of confidence.
        """
        trust_np = trust_scores.detach().cpu().numpy()
        labels_np = correctness_labels.detach().cpu().numpy()
        
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(trust_np, bins, right=True)
        
        ete = 0.0
        total_samples = len(trust_np)
        
        for b in range(1, n_bins + 1):
            mask = (bin_indices == b)
            if np.any(mask):
                bin_trust = np.mean(trust_np[mask])
                bin_acc = np.mean(labels_np[mask])
                weight = np.sum(mask) / total_samples
                ete += weight * np.abs(bin_trust - bin_acc)
                
        return float(ete)

    @staticmethod
    def evaluate_all(trust_scores: torch.Tensor, correctness_labels: torch.Tensor) -> dict[str, float]:
        """Runs the full meta-evaluation suite."""
        # For Pareto Trust (B, 3), we use the L2 norm or mean for general evaluation,
        # or evaluate each dimension separately. Assuming scalar trust here.
        if trust_scores.dim() > 1 and trust_scores.shape[1] > 1:
            trust_scores = trust_scores.mean(dim=1)
            
        return {
            "Trust_AUC": MetaTrustEvaluator.compute_trust_auc(trust_scores, correctness_labels),
            "Trust_PR": MetaTrustEvaluator.compute_trust_pr(trust_scores, correctness_labels),
            "Expected_Trust_Error": MetaTrustEvaluator.compute_expected_trust_error(trust_scores, correctness_labels)
        }


__all__ = ["MetaTrustEvaluator"]
