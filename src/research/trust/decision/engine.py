"""
CropXAI Decision Engine
=========================

Produces actionable decisions (Accept, Reject, Refer to Human) 
based on Trust Scores and Risk-Coverage optimization policies.
"""

from __future__ import annotations

import logging
from typing import Any
import torch

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Translates continuous Trust Scores into discrete operational decisions.
    Supports dynamic thresholding based on Risk-Coverage targets.
    """

    def __init__(self, policy: str = "risk_coverage", cfg: Any = None) -> None:
        self.policy = policy
        self.cfg = cfg
        # e.g., target maximum risk of 5% error
        self.target_risk = getattr(self.cfg, "target_risk", 0.05) if self.cfg else 0.05
        # Pre-computed trust threshold that satisfies the target risk on the validation set
        self.empirical_threshold = getattr(self.cfg, "empirical_threshold", 0.7) if self.cfg else 0.7

    def calibrate_threshold(self, val_trust_scores: torch.Tensor, val_correctness: torch.Tensor) -> float:
        """
        Finds the optimal trust threshold that maximizes coverage while keeping 
        the error rate (1 - correctness) below the target_risk.
        """
        sorted_trust, indices = torch.sort(val_trust_scores, descending=True)
        sorted_correctness = val_correctness[indices]
        
        cumulative_errors = torch.cumsum(1.0 - sorted_correctness, dim=0)
        coverage_counts = torch.arange(1, len(sorted_trust) + 1, device=val_trust_scores.device)
        
        # Risk = cumulative errors / coverage
        risks = cumulative_errors / coverage_counts
        
        # Find maximum coverage where risk <= target_risk
        valid_indices = torch.where(risks <= self.target_risk)[0]
        
        if len(valid_indices) == 0:
            # Cannot satisfy risk, require extremely high trust
            self.empirical_threshold = float(sorted_trust[0].item())
        else:
            best_idx = valid_indices[-1]
            self.empirical_threshold = float(sorted_trust[best_idx].item())
            
        logger.info("Calibrated Trust Threshold: %.3f for Target Risk: %.3f", self.empirical_threshold, self.target_risk)
        return self.empirical_threshold

    def decide(self, predictions: torch.Tensor, trust_scores: torch.Tensor) -> dict[str, Any]:
        """
        Outputs discrete decisions.
        """
        b = predictions.shape[0]
        
        # For Pareto Trust, we might have specific rules for each dimension
        if trust_scores.dim() > 1:
            # Simplify to mean for basic decisioning
            trust_scores = trust_scores.mean(dim=1)
            
        decisions = []
        for i in range(b):
            t = trust_scores[i].item()
            if self.policy == "threshold":
                thresh = self.cfg.threshold if self.cfg and hasattr(self.cfg, "threshold") else 0.5
                if t >= thresh:
                    decisions.append("Accept")
                else:
                    decisions.append("Refer to Human Expert")
            elif self.policy == "risk_coverage":
                if t >= self.empirical_threshold:
                    decisions.append("Accept")
                else:
                    decisions.append("Refer to Human Expert")
            elif self.policy == "utility":
                # Maximize expected utility
                # U = P(Correct) * U_correct + P(Incorrect) * U_incorrect
                # Approximating P(Correct) with Trust Score
                u_correct = 1.0
                u_incorrect = -5.0
                u_human = -0.5 # Cost of querying human
                
                expected_auto_utility = t * u_correct + (1 - t) * u_incorrect
                if expected_auto_utility > u_human:
                    decisions.append("Accept")
                else:
                    decisions.append("Refer to Human Expert")
            else:
                decisions.append("Accept")
                
        return {
            "decisions": decisions,
            "trust_scores": trust_scores.cpu().numpy().tolist()
        }


__all__ = ["DecisionEngine"]
