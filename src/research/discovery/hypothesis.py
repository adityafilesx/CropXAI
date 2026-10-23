"""
Scientific Hypothesis Generator
===============================
Automatically generates scientific hypotheses from correlations and verifies them.
Classifies hypotheses as Accept, Reject, or Inconclusive based on statistical tests.
"""

import logging
from typing import Dict, List, Any
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

logger = logging.getLogger(__name__)

class HypothesisEngine:
    def __init__(self, p_value_threshold: float = 0.01):
        self.p_threshold = p_value_threshold
        
    def generate_and_verify(self, feature_a_name: str, feature_a_data: List[float], 
                            feature_b_name: str, feature_b_data: List[float]) -> Dict[str, Any]:
        """
        Generates a directional hypothesis based on Pearson/Spearman correlation
        and verifies it using a strict p-value threshold.
        """
        if not stats or len(feature_a_data) != len(feature_b_data) or len(feature_a_data) < 5:
            return {"status": "Inconclusive", "reason": "Insufficient data"}
            
        correlation, p_val = stats.pearsonr(feature_a_data, feature_b_data)
        
        if correlation > 0.6:
            direction = "predicts higher"
        elif correlation < -0.6:
            direction = "predicts lower"
        else:
            return {
                "hypothesis": f"High {feature_a_name} does not strongly predict {feature_b_name}.",
                "status": "Reject",
                "p_value": float(p_val)
            }
            
        hypothesis_str = f"High {feature_a_name} {direction} {feature_b_name}."
        
        if p_val < self.p_threshold:
            status = "Accept"
            logger.info(f"Hypothesis ACCEPTED: {hypothesis_str}")
        else:
            status = "Inconclusive"
            logger.info(f"Hypothesis INCONCLUSIVE (p={p_val:.4f}): {hypothesis_str}")
            
        return {
            "hypothesis": hypothesis_str,
            "status": status,
            "evidence": {
                "correlation_coefficient": float(correlation),
                "p_value": float(p_val)
            }
        }
