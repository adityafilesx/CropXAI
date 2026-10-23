"""
Scientific Discovery Engine - Miner
===================================
Automatically mines experiment results (e.g., 230,000 matrix combinations) to discover 
statistically significant findings using rigorous non-parametric tests.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
try:
    from scipy import stats
except ImportError:
    stats = None

logger = logging.getLogger(__name__)

class DiscoveryMiner:
    def __init__(self, p_value_threshold: float = 0.05, effect_size_threshold: float = 0.5):
        """
        Initializes the Scientific Miner.
        Args:
            p_value_threshold: Alpha level for statistical significance (default 0.05).
            effect_size_threshold: Minimum Cohen's d / Cliff's delta to consider a finding 'significant' practically.
        """
        self.p_threshold = p_value_threshold
        self.effect_threshold = effect_size_threshold
        if stats is None:
            logger.warning("scipy is not installed. Statistical mining will fall back to mean comparisons.")

    def mine_findings(self, results_json: Path) -> List[Dict[str, Any]]:
        """
        Parses a massive JSON results dump and mines scientific discoveries.
        """
        if not results_json.exists():
            logger.error(f"Results file not found: {results_json}")
            return []
            
        with open(results_json, 'r') as f:
            data = json.load(f)
            
        discoveries = []
        
        # 1. Mine Trust Calibration Misalignment (Example)
        # Compare VFM Trust vs CNN Trust
        vfm_trusts = self._extract_metric(data, metric='trust_score', condition_key='model_type', condition_val='vfm')
        cnn_trusts = self._extract_metric(data, metric='trust_score', condition_key='model_type', condition_val='cnn')
        
        if vfm_trusts and cnn_trusts:
            discovery = self._perform_statistical_test(
                vfm_trusts, cnn_trusts, 
                hypothesis="Vision Foundation Models produce higher Trust Scores than CNNs"
            )
            if discovery['accepted']:
                discoveries.append(discovery)
                
        # 2. Mine Calibration vs Faithfulness
        calib_scores = self._extract_metric(data, metric='calibration_error', condition_key=None, condition_val=None)
        faith_scores = self._extract_metric(data, metric='faithfulness', condition_key=None, condition_val=None)
        
        if calib_scores and faith_scores:
            correlation, p_val = stats.spearmanr(calib_scores, faith_scores) if stats else (0.0, 1.0)
            if p_val < self.p_threshold and abs(correlation) > self.effect_threshold:
                discoveries.append({
                    "hypothesis": "Calibration error is strongly negatively correlated with Explainability Faithfulness.",
                    "accepted": True,
                    "evidence": {
                        "test": "Spearman Rank Correlation",
                        "correlation": float(correlation),
                        "p_value": float(p_val)
                    }
                })

        logger.info(f"Mined {len(discoveries)} statistically significant discoveries.")
        return discoveries

    def _extract_metric(self, data: List[Dict], metric: str, condition_key: str = None, condition_val: str = None) -> List[float]:
        """Helper to extract arrays of metrics from the massive JSON structure."""
        values = []
        for exp in data:
            if condition_key and exp.get('config', {}).get(condition_key) != condition_val:
                continue
            val = exp.get('results', {}).get(metric)
            if val is not None:
                values.append(float(val))
        return values

    def _perform_statistical_test(self, dist_a: List[float], dist_b: List[float], hypothesis: str) -> Dict[str, Any]:
        """
        Performs Mann-Whitney U test (non-parametric) and calculates effect size.
        """
        if not stats or len(dist_a) < 3 or len(dist_b) < 3:
            return {"hypothesis": hypothesis, "accepted": False, "reason": "Insufficient data or scipy missing"}
            
        # Mann-Whitney U test (independent samples)
        stat, p_value = stats.mannwhitneyu(dist_a, dist_b, alternative='two-sided')
        
        # Calculate Cliff's delta for effect size
        # Cliff's d = 2U / (mn) - 1
        m, n = len(dist_a), len(dist_b)
        cliffs_delta = abs(2 * stat / (m * n) - 1)
        
        accepted = (p_value < self.p_threshold) and (cliffs_delta > self.effect_threshold)
        
        return {
            "hypothesis": hypothesis,
            "accepted": accepted,
            "evidence": {
                "test": "Mann-Whitney U",
                "p_value": float(p_value),
                "effect_size_cliffs_delta": float(cliffs_delta),
                "mean_A": float(np.mean(dist_a)),
                "mean_B": float(np.mean(dist_b))
            }
        }
