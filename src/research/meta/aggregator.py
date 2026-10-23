"""
Meta Analysis Aggregator
========================
Aggregates massive experiment grids to compute overall Bayesian rankings, 
Pareto Frontiers, and Critical Difference (CD) Diagrams.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class MetaAggregator:
    def __init__(self):
        pass

    def compute_pareto_frontier(self, models_data: List[Dict[str, float]], x_metric: str, y_metric: str) -> List[str]:
        """
        Computes the Pareto frontier of models maximizing both x_metric and y_metric.
        (e.g., Accuracy vs Trust)
        """
        pareto_front = []
        for i, model_a in enumerate(models_data):
            is_dominated = False
            for j, model_b in enumerate(models_data):
                if i == j:
                    continue
                # model_b dominates model_a if it's better or equal in both metrics, and strictly better in one
                better_x = model_b[x_metric] >= model_a[x_metric]
                better_y = model_b[y_metric] >= model_a[y_metric]
                strictly_better = (model_b[x_metric] > model_a[x_metric]) or (model_b[y_metric] > model_a[y_metric])
                
                if better_x and better_y and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(model_a['model_name'])
                
        return pareto_front

    def compute_bayesian_ranking(self, results_matrix: np.ndarray, model_names: List[str]) -> List[Tuple[str, float]]:
        """
        Computes a mock Bayesian ranking (Plackett-Luce or Bradley-Terry) for N models.
        In production, this would use PyMC3 or similar to sample the posterior.
        """
        # For prototype, we compute mean rank across datasets/seeds.
        mean_scores = np.mean(results_matrix, axis=1) # Shape: (Models,)
        
        ranked_indices = np.argsort(mean_scores)[::-1]
        ranking = [(model_names[i], float(mean_scores[i])) for i in ranked_indices]
        
        logger.info(f"Bayesian Ranking Computed for {len(model_names)} models.")
        return ranking
