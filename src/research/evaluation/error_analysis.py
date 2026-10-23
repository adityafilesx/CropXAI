"""
CropXAI Failure Gallery (Error Analyzer)
==========================================

Extracts and visualizes model failure modes, including:
  - Misclassifications
  - Confusion Heatmaps
  - Worst Predictions (lowest IoU)
  - Confident Errors
  - Rare Class Failures
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FailureGallery:
    """
    Analyzes model errors and prediction failures post-evaluation.
    Replaces the standard ErrorAnalyzer to provide deep qualitative insights.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.class_names = cfg.dataset.labels.classes
        self.worst_predictions = []
        self.confident_errors = []

    def update(self, images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor) -> None:
        """
        Track failures during the evaluation loop.
        In a real implementation, this would compute per-image IoU 
        and store the top-k worst examples for visualization.
        """
        pass

    def export_gallery(self, output_path: str) -> None:
        """Export the tracked failure modes to disk."""
        logger.info("Exporting Failure Gallery to %s...", output_path)
        # Placeholder for exporting images/heatmaps of worst predictions
        pass

    def plot_confusion_matrix(self, conf_matrix: np.ndarray, output_path: str) -> None:
        """Plot normalized confusion matrix."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from src.research.visualization.export import apply_publication_style, save_figure

        apply_publication_style()
        
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_cm = conf_matrix / row_sums

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            normalized_cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_title("Confusion Matrix (Normalized)")
        
        plt.tight_layout()
        save_figure(fig, "confusion_matrix", output_path)


__all__ = ["FailureGallery"]
