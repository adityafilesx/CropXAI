"""
CropXAI Prediction Metrics — Interface Stubs
=============================================

Interface definitions for prediction quality metrics.
Implementations will be added in Phase 2 when models are introduced.

Metrics planned:
  - Overall Accuracy (OA)
  - Mean Class Accuracy (MCA)
  - F1 Score (macro, weighted, per-class)
  - Mean Intersection-over-Union (mIoU)
  - Cohen's Kappa
  - Confusion Matrix
"""

from __future__ import annotations

from typing import Any

from src.research.evaluation.metrics.base import BaseMetric, MetricResult
from src.core.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register("overall_accuracy")
class OverallAccuracy(BaseMetric):
    """
    Overall Accuracy (OA) for semantic segmentation.

    Phase 1: Interface stub. Implementation in Phase 2.
    """

    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__("OverallAccuracy", cfg)
        self.num_classes: int = cfg.dataset.labels.num_classes if cfg else 7
        self.ignore_index: int = 255

    def compute(self, predictions: Any, targets: Any) -> MetricResult:
        raise NotImplementedError(
            "OverallAccuracy.compute() will be implemented in Phase 2 "
            "when model predictions are available."
        )


@METRIC_REGISTRY.register("mean_iou")
class MeanIoU(BaseMetric):
    """
    Mean Intersection-over-Union (mIoU) for semantic segmentation.

    Phase 1: Interface stub.
    """

    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__("MeanIoU", cfg)
        self.num_classes: int = cfg.dataset.labels.num_classes if cfg else 7

    def compute(self, predictions: Any, targets: Any) -> MetricResult:
        raise NotImplementedError(
            "MeanIoU.compute() will be implemented in Phase 2."
        )


@METRIC_REGISTRY.register("f1_score")
class F1Score(BaseMetric):
    """
    F1 Score (macro average) for multi-class classification.

    Phase 1: Interface stub.
    """

    def __init__(self, average: str = "macro", cfg: Any | None = None) -> None:
        super().__init__(f"F1Score_{average}", cfg)
        self.average = average

    def compute(self, predictions: Any, targets: Any) -> MetricResult:
        raise NotImplementedError(
            "F1Score.compute() will be implemented in Phase 2."
        )


@METRIC_REGISTRY.register("cohen_kappa")
class CohenKappa(BaseMetric):
    """Cohen's Kappa coefficient. Phase 1: Interface stub."""

    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__("CohenKappa", cfg)

    def compute(self, predictions: Any, targets: Any) -> MetricResult:
        raise NotImplementedError("CohenKappa.compute() will be implemented in Phase 2.")


__all__ = ["OverallAccuracy", "MeanIoU", "F1Score", "CohenKappa"]
