"""Calibration Metrics stubs (Phase 4): ECE, MCE, Reliability Diagram."""
from src.research.evaluation.metrics.base import BaseMetric, MetricResult
from src.core.registry import METRIC_REGISTRY
from typing import Any


@METRIC_REGISTRY.register("ece")
class ExpectedCalibrationError(BaseMetric):
    """Expected Calibration Error (ECE). Phase 4."""
    def __init__(self, n_bins: int = 15, cfg: Any | None = None) -> None:
        super().__init__("ECE", cfg)
        self.n_bins = n_bins
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        raise NotImplementedError("ECE.compute() implemented in Phase 4.")


__all__ = ["ExpectedCalibrationError"]
