"""Efficiency Metrics stubs (Phase 3): FLOPs, latency, peak memory."""
from src.research.evaluation.metrics.base import BaseMetric, MetricResult
from src.core.registry import METRIC_REGISTRY
from typing import Any


@METRIC_REGISTRY.register("flops")
class FLOPs(BaseMetric):
    """FLOPs counter per forward pass. Phase 3."""
    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__("FLOPs", cfg)
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        raise NotImplementedError("FLOPs.compute() implemented in Phase 3.")


@METRIC_REGISTRY.register("latency")
class Latency(BaseMetric):
    """Inference latency (ms/batch). Phase 3."""
    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__("Latency", cfg)
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        raise NotImplementedError("Latency.compute() implemented in Phase 3.")


__all__ = ["FLOPs", "Latency"]
