"""Statistical Metrics stubs: KS test, distribution comparison."""
from src.research.evaluation.metrics.base import BaseMetric, MetricResult
from src.core.registry import METRIC_REGISTRY
from typing import Any


@METRIC_REGISTRY.register("ks_test")
class KolmogorovSmirnovTest(BaseMetric):
    """KS test for distribution comparison."""
    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__("KS_Test", cfg)
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        raise NotImplementedError("KS_Test.compute() implemented in Phase 2.")


__all__ = ["KolmogorovSmirnovTest"]
