"""
CropXAI Explainability Metrics — Interface Stubs (Phase 4)

Planned:
  - Faithfulness (perturbation-based)
  - Localization (pointing game)
  - Complexity (explanation entropy)
  - Stability (sensitivity to perturbations)
"""
from src.research.evaluation.metrics.base import BaseMetric, MetricResult
from src.core.registry import METRIC_REGISTRY
from typing import Any


@METRIC_REGISTRY.register("faithfulness")
class Faithfulness(BaseMetric):
    """Faithfulness: how much the explanation reflects model behavior. Phase 4."""
    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__("Faithfulness", cfg)
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        raise NotImplementedError("Faithfulness.compute() implemented in Phase 4.")


@METRIC_REGISTRY.register("localization")
class Localization(BaseMetric):
    """Localization: pointing game score for saliency maps. Phase 4."""
    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__("Localization", cfg)
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        raise NotImplementedError("Localization.compute() implemented in Phase 4.")


__all__ = ["Faithfulness", "Localization"]
