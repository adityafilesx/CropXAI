"""
CropXAI Metrics Base Interface
================================

Abstract base class that all metrics must implement.
Enforces a consistent API across prediction, explainability,
calibration, efficiency, and statistical metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """Container for a metric computation result."""

    metric_name: str
    value: float | dict[str, float]
    unit: str = ""
    higher_is_better: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MetricResult({self.metric_name}={self.value!r})"


class BaseMetric(ABC):
    """
    Abstract base class for all CropXAI metrics.

    All metrics — prediction, XAI, calibration, efficiency — must
    subclass this and implement :meth:`compute`.

    Parameters
    ----------
    name : str
        Human-readable metric name.
    cfg : Any, optional
        Metric-specific configuration.
    """

    def __init__(self, name: str, cfg: Any | None = None) -> None:
        self.name = name
        self.cfg = cfg
        self._results: list[MetricResult] = []

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        """
        Compute the metric.

        Parameters are metric-specific. Implementations should accept
        predictions and targets (for prediction metrics), or model outputs
        and explanations (for XAI metrics), etc.

        Returns
        -------
        MetricResult
        """
        ...

    def reset(self) -> None:
        """Reset accumulated state (for streaming/batched computation)."""
        self._results = []

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Accumulate a batch (optional for streaming metrics).
        Override for metrics that accumulate over batches.
        """

    def finalize(self) -> MetricResult:
        """
        Finalize streaming computation and return aggregated result.
        Override for metrics that accumulate over batches.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming computation. "
            "Call compute() directly."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


__all__ = ["BaseMetric", "MetricResult"]
