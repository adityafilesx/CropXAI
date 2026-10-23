"""
CropXAI Pipeline Architecture
==============================

Defines the composable pipeline system used throughout CropXAI.

Every data processing step is represented as a :class:`PipelineStage` —
a callable with a well-defined input/output contract. Stages are composed
into a :class:`Pipeline` that runs them sequentially, with optional
logging, timing, and error handling at each step.

Usage
-----
::

    from src.core.pipeline import Pipeline
    from src.research.dataset.downloader import DownloadStage
    from src.research.dataset.validator import ValidateStage

    pipeline = DatasetPipeline([
        DownloadStage(cfg),
        ValidateStage(cfg),
    ])
    result = pipeline.run(data)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Stage Result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """Encapsulates the output of a single pipeline stage."""

    stage_name: str
    data: Any
    elapsed_s: float
    success: bool
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"StageResult({status} {self.stage_name!r}, "
            f"elapsed={self.elapsed_s:.3f}s)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Stage
# ──────────────────────────────────────────────────────────────────────────────

class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.

    Subclass this and implement :meth:`process` to create a new stage.
    The :meth:`__call__` method wraps :meth:`process` with timing and logging.

    Parameters
    ----------
    name : str, optional
        Human-readable name for the stage. Defaults to the class name.
    enabled : bool
        If False, the stage is a no-op (data passes through unchanged).
    """

    def __init__(self, name: str | None = None, *, enabled: bool = True) -> None:
        self.name = name or self.__class__.__name__
        self.enabled = enabled

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        Core processing logic.

        Parameters
        ----------
        data : Any
            Input data from the previous stage (or None for the first stage).

        Returns
        -------
        Any
            Processed data to pass to the next stage.
        """
        ...

    def validate_input(self, data: Any) -> None:
        """
        Optional input validation hook.
        Override to add type/schema checks before processing.
        """

    def validate_output(self, data: Any) -> None:
        """
        Optional output validation hook.
        Override to add type/schema checks after processing.
        """

    def __call__(self, data: Any) -> StageResult:
        """Execute the stage with timing, logging, and error handling."""
        if not self.enabled:
            logger.debug("[Pipeline] Stage '%s' is disabled — skipping.", self.name)
            return StageResult(
                stage_name=self.name,
                data=data,
                elapsed_s=0.0,
                success=True,
                metadata={"skipped": True},
            )

        logger.info("[Pipeline] ▶ Starting stage: %s", self.name)
        t0 = time.perf_counter()

        try:
            self.validate_input(data)
            output = self.process(data)
            self.validate_output(output)
            elapsed = time.perf_counter() - t0

            logger.info(
                "[Pipeline] ✓ Stage '%s' completed in %.3fs.", self.name, elapsed
            )
            return StageResult(
                stage_name=self.name,
                data=output,
                elapsed_s=elapsed,
                success=True,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "[Pipeline] ✗ Stage '%s' failed after %.3fs: %s",
                self.name,
                elapsed,
                exc,
                exc_info=True,
            )
            return StageResult(
                stage_name=self.name,
                data=data,
                elapsed_s=elapsed,
                success=False,
                error=exc,
            )

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """
    A sequential pipeline that runs a list of :class:`PipelineStage` objects.

    Parameters
    ----------
    stages : list[PipelineStage]
        Ordered list of stages to execute.
    name : str
        Human-readable pipeline name for logging.
    fail_fast : bool
        If True, abort the pipeline on the first stage failure.
        If False, attempt to continue (data from the failed stage is passed unchanged).
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        name: str = "Pipeline",
        *,
        fail_fast: bool = True,
    ) -> None:
        self.stages = stages
        self.name = name
        self.fail_fast = fail_fast
        self._results: list[StageResult] = []

    def run(self, data: Any = None) -> Any:
        """
        Execute all stages sequentially.

        Parameters
        ----------
        data : Any
            Initial data passed to the first stage.

        Returns
        -------
        Any
            Output of the last successful stage.

        Raises
        ------
        RuntimeError
            If ``fail_fast=True`` and any stage fails.
        """
        self._results = []
        logger.info(
            "[%s] Starting pipeline with %d stage(s).", self.name, len(self.stages)
        )
        total_t0 = time.perf_counter()

        for stage in self.stages:
            result = stage(data)
            self._results.append(result)

            if not result.success:
                if self.fail_fast:
                    raise RuntimeError(
                        f"[{self.name}] Pipeline aborted at stage "
                        f"'{stage.name}': {result.error}"
                    ) from result.error
                logger.warning(
                    "[%s] Stage '%s' failed; continuing with previous data.",
                    self.name,
                    stage.name,
                )
            else:
                data = result.data

        total_elapsed = time.perf_counter() - total_t0
        n_ok = sum(1 for r in self._results if r.success)
        logger.info(
            "[%s] Pipeline finished. %d/%d stages succeeded in %.3fs.",
            self.name,
            n_ok,
            len(self.stages),
            total_elapsed,
        )
        return data

    def summary(self) -> str:
        """Return a formatted summary of the last run's stage results."""
        if not self._results:
            return f"[{self.name}] No run recorded yet."
        lines = [f"Pipeline '{self.name}' — {len(self._results)} stages:"]
        for r in self._results:
            status = "✓" if r.success else "✗"
            lines.append(f"  {status} {r.stage_name:<35} {r.elapsed_s:.3f}s")
        return "\n".join(lines)

    def __repr__(self) -> str:
        stage_names = [s.name for s in self.stages]
        return f"Pipeline(name={self.name!r}, stages={stage_names})"


__all__ = ["PipelineStage", "Pipeline", "StageResult"]
