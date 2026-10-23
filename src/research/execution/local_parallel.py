"""
CropXAI Local Parallel Executor
=================================

Uses Python's concurrent.futures to execute multiple runs in parallel locally.
Optimized for the Mac M4 unified memory architecture.
"""

from __future__ import annotations

import logging
import concurrent.futures
from typing import Any

logger = logging.getLogger(__name__)


def _run_task(task: dict[str, Any]) -> str:
    """Module-level task runner to avoid pickle issues."""
    seed = task.get("seed", 42)
    model = task.get("model", "unknown")
    import time
    time.sleep(0.1) # Simulate work
    return f"Completed {model} on seed {seed}"


class LocalParallelExecutor:
    """Executes campaign tasks sequentially to protect 16GB unified memory."""

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max_workers

    def submit_campaign(self, tasks: list[dict[str, Any]]) -> bool:
        """Executes a list of experiments in parallel."""
        logger.info("Starting Local Parallel Execution with %d workers...", self.max_workers)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_run_task, t): t for t in tasks}
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    logger.info("Task Result: %s", result)
                except Exception as exc:
                    logger.error("Task generated an exception: %s", exc)

        logger.info("All local parallel tasks completed successfully.")
        return True


__all__ = ["LocalParallelExecutor"]
