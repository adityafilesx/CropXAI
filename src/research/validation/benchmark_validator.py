"""
CropXAI Benchmark Compliance Checker
====================================

Thin compatibility wrapper. The legacy class used to return success without inspecting anything; it now delegates to
`src.cropxai_analysis.compliance`, which recomputes every assertion from run records, declared configuration and
artifacts. `verify_compliance()` returns False (never True by default) when any critical check fails.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = (
    "configs/experiments/revision_v1.json",
    "configs/experiments/revision_v1_prithvi_ms6.json",
)


class BenchmarkComplianceChecker:
    """Verifies that run records satisfy the declared protocol matrix."""

    def __init__(self, config_paths: tuple[str, ...] | list[str] | None = None, database: Any = None,
                 protocol: Any = None) -> None:
        if database is not None or protocol is not None:
            logger.warning("Legacy 'database'/'protocol' arguments are ignored; records are read from results/raw.")
        self.config_paths = tuple(config_paths or DEFAULT_CONFIGS)
        self.report: dict[str, Any] | None = None

    def verify_compliance(self, **kwargs: Any) -> bool:
        from src.cropxai_analysis.compliance import check_all

        self.report = check_all(self.config_paths, **kwargs)
        for cohort in self.report["reports"]:
            for check in cohort["checks"]:
                if check["critical"] and not check["passed"]:
                    logger.error("Compliance FAILED [%s] %s: %s", cohort["cohort"], check["check"], check["detail"])
        return bool(self.report["ok"])


__all__ = ["BenchmarkComplianceChecker"]
