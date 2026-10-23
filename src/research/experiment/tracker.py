"""
CropXAI Experiment Tracker
============================

Manages experiment lifecycles:
  - Generates unique run directories
  - Captures environment, hardware, and Git state
  - Snapshots configuration (Hydra)
  - Manages checkpoints and artifacts
  - Provides a unified logger (Rich + TensorBoard + W&B)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.logging.logger import CropXAILogger
from src.utils.random.reproducibility import ReproducibilityManager

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Central experiment tracker and lifecycle manager.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root config.
    run_id : str | None
        Unique run ID (if None, generated via config timestamp).
    """

    def __init__(self, cfg: Any, run_id: str | None = None) -> None:
        self.cfg = cfg
        self.run_id = run_id or cfg.experiment.run_id
        self.exp_name = cfg.experiment.name

        # Paths
        exp_root = Path(cfg.paths.experiments)
        self.run_dir = exp_root / self.exp_name / self.run_id
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.viz_dir = self.run_dir / "visualizations"
        self.log_dir = self.run_dir / "logs"

        # Managers
        self.reproducibility = ReproducibilityManager(
            seed=cfg.experiment.seed,
            deterministic=cfg.experiment.deterministic,
        )
        self._logger: CropXAILogger | None = None

    def setup(self) -> None:
        """Initialize the experiment directory, seeds, and loggers."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds
        self.reproducibility.setup()

        # Capture environment and snapshot config
        self.reproducibility.capture_and_log(self.run_dir, self.cfg)

        # Initialize loggers
        self._logger = CropXAILogger(self.cfg, self.log_dir)
        self._logger.info("Experiment '%s' started. Run ID: %s", self.exp_name, self.run_id)
        self._logger.info("Run directory: %s", self.run_dir)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log scalars to all configured backends (TB/W&B)."""
        if self._logger:
            self._logger.log_metrics(metrics, step)

    def finish(self) -> None:
        """Tear down and close loggers."""
        if self._logger:
            self._logger.info("Experiment '%s' finished.", self.exp_name)
            self._logger.finish()

    @property
    def logger(self) -> CropXAILogger:
        if self._logger is None:
            raise RuntimeError("ExperimentTracker not setup. Call setup() first.")
        return self._logger


__all__ = ["ExperimentTracker"]
