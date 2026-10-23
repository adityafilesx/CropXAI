"""
CropXAI Campaign Manager (ASCOPE)
===================================

The autonomous execution engine for the research campaign.
Supports pluggable executors (Local, SLURM) and tracks execution progress.
"""

from __future__ import annotations

import logging
from typing import Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseExecutor(ABC):
    """Abstract interface for executing experiments."""
    
    @abstractmethod
    def execute(self, experiment_id: str, config: dict[str, Any]) -> bool:
        """Executes a single experiment configuration."""
        pass


class LocalExecutor(BaseExecutor):
    """Executes experiments sequentially on the local machine."""
    
    def execute(self, experiment_id: str, config: dict[str, Any]) -> bool:
        logger.info("[LocalExecutor] Running Experiment %s: %s", experiment_id, config.get('model', 'unknown'))
        # Mocking actual execution
        return True


class SLURMExecutor(BaseExecutor):
    """Generates sbatch scripts and dispatches to a SLURM cluster."""
    
    def execute(self, experiment_id: str, config: dict[str, Any]) -> bool:
        logger.info("[SLURMExecutor] Dispatching Experiment %s to SLURM cluster...", experiment_id)
        # Mocking sbatch generation
        # script = f"sbatch --job-name={experiment_id} run_slurm.sh"
        # subprocess.run(script)
        return True


class CampaignManager:
    """Orchestrates the entire scientific campaign."""

    def __init__(self, executor: BaseExecutor | None = None) -> None:
        self.executor = executor or LocalExecutor()

    def run_campaign(self, configurations: list[dict[str, Any]]) -> None:
        """Dispatches all experiments in the campaign."""
        logger.info("Starting ASCOPE Campaign with %d configurations.", len(configurations))
        
        for i, cfg in enumerate(configurations):
            exp_id = f"EXP_{i:04d}"
            success = self.executor.execute(exp_id, cfg)
            
            if not success:
                logger.error("Experiment %s failed.", exp_id)
                
        logger.info("Campaign Execution Complete.")


__all__ = ["CampaignManager", "BaseExecutor", "LocalExecutor", "SLURMExecutor"]
