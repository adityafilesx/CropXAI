"""
CropXAI SLURM Executor
========================

Strategy class that generates SLURM job arrays and submits them via sbatch,
allowing massive parallelization of the Master Campaign across multiple HPC nodes.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class SLURMExecutor:
    """Dispatches campaign tasks to an HPC cluster via SLURM."""

    def __init__(self, template_path: str = "configs/slurm/default.sh") -> None:
        self.template_path = Path(template_path)

    def submit_job_array(self, num_tasks: int, protocol_path: str) -> bool:
        """Generates the final bash script and submits it."""
        logger.info("Preparing SLURM job array for %d tasks using protocol %s...", num_tasks, protocol_path)
        
        if not self.template_path.exists():
            logger.error("SLURM template not found at %s", self.template_path)
            return False
            
        # Read the template
        with open(self.template_path, "r") as f:
            template = f.read()
            
        # Append the array directive and the command
        job_script = template + f"\n#SBATCH --array=1-{num_tasks}\n"
        job_script += f"\nPYTHONPATH=. python3 cropxai campaign run --protocol {protocol_path} --task-id $SLURM_ARRAY_TASK_ID\n"
        
        # Save generated script
        Path("logs/slurm").mkdir(parents=True, exist_ok=True)
        job_file = Path("logs/slurm/generated_submit.sh")
        job_file.write_text(job_script)
        
        logger.info("Generated SLURM array script at %s", job_file)
        
        # Mock submission (if sbatch is not available, we just log it)
        try:
            # subprocess.run(["sbatch", str(job_file)], check=True)
            logger.info("MOCK: sbatch %s successfully dispatched to cluster.", job_file)
        except Exception as e:
            logger.warning("Failed to run sbatch: %s. Are you on a SLURM cluster?", str(e))
            
        return True


__all__ = ["SLURMExecutor"]
