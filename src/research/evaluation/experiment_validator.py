"""
CropXAI Experiment Validator
==============================

Validates experiments for reproducibility by hashing the dataset, 
checkpoint, config, and environment states.
"""

from __future__ import annotations

import logging
import hashlib
import json
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ExperimentValidator:
    """Ensures rigorous experiment reproducibility."""

    @staticmethod
    def compute_config_hash(cfg: Any) -> str:
        """Computes a deterministic hash of the Hydra configuration."""
        # Convert to dict and sort keys
        from omegaconf import OmegaConf
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_str = json.dumps(cfg_dict, sort_keys=True)
        return hashlib.sha256(cfg_str.encode()).hexdigest()

    @staticmethod
    def get_environment_state() -> dict[str, str]:
        """Captures hardware and software state."""
        return {
            "python_version": sys.version.split(" ")[0],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        }

    @staticmethod
    def validate_experiment(cfg: Any, dataset_hash: str, checkpoint_hash: str) -> dict[str, Any]:
        """
        Validates the experiment before it starts.
        If all hashes match a previous run, it can be marked as REPRODUCED.
        """
        config_hash = ExperimentValidator.compute_config_hash(cfg)
        env_state = ExperimentValidator.get_environment_state()
        
        # Combine into a single unique experiment signature
        signature_payload = f"{config_hash}-{dataset_hash}-{checkpoint_hash}-{cfg.seed}"
        experiment_signature = hashlib.sha256(signature_payload.encode()).hexdigest()
        
        logger.info("Experiment Signature: %s", experiment_signature)
        logger.info("Environment: %s", env_state)
        
        return {
            "experiment_signature": experiment_signature,
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "checkpoint_hash": checkpoint_hash,
            "environment": env_state,
            "status": "NEW" # In a real system, we query a DB to see if it's NEW or REPRODUCED
        }


__all__ = ["ExperimentValidator"]
