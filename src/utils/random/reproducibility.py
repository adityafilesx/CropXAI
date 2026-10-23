"""
CropXAI Reproducibility Utilities
===================================

Provides a single entry point ``set_global_seed()`` for fully reproducible
experiments, plus utilities to capture and export the execution environment.

Usage
-----
::

    from src.utils.random.reproducibility import set_global_seed, ReproducibilityManager

    # At the start of every experiment:
    manager = ReproducibilityManager(seed=42, deterministic=True)
    manager.setup()
    manager.capture_and_log(run_dir)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Optional PyTorch import
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Seed Setting
# ──────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """
    Set global random seeds for full reproducibility.

    Affects:
    - Python ``random`` module
    - NumPy
    - PyTorch (CPU + CUDA)
    - CUDA deterministic algorithms (if ``deterministic=True``)

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic : bool
        If True, enable CUDA deterministic mode.
        This may reduce performance but ensures reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            torch.backends.cudnn.benchmark = True

    logger.info(
        "🌱 Global seed set to %d (deterministic=%s).", seed, deterministic
    )


# ──────────────────────────────────────────────────────────────────────────────
# Environment Capture
# ──────────────────────────────────────────────────────────────────────────────

def capture_environment() -> dict[str, Any]:
    """
    Capture the full execution environment for reproducibility logging.

    Returns
    -------
    dict
        Dictionary containing platform, Python, key package versions,
        hardware info, and timestamp.
    """
    env: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
        },
        "packages": _get_package_versions(),
        "hardware": _get_hardware_info(),
        "git": _get_git_info(),
        "environment_variables": _get_safe_env_vars(),
    }
    return env


def _get_package_versions() -> dict[str, str]:
    """Get versions of key packages."""
    packages_of_interest = [
        "torch", "torchvision", "pytorch_lightning",
        "torchgeo", "rasterio", "geopandas",
        "numpy", "pandas", "scikit-learn",
        "albumentations", "matplotlib", "plotly",
        "hydra-core", "omegaconf", "rich",
        "wandb", "tensorboard",
    ]
    versions: dict[str, str] = {}
    for pkg in packages_of_interest:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "not installed"
    return versions


def _get_hardware_info() -> dict[str, Any]:
    """Get CPU, RAM, and GPU information."""
    import psutil  # type: ignore[import]

    hw: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }

    if _TORCH_AVAILABLE:
        hw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["cuda_device_count"] = torch.cuda.device_count()
            hw["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 2
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        hw["mps_available"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    return hw


def _get_git_info() -> dict[str, str]:
    """Get git repository information (gracefully handles non-git directories)."""
    info: dict[str, str] = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["status"] = subprocess.check_output(
            ["git", "status", "--short"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["commit"] = "N/A (not a git repository)"
    return info


def _get_safe_env_vars() -> dict[str, str]:
    """Capture non-sensitive environment variables relevant to the project."""
    safe_keys = [
        "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "DATA_ROOT",
        "EXPERIMENT_ROOT", "HYDRA_FULL_ERROR",
    ]
    return {k: os.environ.get(k, "") for k in safe_keys}


# ──────────────────────────────────────────────────────────────────────────────
# Config Snapshot
# ──────────────────────────────────────────────────────────────────────────────

def snapshot_config(cfg: Any, output_path: str | Path) -> str:
    """
    Save an OmegaConf config to YAML and return its SHA-256 hash.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Hydra config to snapshot.
    output_path : str | Path
        Where to write the config YAML.

    Returns
    -------
    str
        SHA-256 hash of the serialised config (useful for deduplication).
    """
    from omegaconf import OmegaConf  # type: ignore[import]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg_str = OmegaConf.to_yaml(cfg, resolve=True)
    output.write_text(cfg_str, encoding="utf-8")

    config_hash = hashlib.sha256(cfg_str.encode()).hexdigest()
    logger.info("Config snapshot saved → %s (hash: %s...)", output, config_hash[:12])
    return config_hash


def export_dependencies(output_path: str | Path) -> None:
    """
    Export the current Python environment's frozen dependency list.

    Parameters
    ----------
    output_path : str | Path
        Path to write ``requirements_frozen.txt``.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True,
        )
        Path(output_path).write_text(result.stdout, encoding="utf-8")
        logger.info("Frozen dependencies exported → %s", output_path)
    except subprocess.CalledProcessError as exc:
        logger.warning("Could not export dependencies: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility Manager
# ──────────────────────────────────────────────────────────────────────────────

class ReproducibilityManager:
    """
    Convenience class that bundles all reproducibility operations.

    Parameters
    ----------
    seed : int
        Random seed. Default 42.
    deterministic : bool
        Enable deterministic algorithms. Default True.
    """

    def __init__(self, seed: int = 42, *, deterministic: bool = True) -> None:
        self.seed = seed
        self.deterministic = deterministic

    def setup(self) -> None:
        """Set global seed and configure deterministic mode."""
        set_global_seed(self.seed, deterministic=self.deterministic)

    def capture_and_log(
        self,
        run_dir: str | Path,
        cfg: Any | None = None,
    ) -> dict[str, Any]:
        """
        Capture environment, snapshot config, and export dependencies.

        Parameters
        ----------
        run_dir : str | Path
            Root directory for this experiment run.
        cfg : OmegaConf DictConfig, optional
            Hydra config to snapshot.

        Returns
        -------
        dict
            Environment capture dictionary.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        env = capture_environment()

        # Save environment info
        env_path = run_dir / "environment.json"
        env_path.write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
        logger.info("Environment info saved → %s", env_path)

        # Snapshot config
        if cfg is not None:
            try:
                config_hash = snapshot_config(cfg, run_dir / "config_snapshot.yaml")
                env["config_hash"] = config_hash
            except Exception as exc:
                logger.warning("Config snapshot failed: %s", exc)

        # Export dependencies
        export_dependencies(run_dir / "requirements_frozen.txt")

        return env


__all__ = [
    "set_global_seed",
    "capture_environment",
    "snapshot_config",
    "export_dependencies",
    "ReproducibilityManager",
]
