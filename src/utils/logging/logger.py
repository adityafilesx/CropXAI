"""
CropXAI Logging System
=======================

Unified logging backend integrating:
  - **Rich** for beautiful, structured console output
  - **TensorBoard** for experiment scalar tracking
  - **Weights & Biases** (optional) for cloud experiment management

Usage
-----
::

    from src.utils.logging.logger import get_logger, setup_logging

    # One-time setup (call at program entry)
    setup_logging(cfg)

    # Per-module usage
    logger = get_logger(__name__)
    logger.info("Dataset loaded: %d samples", n)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Rich
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Optional imports
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Custom Rich Theme
# ──────────────────────────────────────────────────────────────────────────────

CROPXAI_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bold green",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "repr.filename": "cyan",
        "repr.number": "bold magenta",
    }
)

console = Console(theme=CROPXAI_THEME, stderr=True)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(cfg: Any | None = None) -> None:
    """
    Configure the global logging system from a Hydra config node.

    This sets up:
    1. Rich console handler
    2. File handler (if ``cfg.logging.rich.log_file`` is set)

    Call once at application startup before any ``get_logger()`` calls.

    Parameters
    ----------
    cfg : OmegaConf DictConfig, optional
        Root CropXAI config. If None, sane defaults are used.
    """
    level_str: str = "INFO"
    log_file: str | None = None
    show_path: bool = False
    show_time: bool = True

    if cfg is not None:
        try:
            log_cfg = cfg.get("logging", {}).get("rich", {})
            level_str = log_cfg.get("level", "INFO")
            log_file = log_cfg.get("log_file", None)
            show_path = log_cfg.get("show_path", False)
            show_time = log_cfg.get("show_time", True)
        except Exception:
            pass

    level = getattr(logging, level_str.upper(), logging.INFO)

    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=show_path,
            show_time=show_time,
            markup=True,
        )
    ]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )

    # Quieten noisy third-party loggers
    for noisy in ["urllib3", "boto3", "botocore", "matplotlib", "PIL", "rasterio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Use ``__name__`` as the name."""
    return logging.getLogger(name)


# ──────────────────────────────────────────────────────────────────────────────
# TensorBoard Writer
# ──────────────────────────────────────────────────────────────────────────────

class TensorBoardLogger:
    """
    Thin wrapper around ``torch.utils.tensorboard.SummaryWriter``
    that degrades gracefully if TensorBoard is not installed.
    """

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = enabled and _TENSORBOARD_AVAILABLE
        self._writer: Any = None
        if self.enabled:
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled and self._writer:
            self._writer.add_scalar(tag, value, global_step=step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        if self.enabled and self._writer:
            self._writer.add_scalars(main_tag, tag_scalar_dict, global_step=step)

    def add_image(self, tag: str, img_tensor: Any, step: int) -> None:
        if self.enabled and self._writer:
            self._writer.add_image(tag, img_tensor, global_step=step)

    def add_figure(self, tag: str, figure: Any, step: int) -> None:
        if self.enabled and self._writer:
            self._writer.add_figure(tag, figure, global_step=step)

    def add_hparams(self, hparam_dict: dict, metric_dict: dict) -> None:
        if self.enabled and self._writer:
            self._writer.add_hparams(hparam_dict, metric_dict)

    def close(self) -> None:
        if self.enabled and self._writer:
            self._writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Weights & Biases Logger
# ──────────────────────────────────────────────────────────────────────────────

class WandBLogger:
    """
    Thin wrapper around ``wandb`` that degrades gracefully.

    Controlled by ``cfg.logging.wandb.enabled``.
    """

    def __init__(self, cfg: Any | None = None) -> None:
        self.enabled = False
        self._run = None

        if not _WANDB_AVAILABLE:
            return

        if cfg is None:
            return

        try:
            wandb_cfg = cfg.get("logging", {}).get("wandb", {})
            self.enabled = wandb_cfg.get("enabled", False)
        except Exception:
            return

        if self.enabled:
            try:
                self._run = wandb.init(
                    project=wandb_cfg.get("project", "cropxai"),
                    entity=wandb_cfg.get("entity", None),
                    name=wandb_cfg.get("name", None),
                    tags=list(wandb_cfg.get("tags", [])),
                    notes=wandb_cfg.get("notes", ""),
                    mode=wandb_cfg.get("mode", "online"),
                    save_code=wandb_cfg.get("save_code", True),
                )
            except Exception as exc:
                get_logger(__name__).warning("W&B init failed: %s", exc)
                self.enabled = False

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._run:
            wandb.log(data, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        if self.enabled and self._run:
            wandb.config.update(config, allow_val_change=True)

    def log_artifact(self, path: str, artifact_type: str = "dataset") -> None:
        if self.enabled and self._run:
            artifact = wandb.Artifact(Path(path).name, type=artifact_type)
            artifact.add_file(path)
            wandb.log_artifact(artifact)

    def finish(self) -> None:
        if self.enabled and self._run:
            wandb.finish()


# ──────────────────────────────────────────────────────────────────────────────
# Composite Logger
# ──────────────────────────────────────────────────────────────────────────────

class CropXAILogger:
    """
    Unified logger that fans out to Rich, TensorBoard, and W&B simultaneously.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root CropXAI config.
    log_dir : str | Path
        Directory for TensorBoard logs.
    """

    def __init__(self, cfg: Any, log_dir: str | Path) -> None:
        self._logger = get_logger("cropxai")
        self.tb = TensorBoardLogger(
            log_dir=Path(log_dir) / "tensorboard",
            enabled=cfg.get("logging", {}).get("tensorboard", {}).get("enabled", True),
        )
        self.wandb = WandBLogger(cfg=cfg)

    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._logger.warning(msg, *args)

    def error(self, msg: str, *args: Any) -> None:
        self._logger.error(msg, *args)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for key, val in metrics.items():
            self.tb.add_scalar(key, val, step)
        self.wandb.log(metrics, step=step)

    def finish(self) -> None:
        self.tb.close()
        self.wandb.finish()


__all__ = [
    "setup_logging",
    "get_logger",
    "TensorBoardLogger",
    "WandBLogger",
    "CropXAILogger",
    "console",
]
