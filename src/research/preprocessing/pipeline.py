"""
CropXAI Preprocessing Pipeline
=================================

Composes all preprocessing steps into a single callable transform.
Built from Hydra config — each stage is individually enabled/disabled.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Composes enabled preprocessing steps into a callable transform.

    This is designed to be used as the ``transform`` argument in
    ``CropXAIDataset``.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root CropXAI config.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.steps: list[tuple[str, Callable]] = []
        self._build_steps()

    def _build_steps(self) -> None:
        """Instantiate enabled preprocessing steps in pipeline order."""
        preproc = self.cfg.preprocessing

        if not preproc.get("enabled", True):
            logger.info("Preprocessing pipeline disabled.")
            return

        if preproc.dtype_conversion.get("enabled", True):
            from src.research.preprocessing.dtype_converter import DTypeConverter
            self.steps.append(("dtype_convert", DTypeConverter.from_cfg(self.cfg)))

        if preproc.band_selection.get("enabled", True):
            from src.research.preprocessing.band_selector import BandSelector
            self.steps.append(("band_select", BandSelector.from_cfg(self.cfg)))

        if preproc.missing_values.get("enabled", True):
            from src.research.preprocessing.missing_value_handler import MissingValueHandler
            self.steps.append(("missing_values", MissingValueHandler.from_cfg(self.cfg)))

        if preproc.normalization.get("enabled", True):
            from src.research.preprocessing.normalization import build_normalizer
            norm = build_normalizer(self.cfg)
            if norm is not None:
                self.steps.append(("normalize", norm))

        if preproc.resize.get("enabled", False):
            from src.research.preprocessing.resizer import ImageResizer
            self.steps.append(("resize", ImageResizer.from_cfg(self.cfg)))

        logger.info(
            "Preprocessing pipeline: %d active steps: %s",
            len(self.steps),
            [name for name, _ in self.steps],
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply all preprocessing steps sequentially."""
        for name, step in self.steps:
            try:
                image = step(image)
            except Exception as exc:
                logger.warning("Preprocessing step '%s' failed: %s", name, exc)
        return image

    def __repr__(self) -> str:
        return f"PreprocessingPipeline(steps={[n for n, _ in self.steps]})"


__all__ = ["PreprocessingPipeline"]
