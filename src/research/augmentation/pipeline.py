"""
CropXAI Augmentation Pipeline
================================

Builds an Albumentations augmentation pipeline from Hydra config.
All 9 augmentations are configurable via ``configs/augmentation/default.yaml``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_augmentation_pipeline(cfg: Any) -> Any | None:
    """
    Build an Albumentations ``Compose`` pipeline from the augmentation config.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root config (reads ``cfg.augmentation``).

    Returns
    -------
    albumentations.Compose | None
        The augmentation pipeline, or None if augmentation is disabled.
    """
    try:
        import albumentations as A
    except ImportError:
        logger.warning("albumentations not installed — augmentation disabled.")
        return None

    aug_cfg = cfg.augmentation
    if not aug_cfg.get("enabled", True):
        return None

    transforms = []

    # ── Geometric ─────────────────────────────────────────────────────────────
    if aug_cfg.horizontal_flip.get("enabled", True):
        transforms.append(A.HorizontalFlip(p=aug_cfg.horizontal_flip.p))

    if aug_cfg.vertical_flip.get("enabled", True):
        transforms.append(A.VerticalFlip(p=aug_cfg.vertical_flip.p))

    if aug_cfg.random_rotation.get("enabled", True):
        transforms.append(
            A.Rotate(
                limit=aug_cfg.random_rotation.limit,
                p=aug_cfg.random_rotation.p,
                border_mode=4,  # BORDER_REFLECT
            )
        )

    if aug_cfg.random_crop.get("enabled", True):
        transforms.append(
            A.RandomResizedCrop(
                size=(aug_cfg.random_crop.height, aug_cfg.random_crop.width),
                p=aug_cfg.random_crop.p,
            )
        )

    if aug_cfg.random_scale.get("enabled", True):
        transforms.append(
            A.RandomScale(
                scale_limit=aug_cfg.random_scale.scale_limit,
                p=aug_cfg.random_scale.p,
            )
        )

    # ── Radiometric ───────────────────────────────────────────────────────────
    if aug_cfg.brightness.get("enabled", True):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.brightness.limit,
                contrast_limit=0.0,
                p=aug_cfg.brightness.p,
            )
        )

    if aug_cfg.contrast.get("enabled", True):
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.0,
                contrast_limit=aug_cfg.contrast.limit,
                p=aug_cfg.contrast.p,
            )
        )

    # ── Noise ─────────────────────────────────────────────────────────────────
    if aug_cfg.gaussian_noise.get("enabled", True):
        var_limit = tuple(aug_cfg.gaussian_noise.var_limit)
        transforms.append(A.GaussNoise(var_limit=var_limit, p=aug_cfg.gaussian_noise.p))

    if aug_cfg.random_blur.get("enabled", True):
        blur_limit = tuple(aug_cfg.random_blur.blur_limit)
        transforms.append(A.Blur(blur_limit=blur_limit, p=aug_cfg.random_blur.p))

    if not transforms:
        return None

    pipeline = A.Compose(transforms)
    logger.info("Augmentation pipeline: %d active transforms.", len(transforms))
    return pipeline


__all__ = ["build_augmentation_pipeline"]
