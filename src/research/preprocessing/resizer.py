"""CropXAI Image Resizer."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Any


class ImageResizer:
    def __init__(self, target_size: tuple[int, int] = (224, 224), mode: str = "bilinear") -> None:
        self.target_size = target_size
        self.mode = mode

    @classmethod
    def from_cfg(cls, cfg: Any) -> "ImageResizer":
        r_cfg = cfg.preprocessing.resize
        return cls(target_size=tuple(r_cfg.target_size), mode=r_cfg.get("method", "bilinear"))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Resize (C, H, W) tensor to target_size."""
        img = image.unsqueeze(0).float()  # (1, C, H, W)
        align = self.mode in ("bilinear", "bicubic")
        out = F.interpolate(img, size=self.target_size, mode=self.mode, align_corners=align if align else None)
        return out.squeeze(0)


__all__ = ["ImageResizer"]
