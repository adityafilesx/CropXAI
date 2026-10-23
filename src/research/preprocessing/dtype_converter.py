"""CropXAI Data Type Converter — safely converts image arrays to float32."""
from __future__ import annotations
import numpy as np
import torch
from typing import Any


class DTypeConverter:
    def __init__(self, output_dtype: str = "float32", scale: bool = True, max_val: float = 10000.0) -> None:
        self.output_dtype = output_dtype
        self.scale = scale
        self.max_val = max_val

    @classmethod
    def from_cfg(cls, cfg: Any) -> "DTypeConverter":
        d_cfg = cfg.preprocessing.dtype_conversion
        return cls(
            output_dtype=d_cfg.get("output_dtype", "float32"),
            scale=d_cfg.get("scale", True),
        )

    def __call__(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = image.float()
        if self.scale:
            image = image / self.max_val
        return image


__all__ = ["DTypeConverter"]
