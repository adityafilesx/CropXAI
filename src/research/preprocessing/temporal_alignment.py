"""
CropXAI Temporal Alignment
============================

Aligns multi-temporal image sequences to a fixed number of timesteps.
Handles missing acquisitions via interpolation or fill strategies.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TemporalAligner:
    """
    Aligns a variable-length temporal sequence to a fixed number of timesteps.

    Parameters
    ----------
    target_length : int
        Target number of temporal frames.
    fill_strategy : str
        "linear" | "nearest" | "forward" | "zero"
    """

    def __init__(self, target_length: int, fill_strategy: str = "linear") -> None:
        self.target_length = target_length
        self.fill_strategy = fill_strategy

    @classmethod
    def from_cfg(cls, cfg: Any) -> "TemporalAligner":
        t_cfg = cfg.preprocessing.temporal_alignment
        return cls(
            target_length=cfg.dataset.temporal.num_timesteps,
            fill_strategy=t_cfg.get("fill_strategy", "linear"),
        )

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Align a temporal sequence.

        Parameters
        ----------
        sequence : torch.Tensor
            Shape: (T, C, H, W) — variable T.

        Returns
        -------
        torch.Tensor
            Shape: (target_length, C, H, W)
        """
        T, C, H, W = sequence.shape
        target = self.target_length

        if T == target:
            return sequence

        if T > target:
            # Subsample evenly
            indices = torch.linspace(0, T - 1, steps=target).long()
            return sequence[indices]

        # T < target: interpolate / pad
        return self._upsample(sequence, target)

    def _upsample(self, sequence: torch.Tensor, target: int) -> torch.Tensor:
        """Upsample a short sequence to target length."""
        T, C, H, W = sequence.shape

        if self.fill_strategy == "linear":
            # Use torch interpolation along temporal dim
            seq_5d = sequence.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, H, W, T)
            seq_5d = seq_5d.float()
            upsampled = torch.nn.functional.interpolate(
                seq_5d.view(1, C * H * W, T),
                size=target,
                mode="linear",
                align_corners=False,
            ).view(C, H, W, target).permute(3, 0, 1, 2)
            return upsampled

        elif self.fill_strategy == "forward":
            # Forward-fill (repeat last frame)
            out = list(sequence)
            while len(out) < target:
                out.append(out[-1])
            return torch.stack(out[:target])

        elif self.fill_strategy == "zero":
            pad = torch.zeros(target - T, C, H, W, dtype=sequence.dtype)
            return torch.cat([sequence, pad], dim=0)

        else:  # nearest
            indices = torch.round(torch.linspace(0, T - 1, steps=target)).long()
            return sequence[indices]


__all__ = ["TemporalAligner"]
