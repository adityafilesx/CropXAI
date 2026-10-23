"""CropXAI Temporal Sequence Plotter."""
from __future__ import annotations
import logging
from typing import Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from src.research.visualization.export import apply_publication_style, save_figure
logger = logging.getLogger(__name__)


def plot_temporal_sequence(
    sequence: np.ndarray,
    dates: list[str] | None = None,
    rgb_indices: list[int] = [2, 1, 0],
    output_dir: str | None = None,
) -> Any:
    """
    Plot a temporal sequence of satellite images as a horizontal strip.

    Parameters
    ----------
    sequence : np.ndarray
        Shape (T, C, H, W), values in [0, 1].
    dates : list[str], optional
        Acquisition dates for titles.
    """
    apply_publication_style()
    from src.research.visualization.image_viewer import _to_rgb

    T = sequence.shape[0]
    dates = dates or [f"T={i}" for i in range(T)]

    fig, axes = plt.subplots(1, T, figsize=(T * 3, 3))
    if T == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = sequence[i]  # (C, H, W)
        rgb = _to_rgb(img, rgb_indices)
        ax.imshow(rgb)
        ax.set_title(dates[i] if i < len(dates) else f"T{i}", fontsize=8)
        ax.axis("off")

    fig.suptitle("Temporal Sequence", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if output_dir:
        save_figure(fig, "temporal_sequence", output_dir)
    return fig


__all__ = ["plot_temporal_sequence"]
