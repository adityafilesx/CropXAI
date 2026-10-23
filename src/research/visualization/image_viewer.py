"""
CropXAI Image Viewer
=====================

Visualizes multi-band satellite imagery as:
  - RGB true-color composites
  - False-color (vegetation) composites
  - Individual band grayscale views
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from src.research.visualization.export import apply_publication_style, save_figure

logger = logging.getLogger(__name__)


def _to_rgb(image: np.ndarray, rgb_indices: list[int] = [2, 1, 0]) -> np.ndarray:
    """
    Convert (C, H, W) float array to (H, W, 3) RGB uint8 for display.

    Parameters
    ----------
    image : np.ndarray
        Shape (C, H, W), values in [0, 1].
    rgb_indices : list[int]
        Band indices for R, G, B channels.
    """
    C, H, W = image.shape
    r_idx = min(rgb_indices[0], C - 1)
    g_idx = min(rgb_indices[1], C - 1)
    b_idx = min(rgb_indices[2], C - 1)

    rgb = np.stack([image[r_idx], image[g_idx], image[b_idx]], axis=-1)
    # Percentile stretch for better visualization
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-8), 0, 1)
    return (rgb * 255).astype(np.uint8)


def plot_rgb(
    image: np.ndarray,
    title: str = "RGB Composite",
    rgb_indices: list[int] = [2, 1, 0],
    ax: Any | None = None,
) -> Any:
    """
    Plot an RGB composite image.

    Parameters
    ----------
    image : np.ndarray
        Shape (C, H, W), values in [0, 1].
    title : str
        Plot title.
    rgb_indices : list[int]
        Band indices for [R, G, B].
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, create a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    apply_publication_style()
    rgb = _to_rgb(image, rgb_indices)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    return ax


def plot_false_color(
    image: np.ndarray,
    title: str = "False Color (NIR-R-G)",
    nir_red_green: list[int] = [3, 2, 1],
    ax: Any | None = None,
) -> Any:
    """Plot a false-color composite (NIR-Red-Green) for vegetation visualization."""
    return plot_rgb(image, title=title, rgb_indices=nir_red_green, ax=ax)


def plot_band_grid(
    image: np.ndarray,
    band_names: list[str] | None = None,
    n_cols: int = 3,
    output_path: str | None = None,
) -> Any:
    """
    Plot all bands in a grid layout.

    Parameters
    ----------
    image : np.ndarray
        Shape (C, H, W).
    band_names : list[str], optional
        Names for each band.
    n_cols : int
        Number of columns in grid.
    """
    apply_publication_style()
    C, H, W = image.shape
    n_rows = (C + n_cols - 1) // n_cols
    band_names = band_names or [f"Band {i + 1}" for i in range(C)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes_flat):
        if i < C:
            im = ax.imshow(image[i], cmap="viridis", interpolation="nearest")
            ax.set_title(band_names[i], fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.8)
        ax.axis("off")

    fig.suptitle("Spectral Band Visualization", fontsize=12, fontweight="bold")
    return fig


__all__ = ["plot_rgb", "plot_false_color", "plot_band_grid", "_to_rgb"]
