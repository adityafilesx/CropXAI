"""
CropXAI Sample Grid Visualizer
================================

Renders a grid of dataset samples (image + label overlay) for inspection.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.research.visualization.export import apply_publication_style, save_figure
from src.research.visualization.image_viewer import _to_rgb

logger = logging.getLogger(__name__)


def plot_sample_grid(
    samples: list[dict[str, Any]],
    n_cols: int = 4,
    rgb_indices: list[int] = [2, 1, 0],
    show_labels: bool = True,
    title: str = "Dataset Samples",
    output_dir: str | None = None,
    paper_dir: str | None = None,
) -> Any:
    """
    Display a grid of dataset samples with optional label overlays.

    Parameters
    ----------
    samples : list[dict]
        List of dicts from CropXAIDataset.get_sample_for_visualization().
    n_cols : int
        Number of columns in the grid.
    rgb_indices : list[int]
        Band indices for R, G, B display.
    show_labels : bool
        If True, overlay the segmentation label mask.
    """
    apply_publication_style()

    n = len(samples)
    n_rows = (n + n_cols - 1) // n_cols
    n_cols_actual = min(n, n_cols)

    fig, axes = plt.subplots(n_rows, n_cols_actual, figsize=(n_cols_actual * 3, n_rows * 3))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.set_visible(False)
            continue

        sample = samples[i]
        image = sample.get("image")  # (C, H, W)
        label = sample.get("label")  # (H, W) or None
        idx = sample.get("idx", i)

        if image is None:
            ax.set_visible(False)
            continue

        rgb = _to_rgb(image, rgb_indices)
        ax.imshow(rgb)

        if show_labels and label is not None:
            # Overlay label as a semi-transparent mask
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(plt.cm.tab10.colors[:8])  # type: ignore[attr-defined]
            ax.imshow(label, alpha=0.35, cmap=cmap, vmin=0, vmax=7)

        ax.set_title(f"#{idx}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        save_figure(fig, "sample_grid", output_dir, paper_dir=paper_dir)

    return fig


__all__ = ["plot_sample_grid"]
