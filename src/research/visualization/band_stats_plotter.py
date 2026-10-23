"""
CropXAI Band Statistics Plotter
==================================

Visualizes per-band histograms and summary statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from src.research.visualization.export import apply_publication_style, save_figure

logger = logging.getLogger(__name__)


def plot_band_histograms(
    image: np.ndarray,
    band_names: list[str] | None = None,
    n_bins: int = 100,
    output_dir: str | Path | None = None,
) -> Any:
    """
    Plot per-band pixel value histograms.

    Parameters
    ----------
    image : np.ndarray
        Shape (C, H, W), values in [0, 1] or raw DN.
    band_names : list[str], optional
        Band labels.
    n_bins : int
        Number of histogram bins.
    """
    apply_publication_style()
    C = image.shape[0]
    band_names = band_names or [f"Band {i + 1}" for i in range(C)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, C))  # type: ignore[attr-defined]

    n_cols = min(C, 3)
    n_rows = (C + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes_flat):
        if i < C:
            vals = image[i].ravel()
            ax.hist(vals, bins=n_bins, color=colors[i], alpha=0.85, edgecolor="none")
            ax.set_title(band_names[i], fontsize=10)
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Count")
            ax.axvline(np.mean(vals), color="red", linestyle="--", linewidth=1, label=f"μ={np.mean(vals):.2f}")
            ax.legend(fontsize=7)
        else:
            ax.set_visible(False)

    fig.suptitle("Spectral Band Histograms", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        save_figure(fig, "band_histograms", output_dir)

    return fig


def plot_band_statistics_table(
    band_stats: list[Any],
    output_dir: str | Path | None = None,
) -> Any:
    """
    Render per-band statistics as a heatmap table.

    Parameters
    ----------
    band_stats : list[BandStatistics]
        From DatasetAnalyzer output.
    """
    apply_publication_style()
    if not band_stats:
        logger.warning("No band stats to plot.")
        return None

    bands = [b.band_name for b in band_stats]
    data = np.array([
        [b.mean, b.std, b.min_val, b.max_val, b.p1, b.p99]
        for b in band_stats
    ])
    col_labels = ["Mean", "Std", "Min", "Max", "P1", "P99"]

    fig, ax = plt.subplots(figsize=(8, max(3, len(bands) * 0.5)))
    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, rotation=45, ha="right")
    ax.set_yticks(range(len(col_labels)))
    ax.set_yticklabels(col_labels)
    plt.colorbar(im, ax=ax)
    ax.set_title("Band Statistics Heatmap", fontweight="bold")
    plt.tight_layout()

    if output_dir:
        save_figure(fig, "band_stats_heatmap", output_dir)

    return fig


__all__ = ["plot_band_histograms", "plot_band_statistics_table"]
