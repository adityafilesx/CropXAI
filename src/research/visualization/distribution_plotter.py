"""
CropXAI Class Distribution Plotter
=====================================

Visualizes class distribution as bar charts and pie charts.
Exports publication-quality figures to paper/figures/.
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


def plot_class_distribution(
    class_names: list[str],
    counts: list[int | float],
    title: str = "Class Distribution",
    normalize: bool = True,
    output_dir: str | Path | None = None,
    paper_dir: str | Path | None = None,
    cfg: Any | None = None,
) -> Any:
    """
    Plot a horizontal bar chart of class pixel distribution.

    Parameters
    ----------
    class_names : list[str]
        Class labels.
    counts : list[int | float]
        Pixel or sample counts per class.
    normalize : bool
        If True, show percentages instead of raw counts.
    """
    apply_publication_style()

    if normalize:
        total = max(sum(counts), 1)
        values = [c / total * 100 for c in counts]
        xlabel = "Percentage (%)"
    else:
        values = list(counts)
        xlabel = "Pixel Count"

    n_classes = len(class_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(7, max(3, n_classes * 0.5)))

    bars = ax.barh(class_names, values, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, values):
        label = f"{val:.1f}%" if normalize else f"{int(val):,}"
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    if output_dir:
        save_figure(fig, "class_distribution", output_dir, paper_dir=paper_dir)

    return fig


def plot_imbalance_ratio(
    class_names: list[str],
    counts: list[int | float],
    output_dir: str | Path | None = None,
) -> Any:
    """
    Plot a normalized imbalance visualization showing dominant vs minority classes.
    """
    apply_publication_style()
    total = max(sum(counts), 1)
    fracs = [c / total for c in counts]
    sorted_pairs = sorted(zip(fracs, class_names), reverse=True)
    sorted_fracs, sorted_names = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(sorted_names)), sorted_fracs, color="#3b82f6", alpha=0.85)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of Pixels")
    ax.set_title("Class Imbalance (Sorted)", fontweight="bold")
    plt.tight_layout()

    if output_dir:
        save_figure(fig, "class_imbalance", output_dir)

    return fig


__all__ = ["plot_class_distribution", "plot_imbalance_ratio"]
