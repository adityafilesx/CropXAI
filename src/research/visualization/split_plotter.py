"""CropXAI Split Distribution Plotter."""
from __future__ import annotations
from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("Agg")
from src.research.visualization.export import apply_publication_style, save_figure


def plot_split_distribution(
    split_info: dict[str, Any],
    output_dir: str | None = None,
) -> Any:
    """Plot train/val/test split sizes as a stacked bar."""
    apply_publication_style()
    sizes = {k: v.get("size", 0) for k, v in split_info.items() if isinstance(v, dict)}
    total = max(sum(sizes.values()), 1)

    fig, ax = plt.subplots(figsize=(6, 2.5))
    colors = {"train": "#3b82f6", "val": "#f59e0b", "test": "#10b981"}
    left = 0
    for split, count in sizes.items():
        frac = count / total
        ax.barh(0, frac, left=left, color=colors.get(split, "gray"), label=f"{split} ({count})", height=0.5)
        ax.text(left + frac / 2, 0, f"{frac * 100:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        left += frac

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Fraction of Dataset")
    ax.set_title("Train / Val / Test Split", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    if output_dir:
        save_figure(fig, "split_distribution", output_dir)
    return fig


__all__ = ["plot_split_distribution"]
