"""CropXAI Augmentation Comparison Viewer."""
from __future__ import annotations
import logging
from typing import Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from src.research.visualization.export import apply_publication_style, save_figure
from src.research.visualization.image_viewer import _to_rgb
logger = logging.getLogger(__name__)


def plot_augmentation_comparison(
    original: np.ndarray,
    augmented_list: list[np.ndarray],
    aug_names: list[str] | None = None,
    rgb_indices: list[int] = [2, 1, 0],
    output_dir: str | None = None,
) -> Any:
    """
    Compare original image with multiple augmented versions.

    Parameters
    ----------
    original : np.ndarray
        Shape (C, H, W).
    augmented_list : list[np.ndarray]
        List of augmented images, each shape (C, H, W).
    aug_names : list[str], optional
        Name for each augmented version.
    """
    apply_publication_style()
    n_aug = len(augmented_list)
    n_total = n_aug + 1
    aug_names = aug_names or [f"Aug {i + 1}" for i in range(n_aug)]

    fig, axes = plt.subplots(1, n_total, figsize=(n_total * 3, 3.5))

    # Original
    axes[0].imshow(_to_rgb(original, rgb_indices))
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Augmented
    for i, (aug_img, name) in enumerate(zip(augmented_list, aug_names)):
        axes[i + 1].imshow(_to_rgb(aug_img, rgb_indices))
        axes[i + 1].set_title(name, fontsize=9)
        axes[i + 1].axis("off")

    fig.suptitle("Augmentation Comparison", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if output_dir:
        save_figure(fig, "augmentation_comparison", output_dir)
    return fig


__all__ = ["plot_augmentation_comparison"]
