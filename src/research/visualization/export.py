"""
CropXAI Publication-Quality Figure Export
==========================================

Exports Matplotlib figures in publication-ready formats with
IEEE column-width sizing, consistent fonts, and high DPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)

# IEEE column widths (inches)
IEEE_SINGLE_COLUMN = 3.5
IEEE_DOUBLE_COLUMN = 7.16
IEEE_PAGE_HEIGHT = 9.0


def apply_publication_style(cfg: Any | None = None) -> None:
    """Apply a publication-ready Matplotlib style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def save_figure(
    fig: Any,
    name: str,
    output_dir: str | Path,
    formats: list[str] | None = None,
    dpi: int = 300,
    paper_dir: str | Path | None = None,
) -> list[Path]:
    """
    Save a Matplotlib figure in one or more formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Base filename (without extension).
    output_dir : str | Path
        Primary output directory (e.g. artifacts/figures/).
    formats : list[str], optional
        Output formats. Default: ["pdf", "png"].
    dpi : int
        Resolution for raster formats. Default: 300.
    paper_dir : str | Path, optional
        If set, also copy the figure to the paper/figures/ directory.

    Returns
    -------
    list[Path]
        List of saved figure paths.
    """
    formats = formats or ["pdf", "png"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight", format=fmt)
        saved_paths.append(path)
        logger.info("Figure saved → %s", path)

        # Mirror to paper/figures/
        if paper_dir is not None:
            import shutil
            paper_path = Path(paper_dir) / f"{name}.{fmt}"
            paper_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(paper_path))

    return saved_paths


def figure_for_ieee(
    width: str = "single",
    height_ratio: float = 0.75,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Create a Matplotlib figure sized for IEEE journal submission.

    Parameters
    ----------
    width : str
        "single" (3.5") or "double" (7.16").
    height_ratio : float
        Height = width × height_ratio.

    Returns
    -------
    tuple[Figure, Axes]
    """
    apply_publication_style()
    w = IEEE_SINGLE_COLUMN if width == "single" else IEEE_DOUBLE_COLUMN
    h = w * height_ratio
    return plt.subplots(figsize=(w, h), **kwargs)


__all__ = [
    "apply_publication_style",
    "save_figure",
    "figure_for_ieee",
    "IEEE_SINGLE_COLUMN",
    "IEEE_DOUBLE_COLUMN",
]
