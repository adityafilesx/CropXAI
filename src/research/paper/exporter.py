"""
CropXAI Paper Exporter
========================

Generates LaTeX tables, collates figures, and builds supplementary
materials for journal submissions (e.g. IEEE TGRS).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.utils.filesystem.io import ensure_dir

logger = logging.getLogger(__name__)


class PaperExporter:
    """
    Automates generating assets for a research paper.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root config.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.paper_dir = Path(cfg.paths.paper)
        self.figures_dir = ensure_dir(self.paper_dir / "figures")
        self.tables_dir = ensure_dir(self.paper_dir / "tables")
        self.supplementary_dir = ensure_dir(self.paper_dir / "supplementary")

    def export_latex_table(self, df: Any, filename: str, caption: str, label: str) -> Path:
        """
        Export a Pandas DataFrame to a formatted LaTeX table.

        Parameters
        ----------
        df : pd.DataFrame
        filename : str
        caption : str
        label : str

        Returns
        -------
        Path
        """
        path = self.tables_dir / f"{filename}.tex"
        tex = df.to_latex(
            index=False,
            float_format="%.4f",
            caption=caption,
            label=f"tab:{label}",
            booktabs=True,
        )
        path.write_text(tex, encoding="utf-8")
        logger.info("LaTeX table exported → %s", path)
        return path


__all__ = ["PaperExporter"]
