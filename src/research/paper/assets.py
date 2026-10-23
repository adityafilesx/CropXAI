"""
CropXAI Automated Paper Assets
================================

Automates the generation of publication-ready assets (LaTeX tables,
JSON summaries, training plots) at the end of an experiment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.filesystem.io import ensure_dir

logger = logging.getLogger(__name__)


class PaperAssetsExporter:
    """Exports artifacts required for the research paper."""

    def __init__(self, exp_dir: str | Path) -> None:
        self.exp_dir = Path(exp_dir)
        self.latex_dir = ensure_dir(self.exp_dir / "latex")
        self.csv_dir = ensure_dir(self.exp_dir / "csv")
        self.json_dir = ensure_dir(self.exp_dir / "json")

    def export_metrics_table(self, aggregated_metrics: dict[str, Any], filename: str = "benchmark_results") -> None:
        """
        Convert aggregated multi-run metrics to LaTeX and CSV.
        
        Expects `aggregated_metrics` formatted as:
        { "ModelA": {"miou": {"mean": 0.8, "std": 0.01}, ...}, ... }
        """
        rows = []
        for exp_key, metrics in aggregated_metrics.items():
            row = {"Experiment": exp_key}
            for m_key, m_stats in metrics.items():
                row[f"{m_key} (Mean)"] = m_stats["mean"]
                row[f"{m_key} (Std)"] = m_stats["std"]
                row[f"{m_key} (String)"] = f"{m_stats['mean']:.4f} ± {m_stats['std']:.4f}"
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # Save CSV
        df.to_csv(self.csv_dir / f"{filename}.csv", index=False)
        
        # Save LaTeX (only the string columns for readability)
        str_cols = ["Experiment"] + [c for c in df.columns if "String" in c]
        latex_df = df[str_cols]
        latex_df.columns = [c.replace(" (String)", "") for c in latex_df.columns]
        
        tex = latex_df.to_latex(index=False, booktabs=True)
        (self.latex_dir / f"{filename}.tex").write_text(tex, encoding="utf-8")
        
        logger.info("Exported metrics table to %s and %s", self.csv_dir, self.latex_dir)

    def export_json(self, data: dict[str, Any], filename: str) -> None:
        """Export raw nested dictionaries to JSON."""
        path = self.json_dir / f"{filename}.json"
        path.write_text(json.dumps(data, indent=4), encoding="utf-8")


__all__ = ["PaperAssetsExporter"]
