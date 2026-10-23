"""
CropXAI Automated Paper Assets & Reports
==========================================

Generates Leaderboards and CNN vs Transformer Comparison Reports.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from src.utils.filesystem.io import ensure_dir

logger = logging.getLogger(__name__)


class BenchmarkReports:
    """Generates leaderboards and cross-architecture comparison reports."""

    def __init__(self, exp_dir: str | Path) -> None:
        self.exp_dir = Path(exp_dir)
        self.reports_dir = ensure_dir(self.exp_dir / "reports")
        self.leaderboard_path = self.reports_dir / "leaderboard.csv"

    def append_to_leaderboard(self, exp_key: str, metrics: dict[str, Any], profile: dict[str, Any]) -> None:
        """Appends a new run to the global leaderboard."""
        row = {
            "Model": exp_key,
            "mIoU": metrics.get("miou", {}).get("mean", 0.0),
            "OA": metrics.get("oa", {}).get("mean", 0.0),
            "Parameters (M)": profile.get("parameters", 0) / 1e6,
            "FLOPs (G)": profile.get("flops", 0) / 1e9,
            "Latency (ms)": profile.get("latency_ms", 0.0),
            "Peak VRAM (MB)": profile.get("vram_mb", 0.0),
            "Throughput (IPS)": profile.get("throughput_ips", 0.0)
        }
        
        df_new = pd.DataFrame([row])
        
        if self.leaderboard_path.exists():
            df_existing = pd.read_csv(self.leaderboard_path)
            # Remove old entry if exists, then append
            df_existing = df_existing[df_existing["Model"] != exp_key]
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
            
        # Sort by mIoU descending
        df_combined = df_combined.sort_values(by="mIoU", ascending=False)
        df_combined.to_csv(self.leaderboard_path, index=False)
        logger.info("Leaderboard updated at %s", self.leaderboard_path)

    def generate_comparison_report(self) -> None:
        """Generates markdown/latex reports analyzing tradeoffs."""
        if not self.leaderboard_path.exists():
            return
            
        df = pd.read_csv(self.leaderboard_path)
        
        from src.research.evaluation.recommendation import RecommendationEngine
        recommendations = RecommendationEngine.generate_recommendations(df)
        
        markdown_content = "# CropXAI Baseline Validation Report\n\n"
        markdown_content += "## 1. Current Leaderboard\n"
        markdown_content += df.to_markdown(index=False)
        markdown_content += "\n\n## 2. Baseline Recommendations\n"
        
        for category, model in recommendations.items():
            markdown_content += f"- **{category}**: `{model}`\n"
            
        markdown_content += "\n## 3. Representation & Calibration Overview\n"
        markdown_content += "Models were evaluated across predictive performance (mIoU, OA), computational efficiency (FLOPs, Latency), and calibration (ECE).\n"
        
        report_path = self.reports_dir / "baseline_validation_report.md"
        report_path.write_text(markdown_content, encoding="utf-8")
        logger.info("Generated %s", report_path)


__all__ = ["BenchmarkReports"]
