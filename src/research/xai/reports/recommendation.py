"""
CropXAI Best XAI Recommendation Engine
========================================

Analyzes XAI Benchmark results to recommend the best explanation methods
based on Faithfulness, Robustness, Localization, and Computational Efficiency.
"""

from __future__ import annotations

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class BestXAIRecommendationEngine:
    """Ranks and recommends the optimal XAI methods."""

    @staticmethod
    def generate_recommendations(benchmark_df: pd.DataFrame, output_dir: str | Path) -> None:
        """
        Parses the benchmark results and writes the best_xai_report.md.
        """
        if benchmark_df.empty:
            return
            
        # Helper to safely get the optimal row
        def get_best(df: pd.DataFrame, col: str, maximize: bool = True) -> str:
            if maximize:
                return df.loc[df[col].idxmax()]["Explainer"]
            else:
                return df.loc[df[col].idxmin()]["Explainer"]

        markdown_content = "# CropXAI Best XAI Recommendation Report\n\n"
        markdown_content += "This report automatically selects the optimal Explainability methods based on quantitative evaluations.\n\n"
        
        # 1. Overall Recommendations (Agnostic of Model)
        # Average across all models for a specific explainer
        explainer_means = benchmark_df.groupby("Explainer").mean(numeric_only=True).reset_index()
        
        if not explainer_means.empty and "deletion_auc_raw" in explainer_means.columns:
            markdown_content += "## 1. Global Optimal Methods (Cross-Architecture)\n"
            markdown_content += f"- **Most Faithful (Lowest Deletion AUC)**: `{get_best(explainer_means, 'deletion_auc_raw', False)}`\n"
            markdown_content += f"- **Most Robust (Highest Stability)**: `{get_best(explainer_means, 'stability_score_raw', True)}`\n"
            markdown_content += f"- **Best Human Agreement (Highest IoU)**: `{get_best(explainer_means, 'human_iou_raw', True)}`\n"
            markdown_content += f"- **Fastest Runtime**: `{get_best(explainer_means, 'runtime_ms_raw', False)}`\n"
            markdown_content += f"- **Simplest Explanation (Highest Sparsity)**: `{get_best(explainer_means, 'sparsity_raw', True)}`\n"
            
        # 2. Per-Architecture Recommendations
        markdown_content += "\n## 2. Optimal Methods per Architecture\n"
        for model_name in benchmark_df["Model"].unique():
            model_df = benchmark_df[benchmark_df["Model"] == model_name]
            
            # Simple heuristic for "Best Overall": 
            # Rank 1 in Deletion (desc), Rank 1 in Stability, etc.
            # Using rank average to find Pareto optimal
            model_df = model_df.copy()
            model_df["rank_faith"] = model_df["deletion_auc_raw"].rank(ascending=True)
            model_df["rank_robust"] = model_df["stability_score_raw"].rank(ascending=False)
            model_df["rank_human"] = model_df["human_iou_raw"].rank(ascending=False)
            
            model_df["overall_score"] = model_df["rank_faith"] + model_df["rank_robust"] + model_df["rank_human"]
            best_overall = model_df.loc[model_df["overall_score"].idxmin()]["Explainer"]
            
            markdown_content += f"### {model_name}\n"
            markdown_content += f"- **Best Overall XAI**: `{best_overall}`\n"
            markdown_content += f"- **Most Faithful**: `{get_best(model_df, 'deletion_auc_raw', False)}`\n"
            markdown_content += f"- **Most Robust**: `{get_best(model_df, 'stability_score_raw', True)}`\n\n"
            
        # 3. Full Benchmark Table
        markdown_content += "## 3. Full Benchmark Leaderboard\n"
        # Drop raw columns for clean table
        display_cols = [c for c in benchmark_df.columns if not c.endswith("_raw")]
        markdown_content += benchmark_df[display_cols].to_markdown(index=False)
        
        output_path = Path(output_dir) / "best_xai_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_content, encoding="utf-8")
        
        logger.info("Generated Best XAI Recommendation Report at %s", output_path)


__all__ = ["BestXAIRecommendationEngine"]
