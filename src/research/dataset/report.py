"""
CropXAI Dataset Summary Report
================================

Generates a comprehensive dataset summary report:
  - Rich terminal table
  - JSON structured data
  - HTML report (using Jinja2)

Used by `cropxai report` CLI command.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.utils.filesystem.io import ensure_dir, save_json

logger = logging.getLogger(__name__)
console = Console()


class DatasetReportGenerator:
    """
    Generates a structured dataset summary report.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Root CropXAI config.
    output_dir : str | Path
        Directory to write report files.
    """

    def __init__(self, cfg: Any, output_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir or cfg.paths.artifacts) / "reports"
        ensure_dir(self.output_dir)

    def generate(
        self,
        dataset_stats: Any | None = None,
        dataset_metadata: Any | None = None,
        split_info: dict[str, Any] | None = None,
        validation_report: Any | None = None,
    ) -> dict[str, Any]:
        """
        Generate the full report.

        Returns
        -------
        dict
            Structured report data.
        """
        report = self._build_report_dict(
            dataset_stats, dataset_metadata, split_info, validation_report
        )

        # Print Rich terminal report
        self._print_rich_report(report)

        # Save JSON report
        json_path = self.output_dir / "dataset_report.json"
        save_json(json_path, report)
        logger.info("Dataset report saved → %s", json_path)

        # Save HTML report
        html_path = self._generate_html(report)
        logger.info("HTML report saved → %s", html_path)

        return report

    def _build_report_dict(self, *args: Any) -> dict[str, Any]:
        dataset_stats, dataset_metadata, split_info, validation_report = args

        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "name": self.cfg.dataset.name,
                "display_name": self.cfg.dataset.display_name,
                "version": self.cfg.dataset.version,
                "num_classes": self.cfg.dataset.labels.num_classes,
                "num_bands": self.cfg.dataset.bands.num_bands,
                "patch_size": self.cfg.dataset.spatial.patch_size,
                "temporal_enabled": self.cfg.dataset.temporal.enabled,
            },
        }

        if dataset_metadata:
            report["metadata"] = {
                "total_files": getattr(dataset_metadata, "total_files", "N/A"),
                "total_size_gb": getattr(dataset_metadata, "total_size_gb", "N/A"),
                "unique_crs": getattr(dataset_metadata, "unique_crs", []),
                "unique_resolutions": getattr(dataset_metadata, "unique_resolutions", []),
                "temporal_range": list(getattr(dataset_metadata, "temporal_range", [None, None])),
            }

        if dataset_stats:
            report["statistics"] = {
                "total_samples": getattr(dataset_stats, "total_samples", "N/A"),
                "imbalance_ratio": round(getattr(dataset_stats, "imbalance_ratio", 1.0), 2),
                "class_distribution": [
                    {
                        "id": c.class_id,
                        "name": c.class_name,
                        "pixel_fraction": round(c.pixel_fraction, 4),
                        "sample_fraction": round(c.sample_fraction, 4),
                    }
                    for c in getattr(dataset_stats, "class_stats", [])
                ],
                "band_statistics": [
                    {
                        "band": b.band_name,
                        "mean": round(b.mean, 2),
                        "std": round(b.std, 2),
                        "p1": round(b.p1, 2),
                        "p99": round(b.p99, 2),
                    }
                    for b in getattr(dataset_stats, "band_stats", [])
                ],
            }

        if split_info:
            report["splits"] = split_info

        if validation_report:
            report["validation"] = {
                "passed": getattr(validation_report, "passed", True),
                "errors": len(getattr(validation_report, "errors", [])),
                "warnings": len(getattr(validation_report, "warnings", [])),
            }

        return report

    def _print_rich_report(self, report: dict[str, Any]) -> None:
        """Print a beautiful Rich summary to the terminal."""
        console.print()
        console.rule("[bold cyan]CropXAI Dataset Report[/]")

        # Header panel
        ds = report["dataset"]
        console.print(Panel(
            f"[bold]{ds['display_name']}[/]\n"
            f"Version: {ds['version']}  |  "
            f"Classes: {ds['num_classes']}  |  "
            f"Bands: {ds['num_bands']}  |  "
            f"Patch: {ds['patch_size']}px  |  "
            f"Temporal: {ds['temporal_enabled']}",
            title="[bold green]Dataset Overview[/]",
            border_style="green",
        ))

        # Metadata table
        if "metadata" in report:
            meta = report["metadata"]
            t = Table(title="File Metadata", box=box.ROUNDED, header_style="bold cyan")
            t.add_column("Property", style="bold")
            t.add_column("Value")
            t.add_row("Total Files", str(meta.get("total_files", "N/A")))
            t.add_row("Total Size", f"{meta.get('total_size_gb', 'N/A')} GB")
            t.add_row("CRS", ", ".join(str(c) for c in meta.get("unique_crs", [])) or "Unknown")
            t.add_row("Resolution", ", ".join(f"{r}m" for r in meta.get("unique_resolutions", [])) or "Unknown")
            t.add_row("Temporal Range", " → ".join(str(d) for d in meta.get("temporal_range", [])) or "N/A")
            console.print(t)

        # Class distribution table
        if "statistics" in report and report["statistics"].get("class_distribution"):
            t2 = Table(title="Class Distribution", box=box.ROUNDED, header_style="bold magenta")
            t2.add_column("ID", justify="right")
            t2.add_column("Class Name")
            t2.add_column("Pixel %", justify="right")
            t2.add_column("Sample %", justify="right")
            for cls in report["statistics"]["class_distribution"]:
                t2.add_row(
                    str(cls["id"]),
                    cls["name"],
                    f"{cls['pixel_fraction'] * 100:.2f}%",
                    f"{cls['sample_fraction'] * 100:.2f}%",
                )
            console.print(t2)

        # Band stats table
        if "statistics" in report and report["statistics"].get("band_statistics"):
            t3 = Table(title="Band Statistics", box=box.ROUNDED, header_style="bold blue")
            t3.add_column("Band")
            t3.add_column("Mean", justify="right")
            t3.add_column("Std", justify="right")
            t3.add_column("P1", justify="right")
            t3.add_column("P99", justify="right")
            for b in report["statistics"]["band_statistics"]:
                t3.add_row(b["band"], str(b["mean"]), str(b["std"]), str(b["p1"]), str(b["p99"]))
            console.print(t3)

        # Validation
        if "validation" in report:
            v = report["validation"]
            status = "[bold green]✅ PASSED[/]" if v["passed"] else "[bold red]❌ FAILED[/]"
            console.print(f"\nValidation: {status}  |  Errors: {v['errors']}  |  Warnings: {v['warnings']}")

        console.rule()

    def _generate_html(self, report: dict[str, Any]) -> Path:
        """Generate a minimal HTML report."""
        html_path = self.output_dir / "dataset_report.html"
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CropXAI Dataset Report — {report["dataset"]["display_name"]}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: #0f172a; color: #e2e8f0; }}
  h1, h2 {{ color: #38bdf8; }} table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th {{ background: #1e3a5f; padding: 0.5rem 1rem; text-align: left; }}
  td {{ padding: 0.5rem 1rem; border-bottom: 1px solid #334155; }}
  .pass {{ color: #4ade80; }} .fail {{ color: #f87171; }}
  pre {{ background: #1e293b; padding: 1rem; border-radius: 8px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>CropXAI Dataset Report</h1>
<p>Generated: {report["generated_at"]}</p>
<h2>{report["dataset"]["display_name"]}</h2>
<pre>{json.dumps(report, indent=2)}</pre>
</body>
</html>"""
        html_path.write_text(html, encoding="utf-8")
        return html_path


__all__ = ["DatasetReportGenerator"]
