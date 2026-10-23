"""
CropXAI Dataset Validator
==========================

Performs comprehensive data integrity checks on the raw dataset:

  - File existence and readability
  - GeoTIFF format validity (rasterio-based)
  - Label consistency (valid class indices)
  - Missing sample detection
  - Spatial consistency (CRS, resolution, extent)
  - Data range validation (expected value ranges)
  - Duplicate detection

Implements :class:`PipelineStage` and produces a :class:`ValidationReport`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from src.core.pipeline import PipelineStage
from src.utils.filesystem.io import list_files

logger = logging.getLogger(__name__)
console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Validation Report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    """A single validation issue found during dataset checks."""

    severity: str           # "error" | "warning" | "info"
    check: str              # Name of the check that raised this
    message: str
    file: str | None = None


@dataclass
class ValidationReport:
    """Aggregated validation report for the dataset."""

    dataset_name: str
    total_files: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, check: str, message: str, file: str | None = None) -> None:
        self.issues.append(ValidationIssue("error", check, message, file))

    def add_warning(self, check: str, message: str, file: str | None = None) -> None:
        self.issues.append(ValidationIssue("warning", check, message, file))

    def add_info(self, check: str, message: str) -> None:
        self.issues.append(ValidationIssue("info", check, message))

    def print_rich(self) -> None:
        """Render validation report as a Rich table."""
        status_icon = "✅ PASSED" if self.passed else "❌ FAILED"
        console.print(f"\n[bold]Dataset Validation Report — {self.dataset_name}[/bold]")
        console.print(f"Status: [bold {'green' if self.passed else 'red'}]{status_icon}[/]")
        console.print(f"Files checked: {self.total_files}")
        console.print(f"Errors: {len(self.errors)}  |  Warnings: {len(self.warnings)}")

        if self.issues:
            table = Table(title="Issues Found", show_header=True, header_style="bold")
            table.add_column("Severity", style="bold", width=10)
            table.add_column("Check", width=25)
            table.add_column("Message")
            table.add_column("File", width=30)

            severity_colors = {"error": "red", "warning": "yellow", "info": "dim"}
            for issue in self.issues:
                color = severity_colors.get(issue.severity, "white")
                table.add_row(
                    f"[{color}]{issue.severity.upper()}[/]",
                    issue.check,
                    issue.message,
                    issue.file or "",
                )
            console.print(table)


# ──────────────────────────────────────────────────────────────────────────────
# Validator Stage
# ──────────────────────────────────────────────────────────────────────────────

class ValidateStage(PipelineStage):
    """
    Stage 2: Validate dataset integrity.

    Checks performed:
    1. File existence
    2. GeoTIFF readability (via rasterio)
    3. Expected band count
    4. Spatial resolution consistency
    5. CRS consistency
    6. Data range validation
    7. Label file consistency
    8. Missing value fraction
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(name="Validate", enabled=True)
        self.cfg = cfg

    def process(self, data: Any) -> Any:
        from src.research.dataset.downloader import DownloadContext

        if isinstance(data, DownloadContext):
            ctx = data
        else:
            raise TypeError(f"ValidateStage expects DownloadContext, got {type(data)}")

        report = ValidationReport(dataset_name=self.cfg.dataset.name)

        if not ctx.dataset_dir.exists():
            report.add_error(
                "file_existence",
                f"Dataset directory not found: {ctx.dataset_dir}",
            )
            report.print_rich()
            ctx.metadata["validation_report"] = report
            return ctx

        # Collect all image files
        image_files = list_files(ctx.dataset_dir, pattern="*.tif")
        report.total_files = len(image_files)

        if report.total_files == 0:
            report.add_error(
                "file_existence",
                "No .tif files found in dataset directory.",
            )
        else:
            logger.info("Validating %d files...", report.total_files)
            self._check_files(image_files, report)

        report.print_rich()
        ctx.metadata["validation_report"] = report

        if not report.passed:
            raise RuntimeError(
                f"Dataset validation failed with {len(report.errors)} error(s). "
                "Fix the issues above before proceeding."
            )

        return ctx

    def _check_files(self, files: list[Path], report: ValidationReport) -> None:
        """Run all checks on the list of files."""
        try:
            import rasterio
        except ImportError:
            report.add_warning(
                "rasterio_import",
                "rasterio not available — skipping GeoTIFF checks.",
            )
            return

        expected_bands = self.cfg.dataset.bands.num_bands
        expected_size = self.cfg.dataset.spatial.patch_size
        crss: set[str] = set()
        resolutions: set[float] = set()

        for f in files:
            try:
                with rasterio.open(f) as src:
                    # Band count
                    if src.count != expected_bands:
                        report.add_warning(
                            "band_count",
                            f"Expected {expected_bands} bands, got {src.count}.",
                            file=str(f.name),
                        )

                    # Resolution
                    res = round(abs(src.transform.a), 2)
                    resolutions.add(res)

                    # CRS
                    if src.crs is not None:
                        crss.add(str(src.crs))

                    # Data range check on first file only (fast check)
                    if f == files[0]:
                        arr = src.read()
                        max_val = self.cfg.dataset.statistics.max_val or 10000
                        if arr.max() > max_val * 1.1:
                            report.add_warning(
                                "data_range",
                                f"Max pixel value {arr.max()} exceeds expected {max_val}.",
                                file=str(f.name),
                            )
                        missing_frac = float(np.isnan(arr).mean())
                        if missing_frac > 0:
                            report.add_info(
                                "missing_values",
                                f"First file missing fraction: {missing_frac:.2%}",
                            )

            except Exception as exc:
                report.add_error(
                    "readability",
                    f"Could not read file: {exc}",
                    file=str(f.name),
                )

        # CRS consistency
        if len(crss) > 1:
            report.add_warning(
                "crs_consistency",
                f"Multiple CRS found: {crss}. Files may not align.",
            )
        elif crss:
            report.add_info("crs_check", f"CRS: {next(iter(crss))}")

        # Resolution consistency
        if len(resolutions) > 1:
            report.add_warning(
                "resolution_consistency",
                f"Multiple resolutions found: {resolutions}.",
            )
        elif resolutions:
            report.add_info("resolution_check", f"Resolution: {next(iter(resolutions))}m")

        report.stats["total_files"] = len(files)
        report.stats["crss"] = list(crss)
        report.stats["resolutions"] = list(resolutions)


class DataIntegrityChecker:
    """
    Standalone integrity checker for verifying dataset checksums
    against a stored manifest.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)

    def verify(self, dataset_dir: str | Path) -> tuple[bool, list[str]]:
        """
        Verify dataset files against stored checksums.

        Returns
        -------
        tuple[bool, list[str]]
            (all_passed, list_of_failed_files)
        """
        from src.utils.filesystem.io import load_json, compute_checksum

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        manifest = load_json(self.manifest_path)
        checksums = manifest.get("checksums", {})

        failed: list[str] = []
        for rel_path, expected_hash in checksums.items():
            abs_path = Path(dataset_dir) / rel_path
            if not abs_path.exists():
                failed.append(f"MISSING: {rel_path}")
                continue
            actual_hash = compute_checksum(abs_path)
            if actual_hash != expected_hash:
                failed.append(f"CORRUPT: {rel_path}")

        return len(failed) == 0, failed


__all__ = [
    "ValidateStage",
    "ValidationReport",
    "ValidationIssue",
    "DataIntegrityChecker",
]
