"""
CropXAI Command-Line Interface (CLI)
======================================

Main entry point for the CropXAI framework.
Uses Hydra for configuration management and sub-commands for actions.

Commands:
  cropxai download    - Download dataset
  cropxai validate    - Validate dataset integrity
  cropxai split       - Generate train/val/test splits
  cropxai analyze     - Compute dataset statistics
  cropxai report      - Generate dataset summary report
  cropxai explore     - Launch interactive dashboard
  cropxai train       - Run model training (Phase 2)
  cropxai evaluate    - Evaluate a trained model (Phase 2)
  cropxai explain     - Run XAI methods (Phase 4)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from src.utils.logging.logger import setup_logging

logger = logging.getLogger(__name__)
console = Console()


def _print_banner() -> None:
    console.print(
        "\n[bold cyan]"
        "  ____                 _  __    _    ___ \n"
        " / ___| _ __ ___  _ __| |/ /   / \\  |_ _|\n"
        "| |    | '__/ _ \\| '_ \\ ' /   / _ \\  | | \n"
        "| |___ | | | (_) | |_) | . \\ / ___ \\ | | \n"
        " \\____||_|  \\___/| .__/|_|\\_/_/   \\_\\___|\n"
        "                 |_|                     \n"
        "[/bold cyan]"
        "Research Framework for Earth Observation\n"
    )


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main CLI entry point managed by Hydra."""
    # Ensure config resolution
    OmegaConf.resolve(cfg)

    # Setup core logging
    setup_logging(cfg)

    _print_banner()

    cmd = cfg.get("command", "help")
    logger.info("Executing command: [bold green]%s[/]", cmd, extra={"markup": True})

    # Dispatch command
    try:
        if cmd == "help":
            _run_help()
        elif cmd == "download":
            _run_download(cfg)
        elif cmd == "validate":
            _run_validate(cfg)
        elif cmd == "split":
            _run_split(cfg)
        elif cmd == "analyze":
            _run_analyze(cfg)
        elif cmd == "report":
            _run_report(cfg)
        elif cmd == "explore":
            _run_explore(cfg)
        elif cmd in ("train", "evaluate", "explain"):
            logger.warning(
                "Command '%s' is scheduled for future implementation (Phases 2-4).", cmd
            )
        else:
            logger.error("Unknown command: %s", cmd)
            sys.exit(1)
    except Exception as exc:
        logger.exception("Command '%s' failed: %s", cmd, exc)
        sys.exit(1)


def _run_help() -> None:
    console.print(
        "Usage: [bold green]cropxai command=COMMAND [options][/]\n\n"
        "Available Commands:\n"
        "  [cyan]download[/]  - Download dataset\n"
        "  [cyan]validate[/]  - Validate dataset integrity\n"
        "  [cyan]split[/]     - Generate train/val/test splits\n"
        "  [cyan]analyze[/]   - Compute dataset statistics\n"
        "  [cyan]report[/]    - Generate dataset summary report\n"
        "  [cyan]explore[/]   - Launch interactive dashboard\n"
        "\nExamples:\n"
        "  cropxai command=validate\n"
        "  cropxai command=download dataset.download.source=huggingface\n"
    )


def _run_download(cfg: Any) -> None:
    from src.research.dataset.downloader import DownloadStage
    stage = DownloadStage(cfg)
    stage.process(None)


def _run_validate(cfg: Any) -> None:
    from src.research.dataset.downloader import DownloadContext
    from src.research.dataset.validator import ValidateStage
    ctx = DownloadContext(cfg)
    stage = ValidateStage(cfg)
    stage.process(ctx)


def _run_split(cfg: Any) -> None:
    from src.research.dataset.downloader import DownloadContext
    from src.research.dataset.splitter import SplitStage
    ctx = DownloadContext(cfg)
    stage = SplitStage(cfg)
    stage.process(ctx)


def _run_analyze(cfg: Any) -> None:
    from src.research.dataset.downloader import DownloadContext
    from src.research.dataset.analyzer import DatasetAnalyzer
    ctx = DownloadContext(cfg)
    stage = DatasetAnalyzer(cfg)
    stage.process(ctx)


def _run_report(cfg: Any) -> None:
    from src.research.dataset.report import DatasetReportGenerator
    from src.research.dataset.analyzer import DatasetAnalyzer
    from src.research.dataset.metadata_reader import MetadataReader
    from src.research.dataset.downloader import DownloadContext

    ctx = DownloadContext(cfg)
    # Fast analysis for report
    stats = DatasetAnalyzer(cfg, max_samples=10)._compute_statistics(
        list(ctx.dataset_dir.glob("*.tif")),
        list(ctx.dataset_dir.glob("*label*.tif"))
    )
    metadata = MetadataReader(cfg, max_files=10).read_dataset(ctx.dataset_dir)

    generator = DatasetReportGenerator(cfg)
    generator.generate(dataset_stats=stats, dataset_metadata=metadata)


def _run_explore(cfg: Any) -> None:
    from src.research.visualization.dashboard import run_dashboard
    run_dashboard(cfg)


if __name__ == "__main__":
    main()
