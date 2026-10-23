"""
CropXAI Campaign CLI
======================

Exposes the MasterCampaignEngine to the `cropxai campaign ...` CLI commands.
Supports running full, pilot, and reproducibility campaigns.
"""

from __future__ import annotations

import argparse
import logging
import yaml
from pathlib import Path
from src.research.execution.master_campaign import MasterCampaignEngine
from src.research.validation.benchmark_validator import BenchmarkComplianceChecker
from src.research.validation.artifact_integrity import ArtifactIntegrityValidator

logger = logging.getLogger(__name__)


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attaches the campaign commands to the main CLI parser."""
    parser = subparsers.add_parser("campaign", help="Manage scientific campaigns (run, validate)")
    campaign_subparsers = parser.add_subparsers(dest="campaign_cmd", required=True)
    
    # cropxai campaign run
    run_parser = campaign_subparsers.add_parser("run", help="Run the master campaign")
    run_parser.add_argument("--protocol", type=str, default="src/research/protocols/protocol_v1.yaml", help="Path to protocol yaml")
    run_parser.add_argument("--pilot", action="store_true", help="Run a miniature pilot campaign (e.g. 1 seed, 2 models)")
    run_parser.add_argument("--model", type=str, help="Specific model to run (for SLURM sharding)")
    run_parser.add_argument("--dataset", type=str, help="Specific dataset to run (for SLURM sharding)")
    run_parser.add_argument("--seed", type=int, help="Specific seed to run (for SLURM sharding)")
    run_parser.add_argument("--epochs", type=int, help="Number of epochs to train (for SLURM sharding)")

def handle_command(args: argparse.Namespace) -> None:
    """Routes the campaign CLI command to the appropriate engine."""
    
    if args.campaign_cmd == "run":
        protocol_path = Path(args.protocol)
        if protocol_path.exists():
            with open(protocol_path, "r") as f:
                protocol = yaml.safe_load(f)
        else:
            protocol = {"version": "mock"}
            
        if args.pilot:
            logger.info("INITIATING PILOT CAMPAIGN (1 Seed, 1 CNN, 1 ViT)")
            protocol["statistics"] = {"seeds": 1}
        
        engine = MasterCampaignEngine(campaign_manager=None, protocol=protocol)
        
        # Override with specific SLURM arguments if provided
        if args.model:
            engine.models = [args.model]
        if args.dataset:
            engine.datasets = [args.dataset]
        if args.seed is not None:
            engine.seeds = [args.seed]
            
        engine.run_full_campaign(epochs=args.epochs)
        
        # Trigger validation
        compliance = BenchmarkComplianceChecker(database=None, protocol=protocol)
        compliance.verify_compliance()
        
        logger.info("Pilot Phase completed. Assets logged to database.")


__all__ = ["setup_parser", "handle_command"]
