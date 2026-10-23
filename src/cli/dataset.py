"""
CropXAI Dataset CLI
=====================

Exposes the DatasetManager to the `cropxai dataset ...` CLI commands.
Supports listing, downloading, verifying, and preprocessing datasets.
"""

from __future__ import annotations

import argparse
import logging
from src.data.manager import DatasetManager
from src.data.preprocessing import PreprocessingEngine

logger = logging.getLogger(__name__)


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attaches the dataset commands to the main CLI parser."""
    parser = subparsers.add_parser("dataset", help="Manage EO datasets (download, verify, preprocess)")
    dataset_subparsers = parser.add_subparsers(dest="dataset_cmd", required=True)
    
    # cropxai dataset list
    dataset_subparsers.add_parser("list", help="List available datasets in the registry")
    
    # cropxai dataset download <id>
    dl_parser = dataset_subparsers.add_parser("download", help="Download a dataset")
    dl_parser.add_argument("dataset_id", type=str, help="Dataset ID (e.g., eurosat)")
    
    # cropxai dataset verify <id>
    ver_parser = dataset_subparsers.add_parser("verify", help="Verify dataset SHA256 checksums")
    ver_parser.add_argument("dataset_id", type=str, help="Dataset ID (e.g., eurosat)")
    
    # cropxai dataset preprocess <id>
    pre_parser = dataset_subparsers.add_parser("preprocess", help="Preprocess raw dataset into standardized tensors")
    pre_parser.add_argument("dataset_id", type=str, help="Dataset ID (e.g., eurosat)")


def handle_command(args: argparse.Namespace) -> None:
    """Routes the dataset CLI command to the appropriate engine."""
    manager = DatasetManager()
    
    if args.dataset_cmd == "list":
        manager.list_datasets()
        
    elif args.dataset_cmd == "download":
        manager.download(args.dataset_id)
        
    elif args.dataset_cmd == "verify":
        manager.verify(args.dataset_id)
        
    elif args.dataset_cmd == "preprocess":
        engine = PreprocessingEngine()
        engine.preprocess(args.dataset_id)


__all__ = ["setup_parser", "handle_command"]
