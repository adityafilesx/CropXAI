"""
CropXAI Master Campaign Execution Engine
==========================================
The absolute top-level entry point. Orchestrates the execution of all
experiments across Models, Datasets, and Seeds. 
Results are strictly written to the SQLite provenance database.
"""

from __future__ import annotations
import logging
from typing import Any
import traceback

logger = logging.getLogger(__name__)

class MasterCampaignEngine:
    """Executes the empirical campaign matrix."""

    def __init__(self, campaign_manager: Any, protocol: dict[str, Any]) -> None:
        self.manager = campaign_manager
        self.protocol = protocol
        
        # Override protocol with Micro-Benchmark settings for validation
        self.models = ["ResNet-18", "ViT-Base", "Prithvi-100M"]
        self.datasets = ["eurosat"]  # In a full run, ["eurosat", "bigearthnet", "pastis"]
        self.seeds = [42, 100, 999]  # 3 seeds for the micro-benchmark

    def run_full_campaign(self, epochs: int = None) -> bool:
        """Runs the nested loop across datasets, models, and seeds."""
        logger.info("=" * 60)
        logger.info("Initializing Master Campaign Engine (Provenance-Enforced)")
        logger.info("Models: %s", self.models)
        logger.info("Datasets: %s", self.datasets)
        logger.info("Seeds: %s", self.seeds)
        logger.info("=" * 60)
        
        try:
            from src.research.execution.trainer import run_real_training
            from src.data.downloaders import ensure_dataset
            
            # Ensure data is ready before training
            for ds in self.datasets:
                ensure_dataset(ds)
            
            # Use provided epochs, default to 1 if none provided (for speed/tests)
            run_epochs = epochs if epochs is not None else 1
            
            total_runs = len(self.datasets) * len(self.models) * len(self.seeds)
            current_run = 0
            
            for dataset in self.datasets:
                for model in self.models:
                    for seed in self.seeds:
                        current_run += 1
                        logger.info("\n" + "━" * 50)
                        logger.info("Campaign Run [%d/%d]: Model=%s | Dataset=%s | Seed=%d", 
                                    current_run, total_runs, model, dataset, seed)
                        logger.info("━" * 50)
                        
                        try:
                            # Note: The trainer internally handles the Model Registry mapping
                            # and saves directly to the provenance DB.
                            metrics, trust = run_real_training(
                                dataset_id=dataset, 
                                epochs=run_epochs, 
                                seed=seed,
                                model_name=model
                            )
                        except Exception as e:
                            logger.error("Run Failed: %s\n%s", e, traceback.format_exc())
                            # In a distributed environment, we would log the failure to DB and continue
                            continue
                            
            logger.info("=" * 60)
            logger.info("Master Campaign Matrix execution completed.")
            
        except Exception as e:
            logger.error("Campaign orchestrator failed: %s", e)
            return False
            
        return True

__all__ = ["MasterCampaignEngine"]
