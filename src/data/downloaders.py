"""
CropXAI Data Downloaders
========================
Automated dataset acquisition and verification for the empirical campaign.
Supports:
- EuroSAT (via TorchGeo)
- BigEarthNet (via TorchGeo)
- CropHarvest / PASTIS (Stubs for future scale-up)
"""

import logging
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")

def ensure_dataset(dataset_id: str):
    """
    Ensures that the requested dataset is downloaded and verified.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensuring dataset '{dataset_id}' is available at {DATA_DIR}...")
    
    if dataset_id == "eurosat":
        _download_eurosat()
    elif dataset_id == "bigearthnet":
        _download_bigearthnet()
    else:
        logger.warning(f"Downloader for {dataset_id} not implemented yet.")
        
def _download_eurosat():
    try:
        from torchgeo.datasets import EuroSAT
        logger.info("Downloading/Verifying EuroSAT (RGB)...")
        # Initialize the dataset; TorchGeo handles the download and extraction
        dataset = EuroSAT(root=str(DATA_DIR / "eurosat"), download=True, bands=["B04", "B03", "B02"])
        logger.info(f"EuroSAT loaded successfully: {len(dataset)} samples.")
    except ImportError:
        logger.error("torchgeo not installed. Cannot download EuroSAT.")
    except Exception as e:
        logger.error(f"Error downloading EuroSAT: {e}")

def _download_bigearthnet():
    try:
        from torchgeo.datasets import BigEarthNet
        logger.info("Downloading/Verifying BigEarthNet (Sentinel-2)...")
        # BigEarthNet requires kaggle credentials or manual download for the raw files,
        # but torchgeo can structure it if paths are provided. 
        # For this micro-benchmark, we will log instructions if it fails.
        dataset = BigEarthNet(root=str(DATA_DIR / "bigearthnet"), download=True, bands="s2")
        logger.info(f"BigEarthNet loaded successfully: {len(dataset)} samples.")
    except ImportError:
        logger.error("torchgeo not installed.")
    except Exception as e:
        logger.error(f"BigEarthNet automatic download requires manual intervention: {e}")
        logger.info("Please download BigEarthNet-S2 manually to data/raw/bigearthnet")
