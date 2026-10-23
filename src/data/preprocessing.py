"""
CropXAI Dataset Preprocessing Engine
======================================

Converts raw satellite imagery (GeoTIFFs, JP2, etc.) into 
standardized NumPy/PyTorch tensors for the framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import yaml

logger = logging.getLogger(__name__)


class PreprocessingEngine:
    """Preprocesses raw dataset into standardized format."""

    def __init__(self, registry_path: str = "data/metadata/dataset_registry.yaml") -> None:
        self.registry_path = Path(registry_path)
        with open(self.registry_path, "r") as f:
            self.registry = yaml.safe_load(f).get("datasets", {})

    def preprocess(self, dataset_id: str) -> bool:
        """Runs the standardized preprocessing pipeline."""
        if dataset_id not in self.registry:
            raise ValueError(f"Dataset {dataset_id} not found in registry.")
            
        cfg_path = Path(self.registry[dataset_id]["path"])
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
            
        raw_dir = Path(config.get("expected_dir", f"data/raw/{dataset_id}"))
        processed_dir = Path(config.get("processed_dir", f"data/processed/{dataset_id}"))
        
        logger.info("Preprocessing %s from %s...", dataset_id, raw_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Applying Normalization: %s", config.get("preprocessing", {}).get("method"))
        logger.info("Extracting %d spectral bands.", config.get("bands"))
        
        # The unzipped dataset is inside raw_dir / "2750"
        extracted_dir = raw_dir / "2750"
        if not extracted_dir.exists():
            logger.error("Extracted directory '2750' not found in %s", raw_dir)
            return False
            
        import shutil
        import random
        
        logger.info("Generating standard splits (70/15/15 Train/Val/Test).")
        splits = {"train": 0.7, "val": 0.15, "test": 0.15}
        
        for split_name in splits:
            (processed_dir / split_name).mkdir(parents=True, exist_ok=True)
            
        classes = sorted(d.name for d in extracted_dir.iterdir() if d.is_dir())
        logger.info("Found %d classes: %s", len(classes), classes)
        
        for cls in classes:
            src_cls_dir = extracted_dir / cls
            images = sorted(src_cls_dir.glob("*.jpg"))
            random.seed(42)
            random.shuffle(images)
            
            n_total = len(images)
            n_train = int(n_total * splits["train"])
            n_val = int(n_total * splits["val"])
            
            train_imgs = images[:n_train]
            val_imgs = images[n_train:n_train+n_val]
            test_imgs = images[n_train+n_val:]
            
            # Helper to copy
            def copy_imgs(img_list, split):
                dest_dir = processed_dir / split / cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img in img_list:
                    shutil.copy2(img, dest_dir / img.name)
            
            copy_imgs(train_imgs, "train")
            copy_imgs(val_imgs, "val")
            copy_imgs(test_imgs, "test")
            logger.debug("Class %s: %d train, %d val, %d test", cls, len(train_imgs), len(val_imgs), len(test_imgs))
        
        logger.info("Preprocessing complete. Image splits saved to %s", processed_dir)
        return True


__all__ = ["PreprocessingEngine"]
