"""
CropXAI Lightning DataModule
==============================

Wraps the PyTorch datasets created in Phase 1 into a LightningDataModule.
Manages Train, Validation, and Test data loaders.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.research.dataset.dataset import CropXAIDataset
from src.research.dataset.splitter import SplitStage
from src.research.dataset.downloader import DownloadContext

logger = logging.getLogger(__name__)


class CropXAIDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for CropXAI datasets.

    Parameters
    ----------
    cfg : Any
        Root configuration node.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_dir = cfg.paths.dataset
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        self.pin_memory = cfg.training.pin_memory
        
        self.train_dataset: CropXAIDataset | None = None
        self.val_dataset: CropXAIDataset | None = None
        self.test_dataset: CropXAIDataset | None = None

    def prepare_data(self) -> None:
        """Called only on 1 GPU to download/split data."""
        # Ensure splits are generated (Phase 1 SplitStage)
        ctx = DownloadContext(self.cfg)
        splitter = SplitStage(self.cfg)
        if not (ctx.dataset_dir / "splits").exists():
            splitter.process(ctx)

    def setup(self, stage: str | None = None) -> None:
        """Called on every GPU. Load datasets."""
        # Note: In a real implementation, augmentation pipelines from Phase 1
        # should be instantiated here and passed to the respective datasets.
        
        if stage == "fit" or stage is None:
            self.train_dataset = CropXAIDataset(self.cfg, split="train")
            self.val_dataset = CropXAIDataset(self.cfg, split="val")
            
        if stage == "test" or stage is None:
            self.test_dataset = CropXAIDataset(self.cfg, split="test")
            
        if stage == "predict":
            # For inference, might use full dataset or test set without labels
            self.predict_dataset = CropXAIDataset(self.cfg, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


__all__ = ["CropXAIDataModule"]
