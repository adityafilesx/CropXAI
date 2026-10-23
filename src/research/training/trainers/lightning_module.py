"""
CropXAI Lightning Trainer Wrapper
===================================

Generic wrapper that connects any BaseModel to PyTorch Lightning.
It strictly handles training and validation logic.
Evaluation is handled separately by the Evaluation Engine.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch

from src.research.models.base_model import BaseModel
from src.research.training.optimizers import build_optimizer
from src.research.training.schedulers import build_scheduler
from src.research.training.losses.registry import LOSS_REGISTRY

logger = logging.getLogger(__name__)


class CropXAITrainerWrapper(pl.LightningModule):
    """
    Lightning Module that encapsulates a model, loss, optimizer, and metrics.
    Only manages optimization; separated from evaluation.

    Parameters
    ----------
    cfg : Any
        Root configuration node.
    model : BaseModel
        The vision model to train.
    """

    def __init__(self, cfg: Any, model: BaseModel) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        
        # We don't save full cfg to hparams to avoid Hydra issues, 
        # but we can save specific hyperparameters.
        self.save_hyperparameters(ignore=["model"])
        
        # Loss function
        loss_cfg = cfg.loss
        self.criterion = LOSS_REGISTRY.build(loss_cfg.name, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Any:
        optimizer = build_optimizer(self.cfg, self.parameters())
        scheduler_config = build_scheduler(self.cfg, optimizer)
        
        if scheduler_config is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Usually delegated to the standalone Evaluator, but implemented
        here for framework completeness in case pl.Trainer.test is called.
        """
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, sync_dist=True)

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x = batch["image"]
        logits = self(x)
        return torch.argmax(logits, dim=1)


__all__ = ["CropXAITrainerWrapper"]
