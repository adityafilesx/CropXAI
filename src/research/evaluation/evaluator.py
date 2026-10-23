"""
CropXAI Evaluation Engine
===========================

Standalone evaluation engine decoupled from the training loop.
Takes a trained (or frozen) model and a dataloader, computes all 
metrics using TorchMetrics, and manages reporting/visualization triggers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import torchmetrics
except ImportError:
    torchmetrics = None

from src.research.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Evaluates a frozen model against a dataset.

    Parameters
    ----------
    cfg : Any
        Root configuration node.
    model : BaseModel
        The frozen, evaluation-ready model.
    """

    def __init__(self, cfg: Any, model: BaseModel) -> None:
        if torchmetrics is None:
            raise ImportError("Please install torchmetrics: pip install torchmetrics")

        self.cfg = cfg
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.num_classes = cfg.dataset.labels.num_classes
        self.ignore_index = 255

        # Initialize TorchMetrics collection
        self.metrics = torchmetrics.MetricCollection({
            "oa": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index),
            "macro_f1": torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=self.ignore_index),
            "miou": torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index),
            "kappa": torchmetrics.CohenKappa(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index),
            "confusion_matrix": torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index),
        }).to(self.device)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        """
        Run the evaluation loop.

        Returns
        -------
        dict[str, Any]
            Dictionary of computed metrics.
        """
        self.metrics.reset()
        
        logger.info("Starting Evaluation Engine on %d batches...", len(dataloader))
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)
            
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)
            
            self.metrics.update(preds, y)
            
        results = self.metrics.compute()
        
        # Format results (tensors to floats/numpy)
        formatted_results = {}
        for k, v in results.items():
            if k == "confusion_matrix":
                formatted_results[k] = v.cpu().numpy()
            else:
                formatted_results[k] = v.item()
                
        logger.info("Evaluation complete. mIoU: %.4f | OA: %.4f", 
                    formatted_results["miou"], formatted_results["oa"])
                    
        return formatted_results


__all__ = ["EvaluationEngine"]
