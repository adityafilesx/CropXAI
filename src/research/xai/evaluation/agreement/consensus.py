"""
CropXAI Consensus & Agreement Metrics
=======================================

Computes Pairwise IoU, Spearman/Pearson correlation between 
different XAI methods, and Human Agreement against Ground Truth masks.
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConsensusAnalyzer:
    """Computes agreement metrics between multiple explanations or ground truth."""

    @staticmethod
    def compute_pairwise_iou(attr1: torch.Tensor, attr2: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Binarizes heatmaps and computes Intersection over Union.
        """
        # Ensure normalized [0, 1]
        a1 = (attr1 - attr1.min()) / (attr1.max() - attr1.min() + 1e-8)
        a2 = (attr2 - attr2.min()) / (attr2.max() - attr2.min() + 1e-8)
        
        mask1 = (a1 > threshold).float()
        mask2 = (a2 > threshold).float()
        
        intersection = (mask1 * mask2).sum()
        union = mask1.sum() + mask2.sum() - intersection
        
        if union == 0:
            return 1.0 # Both empty
        return float((intersection / union).item())

    @staticmethod
    def compute_human_agreement(attr: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Computes Dice/IoU of the explanation against the ground truth semantic mask.
        Indicates if the model focuses on the right object.
        """
        a = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        pred_mask = (a > threshold).float()
        
        # Flatten
        pred_mask = pred_mask.view(-1)
        gt_mask = gt_mask.view(-1)
        
        # Ignore index 255
        valid = (gt_mask != 255)
        pred_mask = pred_mask[valid]
        gt_mask = gt_mask[valid]
        
        # Assume gt_mask is binary for the specific class being explained
        # In a real pipeline, gt_mask needs to be thresholded for target_class
        
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection
        
        if union == 0:
            return 0.0
        return float((intersection / union).item())


__all__ = ["ConsensusAnalyzer"]
