"""
CropXAI Calibration Metrics
=============================

Computes pixel-level Expected Calibration Error (ECE), 
Maximum Calibration Error (MCE), and Brier Score for semantic segmentation.
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CalibrationMetrics:
    """Computes calibration statistics at the pixel level."""

    def __init__(self, num_bins: int = 10) -> None:
        self.num_bins = num_bins

    def compute_ece(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        """
        Computes pixel-level Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        
        # Flatten for pixel-level computation
        confidences = confidences.flatten()
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Mask out ignore index (e.g., 255) if applicable
        valid_mask = (targets != 255)
        confidences = confidences[valid_mask]
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        accuracies = predictions.eq(targets)
        
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=logits.device)
        
        ece = 0.0
        mce = 0.0
        
        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                error = torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += prop_in_bin * error
                mce = max(mce, error.item())
                
        return {
            "ece": ece.item(),
            "mce": mce
        }

    def compute_brier_score(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Computes the Brier Score (Mean Squared Error between probabilities and one-hot targets).
        """
        probs = F.softmax(logits, dim=1) # (B, C, H, W)
        num_classes = probs.shape[1]
        
        # Create one-hot targets, ignoring invalid indices
        valid_mask = (targets != 255)
        
        # Flatten
        probs = probs.permute(0, 2, 3, 1).reshape(-1, num_classes) # (N, C)
        targets_flat = targets.flatten()
        
        probs = probs[valid_mask.flatten()]
        targets_flat = targets_flat[valid_mask.flatten()]
        
        targets_one_hot = F.one_hot(targets_flat, num_classes=num_classes).float()
        
        brier = torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=1))
        return float(brier.item())


__all__ = ["CalibrationMetrics"]
