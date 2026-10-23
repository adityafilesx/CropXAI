"""CropXAI Label Encoder — integer or one-hot encoding for segmentation labels."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Any


class LabelEncoder:
    def __init__(self, num_classes: int, encoding: str = "integer", ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.encoding = encoding
        self.ignore_index = ignore_index

    @classmethod
    def from_cfg(cls, cfg: Any) -> "LabelEncoder":
        le_cfg = cfg.preprocessing.label_encoding
        return cls(
            num_classes=cfg.dataset.labels.num_classes,
            encoding=le_cfg.get("encoding", "integer"),
            ignore_index=le_cfg.get("ignore_index", 255),
        )

    def __call__(self, label: torch.Tensor) -> torch.Tensor:
        """Encode a (H, W) label tensor."""
        label = label.long()
        # Clamp ignore_index values
        label = torch.where(
            label >= self.num_classes,
            torch.tensor(self.ignore_index, dtype=torch.long),
            label,
        )
        if self.encoding == "onehot":
            valid = label != self.ignore_index
            safe_label = label.clone()
            safe_label[~valid] = 0
            onehot = F.one_hot(safe_label, num_classes=self.num_classes).float()
            onehot[~valid] = 0
            return onehot.permute(2, 0, 1)  # (C, H, W)
        return label  # integer encoding


__all__ = ["LabelEncoder"]
