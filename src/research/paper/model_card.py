"""
CropXAI Model Card Generator
==============================

Automatically generates a standardized Markdown Model Card containing:
- Architecture Details
- Parameters & FLOPs
- Inference Latency
- Paper Citation placeholder
- Training Config snapshot
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.research.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """Generates Markdown Model Cards."""

    def __init__(self, exp_dir: str | Path) -> None:
        self.exp_dir = Path(exp_dir)

    def generate(self, model: BaseModel, cfg: Any, metrics: dict[str, float] | None = None) -> Path:
        """Create the model card."""
        card_path = self.exp_dir / "MODEL_CARD.md"
        
        summary = model.model_summary()
        
        content = f"""# Model Card: {summary['model_name']}

## 1. Architecture
- **Backbone**: {cfg.model.get('backbone', 'Unknown')}
- **Decoder**: {cfg.model.get('decoder', 'Unknown')}
- **Input Channels**: {summary['in_channels']}
- **Output Classes**: {summary['num_classes']}

## 2. Complexity
- **Total Parameters**: {summary['total_parameters']:,}
- **Trainable Parameters**: {summary['trainable_parameters']:,}
- **FLOPs (estimated)**: TBD in Phase 3
- **Peak VRAM**: TBD in Phase 3

## 3. Configuration Snapshot
```yaml
Seed: {cfg.experiment.seed}
Optimizer: {cfg.optimizer.name}
Learning Rate: {cfg.optimizer.lr}
Loss Function: {cfg.loss.name}
```
"""
        
        if metrics:
            content += "\n## 4. Evaluation Metrics\n"
            for k, v in metrics.items():
                content += f"- **{k}**: {v:.4f}\n"
                
        content += "\n## 5. Citation\n"
        content += "```bibtex\n@article{cropxai2026,\n  title={...},\n  author={...},\n  year={2026}\n}\n```\n"

        card_path.write_text(content, encoding="utf-8")
        logger.info("Generated Model Card → %s", card_path)
        
        return card_path


__all__ = ["ModelCardGenerator"]
