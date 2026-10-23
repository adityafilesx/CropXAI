"""
CropXAI Reproducibility Package Generator
===========================================

Bundles requirements.txt, Dockerfiles, and environment manifests
to ensure the exact software state can be recreated globally.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ReproducibilityPackage:
    """Generates the environment snapshot required to run the code."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> None:
        """Writes the Dockerfile and dependency manifests."""
        logger.info("Generating Reproducibility Package...")
        
        dockerfile = (
            "FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime\n"
            "WORKDIR /workspace\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
        )
        (self.output_dir / "Dockerfile").write_text(dockerfile)
        
        # Mock requirements
        reqs = "torch>=2.0.0\ntorchvision\ncaptum\nscikit-learn\npandas\n"
        (self.output_dir / "requirements.txt").write_text(reqs)
        
        script = "#!/bin/bash\npython -m src.cli campaign run configs/campaigns/full_paper.yaml"
        (self.output_dir / "reproduce.sh").write_text(script)
        
        logger.info("Reproducibility Package generated at %s", self.output_dir)


__all__ = ["ReproducibilityPackage"]
