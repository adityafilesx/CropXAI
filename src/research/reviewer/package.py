"""
CropXAI Reviewer Package Generator
====================================

Bundles the compiled paper, supplementary PDF, artifact registry, and a 
reproduce.sh script specifically designed for anonymous reviewers to 
independently validate the manuscript claims.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class ReviewerPackage:
    """Generates the review_package.zip for double-blind submission."""

    def __init__(self, output_dir: str | Path = "review_package/") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def bundle(self, paper_src_dir: str | Path) -> None:
        """Copies final assets into a reviewer-friendly bundle."""
        logger.info("Bundling Reviewer Package...")
        
        src = Path(paper_src_dir)
        
        # Mock copying of final PDFs and reports
        if (src / "main.pdf").exists():
            shutil.copy(src / "main.pdf", self.output_dir / "paper.pdf")
            
        (self.output_dir / "review_report.md").touch()
        (self.output_dir / "artifact_checksums.csv").touch()
        
        script = "#!/bin/bash\ncropxai reproduce reviewer_config.yaml"
        (self.output_dir / "reproduce.sh").write_text(script)
        
        logger.info("Reviewer Package ready at %s", self.output_dir)


__all__ = ["ReviewerPackage"]
