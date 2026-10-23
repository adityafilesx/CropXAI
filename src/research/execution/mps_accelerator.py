"""
CropXAI MPS Hardware Accelerator
==================================

Detects and binds execution to Apple Silicon (Mac M4) Metal Performance Shaders (MPS).
Ensures that massive matrix multiplications are offloaded to the M4 GPU 
instead of running sequentially on the CPU.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MPSAccelerator:
    """Manages the mapping of models and tensors to Apple Silicon GPU."""

    @staticmethod
    def get_device() -> str:
        """Determines the optimal device for the current hardware."""
        try:
            import torch
            if torch.backends.mps.is_available():
                logger.info("Apple Silicon MPS (Metal Performance Shaders) is AVAILABLE.")
                return "mps"
            elif torch.cuda.is_available():
                logger.info("NVIDIA CUDA is AVAILABLE.")
                return "cuda"
            else:
                logger.warning("No hardware accelerator found. Falling back to CPU. This will be slow.")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not installed. Falling back to CPU simulation.")
            return "cpu"
            
    @staticmethod
    def optimize_memory() -> None:
        """Sets environment variables to optimize M4 unified memory usage."""
        import os
        # M4 Unified Memory performs better with higher limits on fallback
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        logger.info("Optimized PyTorch MPS unified memory configuration.")


__all__ = ["MPSAccelerator"]
