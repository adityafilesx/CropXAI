"""
CropXAI Environment Capture
=============================

Standalone environment introspection utilities.
Re-exports from reproducibility for convenience.
"""

from src.utils.random.reproducibility import capture_environment, _get_git_info  # noqa: F401

__all__ = ["capture_environment", "_get_git_info"]
