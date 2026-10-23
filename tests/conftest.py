"""
CropXAI Pytest Configuration
=============================

Provides shared fixtures for all test tiers (unit, integration, e2e,
performance, regression).
"""

import pytest
from pathlib import Path
from hydra import compose, initialize_config_dir

@pytest.fixture(scope="session")
def cropxai_config():
    """Load the root Hydra configuration for testing."""
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config")
    return cfg

@pytest.fixture
def dummy_image_tensor():
    """Provide a dummy 4D tensor (T, C, H, W) for model tests."""
    import torch
    return torch.randn(1, 3, 10, 64, 64)  # B=1, T=3, C=10, H=64, W=64
