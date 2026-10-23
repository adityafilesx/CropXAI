"""
CropXAI Pretrained Registry
===========================

Maintains metadata about available pretrained foundation models,
including architectural capabilities, licensing, citations, and download links.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    name: str
    checkpoint_name: str
    source: str  # e.g., "huggingface", "timm", "local"
    version: str
    training_dataset: str
    license: str
    embedding_dim: int
    input_resolution: tuple[int, ...]
    publication_reference: str
    doi: str
    publication_year: int
    hash_sha256: str | None = None
    remote_url: str | None = None


class PretrainedRegistry:
    """Registry holding metadata for all officially supported Foundation Models."""

    _registry: dict[str, CheckpointMetadata] = {}

    @classmethod
    def register(cls, meta: CheckpointMetadata) -> None:
        cls._registry[meta.name] = meta
        logger.debug("Registered pretrained checkpoint metadata for: %s", meta.name)

    @classmethod
    def get(cls, name: str) -> CheckpointMetadata:
        if name not in cls._registry:
            raise KeyError(f"Checkpoint {name} not found in PretrainedRegistry.")
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())


# Register core models
PretrainedRegistry.register(
    CheckpointMetadata(
        name="prithvi_eo2",
        checkpoint_name="ibm-nasa-geospatial/Prithvi-100M",
        source="huggingface",
        version="v1.0",
        training_dataset="HLS (Harmonized Landsat Sentinel-2)",
        license="Apache 2.0",
        embedding_dim=768,
        input_resolution=(6, 224, 224),
        publication_reference="Jakubik et al. (2023). Foundation Models for Earth Observation.",
        doi="10.1109/IGARSS52108.2023.10281691",
        publication_year=2023,
    )
)

PretrainedRegistry.register(
    CheckpointMetadata(
        name="segformer_b2",
        checkpoint_name="nvidia/mit-b2",
        source="huggingface",
        version="v1.0",
        training_dataset="ImageNet-1k",
        license="MIT",
        embedding_dim=512,
        input_resolution=(3, 224, 224),
        publication_reference="Xie et al. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation.",
        doi="10.48550/arXiv.2105.15203",
        publication_year=2021,
    )
)

PretrainedRegistry.register(
    CheckpointMetadata(
        name="swin_t",
        checkpoint_name="swin_tiny_patch4_window7_224",
        source="timm",
        version="v1.0",
        training_dataset="ImageNet-1k",
        license="Apache 2.0",
        embedding_dim=768,
        input_resolution=(3, 224, 224),
        publication_reference="Liu et al. (2021). Swin Transformer.",
        doi="10.1109/ICCV48922.2021.00986",
        publication_year=2021,
    )
)

PretrainedRegistry.register(
    CheckpointMetadata(
        name="satmae",
        checkpoint_name="satmae_vit_base_patch16",
        source="local",
        version="v1.0",
        training_dataset="fMoW",
        license="MIT",
        embedding_dim=1024,
        input_resolution=(10, 224, 224), # e.g. S2 10-band
        publication_reference="Cong et al. (2022). SatMAE: Pre-training Transformers for Spaceborne Earth Observation.",
        doi="10.48550/arXiv.2207.08051",
        publication_year=2022,
    )
)

__all__ = ["CheckpointMetadata", "PretrainedRegistry"]
