"""
CropXAI Representation Analyzer
=================================

Computes quantitative metrics of the learned feature space, including
Intrinsic Dimensionality, CKA, Feature Redundancy, and Effective Rank.
Also handles visualizations (PCA, UMAP, t-SNE).
"""

from __future__ import annotations

import logging
from typing import Any
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
except ImportError:
    PCA = None
    TSNE = None
    umap = None

logger = logging.getLogger(__name__)


class RepresentationAnalyzer:
    """Analyzes the structural properties of Foundation Model embeddings."""

    @staticmethod
    def compute_intrinsic_dimensionality(features: torch.Tensor, threshold: float = 0.99) -> int:
        """Estimates Intrinsic Dimensionality using PCA variance explained."""
        if PCA is None:
            logger.warning("sklearn not installed. Skipping ID computation.")
            return -1
            
        X = features.cpu().numpy()
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
            
        pca = PCA().fit(X)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        return int(np.argmax(cumulative_variance >= threshold) + 1)

    @staticmethod
    def compute_effective_rank(features: torch.Tensor) -> float:
        """
        Computes the effective rank of the feature covariance matrix.
        Measures the uniformness of the singular value distribution.
        """
        X = features
        if X.ndim > 2:
            X = X.flatten(1)
            
        # Center features
        X = X - X.mean(dim=0, keepdim=True)
        # Covariance
        cov = (X.T @ X) / (X.shape[0] - 1)
        
        # Singular values
        try:
            _, S, _ = torch.svd(cov)
        except RuntimeError:
            return 0.0 # SVD fails if matrix is degenerate
            
        # Normalize singular values
        p = S / S.sum()
        
        # Shannon entropy of singular value distribution
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        effective_rank = math.exp(entropy.item())
        return effective_rank

    @staticmethod
    def compute_feature_redundancy(features: torch.Tensor) -> float:
        """
        Computes feature redundancy based on mean absolute correlation 
        between feature dimensions.
        """
        X = features
        if X.ndim > 2:
            X = X.flatten(1)
            
        # Normalize each feature dimension
        X = X - X.mean(dim=0, keepdim=True)
        X = X / (X.std(dim=0, keepdim=True) + 1e-8)
        
        # Correlation matrix
        corr = (X.T @ X) / (X.shape[0] - 1)
        
        # Mean absolute correlation (excluding diagonal)
        n = corr.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=corr.device)
        redundancy = corr[mask].abs().mean()
        return float(redundancy.item())

    @staticmethod
    def plot_umap(features: np.ndarray, labels: np.ndarray, output_path: str) -> None:
        """Generates a UMAP projection colored by class labels."""
        if umap is None:
            logger.warning("umap-learn not installed. Skipping UMAP.")
            return
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        from src.research.visualization.export import apply_publication_style, save_figure
        
        apply_publication_style()
        
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
            
        reducer = umap.UMAP(random_state=42)
        embedding = reducer.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=embedding[:, 0], 
            y=embedding[:, 1], 
            hue=labels,
            palette="tab10",
            s=20,
            alpha=0.8,
            ax=ax
        )
        ax.set_title("UMAP Projection of Pretrained Features")
        
        save_figure(fig, "umap_projection", output_path)


__all__ = ["RepresentationAnalyzer"]
