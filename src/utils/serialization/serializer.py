"""
CropXAI Serialization Utilities
=================================

Safe, format-agnostic serialization for experiment artifacts:
  - YAML (OmegaConf / PyYAML)
  - JSON
  - NumPy arrays
  - PyTorch tensors
  - Pandas DataFrames
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# JSON
# ──────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Serialize ``data`` to a JSON file, handling NumPy types."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=_NumpyEncoder)


def load_json(path: str | Path) -> Any:
    """Deserialize a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# YAML
# ──────────────────────────────────────────────────────────────────────────────

def save_yaml(path: str | Path, data: Any) -> None:
    """Serialize a plain dict to YAML (uses PyYAML)."""
    import yaml
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def load_yaml(path: str | Path) -> Any:
    """Deserialize a YAML file."""
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_omegaconf(path: str | Path, cfg: Any) -> None:
    """Save an OmegaConf DictConfig to YAML (resolved)."""
    from omegaconf import OmegaConf  # type: ignore[import]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# NumPy
# ──────────────────────────────────────────────────────────────────────────────

def save_numpy(path: str | Path, array: np.ndarray) -> None:
    """Save a NumPy array to ``.npy`` format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(p), array)


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from ``.npy`` format."""
    return np.load(str(path), allow_pickle=False)


# ──────────────────────────────────────────────────────────────────────────────
# Pandas
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(path: str | Path, df: pd.DataFrame, index: bool = False) -> None:
    """Save a DataFrame to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file as a DataFrame."""
    return pd.read_csv(path)


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append a single row to a CSV file (creates headers if new)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    df_row.to_csv(p, mode="a", header=not p.exists(), index=False)


__all__ = [
    "save_json", "load_json",
    "save_yaml", "load_yaml", "save_omegaconf",
    "save_numpy", "load_numpy",
    "save_csv", "load_csv", "append_csv",
]
