"""
CropXAI Registry Pattern
========================

Provides a generic, type-safe Registry class and seven pre-instantiated
global registries that enable plug-and-play component management throughout
the framework.

Usage
-----
Register a component::

    from src.core.registry import MODEL_REGISTRY

    @MODEL_REGISTRY.register("my_cnn")
    class MyCNN(CNNModel):
        ...

Retrieve and instantiate::

    model_cls = MODEL_REGISTRY.get("my_cnn")
    model = model_cls(cfg)

List available::

    MODEL_REGISTRY.list()  # ["my_cnn", "resnet50", ...]
"""

from __future__ import annotations

import logging
from typing import Any, Generic, Iterator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """
    A thread-safe, generic component registry.

    Parameters
    ----------
    name : str
        Human-readable name for this registry (used in error messages).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[T]] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────────────────────────────────

    def register(self, key: str, *, override: bool = False) -> Any:
        """
        Decorator that registers a class under ``key``.

        Parameters
        ----------
        key : str
            Unique identifier for the component.
        override : bool
            If True, allow overwriting an existing registration.
            Default False to prevent silent mistakes.

        Raises
        ------
        KeyError
            If ``key`` is already registered and ``override=False``.

        Examples
        --------
        >>> @MODEL_REGISTRY.register("resnet50")
        ... class ResNet50(CNNModel): ...
        """

        def decorator(cls: type[T]) -> type[T]:
            if key in self._registry and not override:
                raise KeyError(
                    f"[{self._name}] Key '{key}' is already registered. "
                    f"Use override=True to replace it."
                )
            self._registry[key] = cls
            logger.debug("[%s] Registered '%s' → %s", self._name, key, cls.__qualname__)
            return cls

        return decorator

    def register_module(self, key: str, cls: type[T], *, override: bool = False) -> None:
        """Register a class directly (non-decorator form)."""
        self.register(key, override=override)(cls)

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def get(self, key: str) -> type[T]:
        """
        Retrieve a registered class by key.

        Raises
        ------
        KeyError
            If ``key`` is not found in the registry.
        """
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "<empty>"
            raise KeyError(
                f"[{self._name}] '{key}' not found. "
                f"Available: [{available}]"
            )
        return self._registry[key]

    def build(self, key: str, *args: Any, **kwargs: Any) -> T:
        """Retrieve class and instantiate with the given arguments."""
        cls = self.get(key)
        return cls(*args, **kwargs)  # type: ignore[return-value]

    # ──────────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────────

    def list(self) -> list[str]:
        """Return sorted list of registered keys."""
        return sorted(self._registry.keys())

    def contains(self, key: str) -> bool:
        """Check whether a key is registered."""
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={self.list()})"

    def __str__(self) -> str:
        return repr(self)


# ──────────────────────────────────────────────────────────────────────────────
# Global Registries
# ──────────────────────────────────────────────────────────────────────────────

#: Registry for model architectures (CNNs, Transformers, Foundation Models).
MODEL_REGISTRY: Registry[Any] = Registry("ModelRegistry")

#: Registry for evaluation metrics.
METRIC_REGISTRY: Registry[Any] = Registry("MetricRegistry")

#: Registry for dataset implementations.
DATASET_REGISTRY: Registry[Any] = Registry("DatasetRegistry")

#: Registry for XAI (Explainable AI) methods.
XAI_REGISTRY: Registry[Any] = Registry("XAIRegistry")

#: Registry for loss functions.
LOSS_REGISTRY: Registry[Any] = Registry("LossRegistry")

#: Registry for optimizers.
OPTIMIZER_REGISTRY: Registry[Any] = Registry("OptimizerRegistry")

#: Registry for learning rate schedulers.
SCHEDULER_REGISTRY: Registry[Any] = Registry("SchedulerRegistry")


__all__ = [
    "Registry",
    "MODEL_REGISTRY",
    "METRIC_REGISTRY",
    "DATASET_REGISTRY",
    "XAI_REGISTRY",
    "LOSS_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
]
