from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


AugmentationFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, Optional[dict[str, Any]], float, Optional[np.random.Generator]],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


@dataclass(frozen=True)
class AugmentationSpec:
    name: str
    fn: AugmentationFn
    changes_bboxes: bool = False


_REGISTRY: dict[str, AugmentationSpec] = {}


def register_augmentation(
    name: str,
    fn: AugmentationFn | None = None,
    *,
    changes_bboxes: bool = False,
) -> AugmentationFn | Callable[[AugmentationFn], AugmentationFn]:
    """Register an augmentation function by name."""

    def decorator(inner_fn: AugmentationFn) -> AugmentationFn:
        key = name.strip().lower()
        if not key:
            raise ValueError("augmentation name cannot be empty")
        _REGISTRY[key] = AugmentationSpec(name=key, fn=inner_fn, changes_bboxes=changes_bboxes)
        return inner_fn

    if fn is not None:
        return decorator(fn)
    return decorator


def get_augmentation(name: str) -> AugmentationFn:
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown augmentation '{name}'. Available: {available}")
    return _REGISTRY[key].fn


def get_augmentation_spec(name: str) -> AugmentationSpec:
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown augmentation '{name}'. Available: {available}")
    return _REGISTRY[key]


def list_augmentations(*, changes_bboxes: bool | None = None) -> list[str]:
    names = []
    for name, spec in _REGISTRY.items():
        if changes_bboxes is None or spec.changes_bboxes == changes_bboxes:
            names.append(name)
    return sorted(names)


def apply_augmentation(
    name: str,
    image: np.ndarray,
    labels: np.ndarray | None = None,
    bboxes: np.ndarray | None = None,
    *,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fn = get_augmentation(name)
    return fn(image, labels, bboxes, params, strength, rng)
