from __future__ import annotations

from AutoAugment.augmentations import ops as _ops  # noqa: F401
from AutoAugment.augmentations.registry import (
    apply_augmentation,
    get_augmentation,
    get_augmentation_spec,
    list_augmentations,
    register_augmentation,
)

__all__ = [
    "apply_augmentation",
    "get_augmentation",
    "get_augmentation_spec",
    "list_augmentations",
    "register_augmentation",
]
