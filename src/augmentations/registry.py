from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from src.augmentations.blur import blur
from src.augmentations.geometric import flip, mechanical_deviation, perspective_transform, rotation, shift
from src.augmentations.noise import gaussian_noise
from src.augmentations.photometric import brightness, contrast, gamma_correction, reflection, saturation

AugmentationFn = Callable[[np.ndarray, Mapping[str, Any]], np.ndarray]

AUGMENTATION_REGISTRY: dict[str, AugmentationFn] = {
    "brightness": brightness,
    "contrast": contrast,
    "blur": blur,
    "gaussian_noise": gaussian_noise,
    "noise": gaussian_noise,
    "rotation": rotation,
    "flip": flip,
    "shift": shift,
    "perspective": perspective_transform,
    "perspective_transform": perspective_transform,
    "gamma": gamma_correction,
    "gamma_correction": gamma_correction,
    "saturation": saturation,
    "reflection": reflection,
    "lighting_variation": reflection,
    "mechanical_deviation": mechanical_deviation,
}

CANONICAL_AUGMENTATIONS: tuple[str, ...] = (
    "brightness",
    "contrast",
    "blur",
    "gaussian_noise",
    "rotation",
    "flip",
    "shift",
    "perspective",
    "gamma",
    "saturation",
    "reflection",
)


def get_augmentation(name: str) -> AugmentationFn:
    """按名称获取增强算子。"""
    key = name.strip().lower()
    if key not in AUGMENTATION_REGISTRY:
        available = ", ".join(sorted(AUGMENTATION_REGISTRY))
        raise KeyError(f"未知增强算子: {name}，可用算子: {available}")
    return AUGMENTATION_REGISTRY[key]


def apply_augmentation(name: str, image: np.ndarray, params: Mapping[str, Any] | None = None) -> np.ndarray:
    """使用统一接口调用增强算子。"""
    return get_augmentation(name)(image, params or {})


def list_augmentations(include_aliases: bool = False) -> tuple[str, ...]:
    """列出可用增强算子名称。"""
    if include_aliases:
        return tuple(sorted(AUGMENTATION_REGISTRY))
    return CANONICAL_AUGMENTATIONS
