from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from src.augmentations.base import get_strength
from src.utils.image import ensure_uint8, odd_kernel_size


def blur(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """高斯模糊增强，模拟离焦、运动残留和成像锐度下降。"""
    image = ensure_uint8(image)
    strength = get_strength(params)
    if "kernel_size" in params:
        kernel_size = odd_kernel_size(int(params["kernel_size"]))
    else:
        max_kernel = odd_kernel_size(int(params.get("max_kernel_size", 9)))
        kernel_size = odd_kernel_size(1 + int(round((max_kernel - 1) * strength)))
    sigma = float(params.get("sigma", 0.0))
    if kernel_size <= 1:
        return image.copy()
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)
