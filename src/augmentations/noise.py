from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.augmentations.base import get_param_rng, get_strength
from src.utils.image import ensure_uint8


def gaussian_noise(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """高斯噪声增强，模拟粉尘、传感器噪声和采集干扰。"""
    image = ensure_uint8(image)
    rng = get_param_rng(params)
    strength = get_strength(params)
    mean = float(params.get("mean", 0.0))
    if "sigma" in params:
        sigma = float(params["sigma"])
    else:
        sigma_range = params.get("sigma_range", (0.01, 0.05))
        sigma = float(rng.uniform(float(sigma_range[0]), float(sigma_range[1]))) * strength
    image_float = image.astype(np.float32) / 255.0
    noise = rng.normal(mean, sigma, image_float.shape)
    result = np.clip(image_float + noise, 0.0, 1.0)
    return (result * 255.0).astype(np.uint8)
