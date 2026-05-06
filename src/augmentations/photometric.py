from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from src.augmentations.base import centered_factor, get_param_rng, get_strength
from src.utils.image import clip_uint8, ensure_uint8


def brightness(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """亮度调整，模拟光源整体强弱变化。"""
    image = ensure_uint8(image)
    alpha = centered_factor(params, "alpha", "alpha_range", (0.75, 1.25))
    beta = float(params.get("beta", 0.0)) * get_strength(params)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def contrast(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """对比度调整，模拟材质反射率差异。"""
    image = ensure_uint8(image)
    alpha = centered_factor(params, "alpha", "alpha_range", (0.75, 1.25))
    mean = np.mean(image)
    table = np.array([((index - mean) * alpha + mean) for index in range(256)])
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def gamma_correction(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """伽马校正，模拟非线性成像响应变化。"""
    image = ensure_uint8(image)
    gamma = centered_factor(params, "gamma", "gamma_range", (0.6, 1.8))
    gamma = max(0.05, gamma)
    table = np.array([((index / 255.0) ** (1.0 / gamma)) * 255.0 for index in range(256)])
    return cv2.LUT(image, np.clip(table, 0, 255).astype(np.uint8))


def saturation(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """饱和度调整，模拟彩色相机色彩响应波动。"""
    image = ensure_uint8(image)
    if image.ndim == 2 or image.shape[2] == 1:
        return image.copy()
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha_channel = image[:, :, 3]
    else:
        bgr = image
        alpha_channel = None
    factor = centered_factor(params, "factor", "factor_range", (0.5, 1.5))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if alpha_channel is not None:
        return np.dstack([result, alpha_channel])
    return result


def reflection(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """反光与局部光照变化模拟，用于覆盖高亮斑和照明不均。"""
    image = ensure_uint8(image)
    strength = get_strength(params)
    rng = get_param_rng(params)
    height, width = image.shape[:2]
    if "center" in params:
        center_x, center_y = params["center"]
        center_x, center_y = float(center_x), float(center_y)
    else:
        center_x = float(rng.uniform(0.25 * width, 0.75 * width))
        center_y = float(rng.uniform(0.25 * height, 0.75 * height))
    radius_fraction = float(params.get("radius_fraction", 0.35))
    radius_x = max(1.0, width * radius_fraction)
    radius_y = max(1.0, height * radius_fraction * float(params.get("aspect_ratio", 0.65)))
    max_gain = float(params.get("max_gain", 95.0)) * strength
    y_grid, x_grid = np.ogrid[:height, :width]
    distance = ((x_grid - center_x) / radius_x) ** 2 + ((y_grid - center_y) / radius_y) ** 2
    mask = np.exp(-distance * float(params.get("falloff", 2.2))) * max_gain
    if image.ndim == 2 or image.shape[2] == 1:
        return clip_uint8(image.astype(np.float32) + mask)
    result = image.astype(np.float32)
    color = np.asarray(params.get("color", (1.0, 1.0, 1.0)), dtype=np.float32)
    editable_channels = min(3, result.shape[2])
    color = color[:editable_channels]
    while len(color) < editable_channels:
        color = np.append(color, 1.0)
    result[:, :, :editable_channels] += mask[:, :, None] * color.reshape(1, 1, -1)
    return clip_uint8(result)
