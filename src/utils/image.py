from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np


def validate_image(image: np.ndarray) -> np.ndarray:
    """校验图像是否为非空的灰度图或彩色图。"""
    if image is None:
        raise ValueError("图像不能为空")
    if not isinstance(image, np.ndarray):
        raise TypeError("图像必须是 numpy.ndarray")
    if image.size == 0:
        raise ValueError("图像内容不能为空")
    if image.ndim not in (2, 3):
        raise ValueError("图像维度必须是二维灰度图或三维彩色图")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("彩色图通道数必须是 1、3 或 4")
    return image


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """将图像转换为 OpenCV 增强算子常用的 uint8 格式。"""
    image = validate_image(image)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        max_value = float(np.nanmax(image))
        scaled = image * 255.0 if max_value <= 1.0 else image
        return np.clip(scaled, 0, 255).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def clip_uint8(values: np.ndarray) -> np.ndarray:
    """将数值数组裁剪到 uint8 图像范围。"""
    return np.clip(values, 0, 255).astype(np.uint8)


def to_float01(image: np.ndarray) -> np.ndarray:
    """将图像转换为归一化浮点格式。"""
    return ensure_uint8(image).astype(np.float32) / 255.0


def get_range(params: Mapping[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    """从参数中读取范围，并校验上下界。"""
    value = params.get(key, default)
    if len(value) != 2:
        raise ValueError(f"{key} 必须包含两个数值")
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{key} 的下界不能大于上界")
    return low, high


def odd_kernel_size(value: int) -> int:
    """将整数调整为 OpenCV 高斯模糊需要的正奇数核。"""
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def resize_for_feature(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """将图像缩放到特征提取所需尺寸。"""
    image = ensure_uint8(image)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
