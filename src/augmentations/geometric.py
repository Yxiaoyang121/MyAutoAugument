from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from src.augmentations.base import get_param_rng, get_strength, signed_value
from src.utils.image import ensure_uint8


def rotation(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """旋转增强，模拟相机或工件姿态角度偏差。"""
    image = ensure_uint8(image)
    height, width = image.shape[:2]
    angle = signed_value(params, "angle", "max_angle", 12.0)
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def flip(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """翻转增强，模拟方向不敏感场景下的样本扩展。"""
    image = ensure_uint8(image)
    if get_strength(params) <= 0.0:
        return image.copy()
    mode = params.get("mode")
    if mode is None:
        rng = get_param_rng(params)
        mode = rng.choice(["horizontal", "vertical", "both"])
    flip_code = {"horizontal": 1, "vertical": 0, "both": -1}.get(str(mode), mode)
    if flip_code not in (-1, 0, 1):
        raise ValueError("翻转模式必须是 horizontal、vertical、both、-1、0 或 1")
    return cv2.flip(image, int(flip_code))


def shift(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """平移增强，模拟机械定位和装夹误差。"""
    image = ensure_uint8(image)
    height, width = image.shape[:2]
    strength = get_strength(params)
    rng = get_param_rng(params)
    if "dx" in params:
        dx = float(params["dx"])
    else:
        max_dx = float(params.get("max_dx", width * float(params.get("max_fraction", 0.06)))) * strength
        dx = float(rng.uniform(-max_dx, max_dx))
    if "dy" in params:
        dy = float(params["dy"])
    else:
        max_dy = float(params.get("max_dy", height * float(params.get("max_fraction", 0.06)))) * strength
        dy = float(rng.uniform(-max_dy, max_dy))
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def perspective_transform(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """透视变换，模拟相机视角和工件平面姿态变化。"""
    image = ensure_uint8(image)
    height, width = image.shape[:2]
    strength = get_strength(params)
    rng = get_param_rng(params)
    jitter_fraction = float(params.get("jitter_fraction", 0.08)) * strength
    jitter_x = width * jitter_fraction
    jitter_y = height * jitter_fraction
    source = np.float32(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ]
    )
    if "dst_points" in params:
        target = np.float32(params["dst_points"])
    else:
        offsets = rng.uniform(
            low=[-jitter_x, -jitter_y],
            high=[jitter_x, jitter_y],
            size=(4, 2),
        ).astype(np.float32)
        target = source + offsets
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def mechanical_deviation(image: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """机械位姿偏差，兼容旧接口中的亚像素平移和微小旋转。"""
    image = ensure_uint8(image)
    height, width = image.shape[:2]
    dx = float(params.get("dx", 0.0))
    dy = float(params.get("dy", 0.0))
    angle = float(params.get("angle_deg", params.get("angle", 0.0)))
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
