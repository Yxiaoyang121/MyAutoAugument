from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.utils.image import get_range
from src.utils.random import get_rng


def get_strength(params: Mapping[str, Any], default: float = 0.5) -> float:
    """读取并裁剪增强强度。"""
    return float(np.clip(float(params.get("strength", default)), 0.0, 1.0))


def get_param_rng(params: Mapping[str, Any]) -> np.random.Generator:
    """从参数中读取随机数生成器。"""
    return get_rng(params.get("rng"))


def centered_factor(
    params: Mapping[str, Any],
    value_key: str,
    range_key: str,
    default_range: tuple[float, float],
) -> float:
    """围绕 1.0 生成可由强度控制的缩放因子。"""
    if value_key in params:
        return float(params[value_key])
    low, high = get_range(params, range_key, default_range)
    strength = get_strength(params)
    rng = get_param_rng(params)
    down = max(0.0, 1.0 - low)
    up = max(0.0, high - 1.0)
    return 1.0 + float(rng.uniform(-down, up)) * strength


def signed_value(
    params: Mapping[str, Any],
    value_key: str,
    max_key: str,
    default_max: float,
) -> float:
    """生成带正负方向的强度控制数值。"""
    if value_key in params:
        return float(params[value_key])
    strength = get_strength(params)
    rng = get_param_rng(params)
    max_value = float(params.get(max_key, default_max)) * strength
    return float(rng.uniform(-max_value, max_value))
