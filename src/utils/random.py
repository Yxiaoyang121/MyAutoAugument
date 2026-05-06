from __future__ import annotations

import random

import numpy as np


def create_rng(seed: int | None = None) -> np.random.Generator:
    """创建可复现的随机数生成器。"""
    return np.random.default_rng(seed)


def get_rng(seed_or_rng: int | np.random.Generator | None = None) -> np.random.Generator:
    """将种子或已有生成器统一转换为生成器。"""
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return create_rng(seed_or_rng)


def seed_everything(seed: int) -> np.random.Generator:
    """固定 Python、NumPy 和框架内部使用的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    return create_rng(seed)
