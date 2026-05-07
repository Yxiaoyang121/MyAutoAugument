from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from AutoAugment.transforms.base import BaseTransform, validate_sample
from AutoAugment.types import SampleDict


class Compose:
    """按顺序组合多个目标检测增强算子。"""

    def __init__(self, transforms: Iterable[BaseTransform], seed: int | None = None) -> None:
        self.transforms = list(transforms)
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: SampleDict) -> SampleDict:
        """对 sample 顺序执行增强流水线。"""
        result = validate_sample(sample)
        for transform in self.transforms:
            result = transform(result, rng=self.rng)
        return result
