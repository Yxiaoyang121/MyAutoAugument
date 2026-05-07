from __future__ import annotations

import cv2
import numpy as np

from AutoAugment.bbox.affine import apply_affine_to_bboxes
from AutoAugment.bbox.clip import clip_filter_bboxes
from AutoAugment.transforms.base import BaseTransform, build_sample, choose_value, sample_image_size
from AutoAugment.types import SampleDict


class Scale(BaseTransform):
    """以图像中心为基准缩放，模拟距离和成像倍率变化。"""

    def __init__(
        self,
        scale: float | None = None,
        scale_range: tuple[float, float] = (0.8, 1.2),
        probability: float = 1.0,
        seed: int | None = None,
        border_value: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.scale = scale
        self.scale_range = scale_range
        self.border_value = border_value

    def apply(self, sample: SampleDict, rng: np.random.Generator) -> SampleDict:
        """执行中心缩放并同步变换边界框。"""
        width, height = sample_image_size(sample)
        scale = choose_value(rng, self.scale, self.scale_range)
        if scale <= 0:
            raise ValueError("scale 必须大于 0")
        center_x = width / 2.0
        center_y = height / 2.0
        matrix = np.array(
            [
                [scale, 0.0, center_x * (1.0 - scale)],
                [0.0, scale, center_y * (1.0 - scale)],
            ],
            dtype=np.float32,
        )
        image = cv2.warpAffine(
            sample["image"],
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )
        bboxes = apply_affine_to_bboxes(sample["bboxes"], matrix)
        bboxes, labels = clip_filter_bboxes(bboxes, sample["labels"], width, height)
        return build_sample(image, bboxes, labels, sample)
