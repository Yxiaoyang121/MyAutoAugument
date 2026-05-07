from __future__ import annotations

import cv2
import numpy as np

from AutoAugment.bbox.affine import apply_affine_to_bboxes
from AutoAugment.bbox.clip import clip_filter_bboxes
from AutoAugment.transforms.base import BaseTransform, build_sample, choose_value, sample_image_size
from AutoAugment.types import SampleDict


class Rotate(BaseTransform):
    """仿射旋转增强，通过四角点变换重新计算 bbox 外接矩形。"""

    def __init__(
        self,
        angle: float | None = None,
        angle_range: tuple[float, float] = (-10.0, 10.0),
        probability: float = 1.0,
        seed: int | None = None,
        border_value: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.angle = angle
        self.angle_range = angle_range
        self.border_value = border_value

    def apply(self, sample: SampleDict, rng: np.random.Generator) -> SampleDict:
        """执行旋转并同步变换 bbox 四个角点。"""
        width, height = sample_image_size(sample)
        angle = choose_value(rng, self.angle, self.angle_range)
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
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
