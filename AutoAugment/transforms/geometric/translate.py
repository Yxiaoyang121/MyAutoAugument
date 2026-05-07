from __future__ import annotations

import cv2
import numpy as np

from AutoAugment.bbox.affine import apply_affine_to_bboxes
from AutoAugment.bbox.clip import clip_filter_bboxes
from AutoAugment.transforms.base import BaseTransform, build_sample, choose_value, sample_image_size
from AutoAugment.types import SampleDict


class Translate(BaseTransform):
    """平移增强，用于模拟相机、工装或 ROI 定位偏差。"""

    def __init__(
        self,
        dx: float | None = None,
        dy: float | None = None,
        dx_range: tuple[float, float] = (-0.08, 0.08),
        dy_range: tuple[float, float] = (-0.08, 0.08),
        probability: float = 1.0,
        seed: int | None = None,
        border_value: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.dx = dx
        self.dy = dy
        self.dx_range = dx_range
        self.dy_range = dy_range
        self.border_value = border_value

    def apply(self, sample: SampleDict, rng: np.random.Generator) -> SampleDict:
        """执行平移并同步变换边界框。"""
        width, height = sample_image_size(sample)
        dx = choose_value(rng, self.dx, self.dx_range)
        dy = choose_value(rng, self.dy, self.dy_range)
        dx_pixels = dx * width if abs(dx) <= 1.0 else dx
        dy_pixels = dy * height if abs(dy) <= 1.0 else dy
        matrix = np.array([[1.0, 0.0, dx_pixels], [0.0, 1.0, dy_pixels]], dtype=np.float32)
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
