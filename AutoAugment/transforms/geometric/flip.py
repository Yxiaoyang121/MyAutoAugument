from __future__ import annotations

import cv2
import numpy as np

from AutoAugment.bbox.clip import clip_filter_bboxes
from AutoAugment.transforms.base import BaseTransform, build_sample, sample_image_size
from AutoAugment.types import SampleDict


class HorizontalFlip(BaseTransform):
    """水平翻转，同时同步更新 xyxy 边界框。"""

    def apply(self, sample: SampleDict, rng: np.random.Generator) -> SampleDict:
        """执行水平翻转。"""
        width, height = sample_image_size(sample)
        image = cv2.flip(sample["image"], 1)
        bboxes = sample["bboxes"].copy()
        if len(bboxes) > 0:
            x1 = bboxes[:, 0].copy()
            x2 = bboxes[:, 2].copy()
            bboxes[:, 0] = width - x2
            bboxes[:, 2] = width - x1
        bboxes, labels = clip_filter_bboxes(bboxes, sample["labels"], width, height)
        return build_sample(image, bboxes, labels, sample)


class VerticalFlip(BaseTransform):
    """垂直翻转，同时同步更新 xyxy 边界框。"""

    def apply(self, sample: SampleDict, rng: np.random.Generator) -> SampleDict:
        """执行垂直翻转。"""
        width, height = sample_image_size(sample)
        image = cv2.flip(sample["image"], 0)
        bboxes = sample["bboxes"].copy()
        if len(bboxes) > 0:
            y1 = bboxes[:, 1].copy()
            y2 = bboxes[:, 3].copy()
            bboxes[:, 1] = height - y2
            bboxes[:, 3] = height - y1
        bboxes, labels = clip_filter_bboxes(bboxes, sample["labels"], width, height)
        return build_sample(image, bboxes, labels, sample)
