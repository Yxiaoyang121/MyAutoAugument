from __future__ import annotations

import numpy as np

from AutoAugment.bbox.clip import clip_filter_bboxes
from AutoAugment.transforms.base import BaseTransform, build_sample, choose_int_value, sample_image_size
from AutoAugment.types import SampleDict


class RandomCrop(BaseTransform):
    """随机裁剪增强，裁剪后平移、裁剪并过滤 bbox。"""

    def __init__(
        self,
        crop_box: tuple[int, int, int, int] | None = None,
        crop_size: tuple[int, int] | None = None,
        width_range: tuple[int, int] | None = None,
        height_range: tuple[int, int] | None = None,
        min_visibility: float = 0.2,
        max_trials: int = 20,
        probability: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(probability=probability, seed=seed)
        self.crop_box = crop_box
        self.crop_size = crop_size
        self.width_range = width_range
        self.height_range = height_range
        self.min_visibility = min_visibility
        self.max_trials = max(1, max_trials)

    def apply(self, sample: SampleDict, rng: np.random.Generator) -> SampleDict:
        """执行裁剪并确保保留有效 bbox。"""
        width, height = sample_image_size(sample)
        original_area = _bbox_area(sample["bboxes"])
        for _ in range(self.max_trials):
            x1, y1, x2, y2 = self._sample_crop_box(width, height, rng)
            image, bboxes, labels = self._crop_once(sample, x1, y1, x2, y2, original_area)
            if len(sample["bboxes"]) == 0 or len(bboxes) > 0:
                return build_sample(image, bboxes, labels, sample)
        return build_sample(sample["image"].copy(), sample["bboxes"].copy(), sample["labels"].copy(), sample)

    def _sample_crop_box(self, width: int, height: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
        """采样裁剪窗口。"""
        if self.crop_box is not None:
            x1, y1, x2, y2 = self.crop_box
            return _sanitize_crop_box(x1, y1, x2, y2, width, height)
        if self.crop_size is not None:
            crop_width, crop_height = self.crop_size
        else:
            width_range = self.width_range or (max(1, int(width * 0.6)), width)
            height_range = self.height_range or (max(1, int(height * 0.6)), height)
            crop_width = choose_int_value(rng, None, width_range)
            crop_height = choose_int_value(rng, None, height_range)
        crop_width = int(np.clip(crop_width, 1, width))
        crop_height = int(np.clip(crop_height, 1, height))
        max_x = width - crop_width
        max_y = height - crop_height
        x1 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y1 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        return x1, y1, x1 + crop_width, y1 + crop_height

    def _crop_once(
        self,
        sample: SampleDict,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        original_area: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """执行一次裁剪并过滤可见面积不足的 bbox。"""
        image = sample["image"][y1:y2, x1:x2].copy()
        crop_width = x2 - x1
        crop_height = y2 - y1
        bboxes = sample["bboxes"].copy()
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] -= x1
            bboxes[:, [1, 3]] -= y1
        clipped, labels, keep = clip_filter_bboxes(
            bboxes,
            sample["labels"],
            crop_width,
            crop_height,
            return_indices=True,
        )
        if len(clipped) == 0:
            return image, clipped, labels
        clipped_area = _bbox_area(clipped)
        visibility = clipped_area / np.maximum(original_area[keep], 1e-6)
        visible_keep = visibility >= self.min_visibility
        return image, clipped[visible_keep], labels[visible_keep]


def _sanitize_crop_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    """将固定裁剪窗口限制在图像范围内。"""
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def _bbox_area(bboxes: np.ndarray) -> np.ndarray:
    """计算 bbox 面积。"""
    if len(bboxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    width = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0])
    height = np.maximum(0.0, bboxes[:, 3] - bboxes[:, 1])
    return width * height
