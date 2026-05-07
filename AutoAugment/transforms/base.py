from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from AutoAugment.types import SampleDict


class BaseTransform(ABC):
    """目标检测增强基类，所有增强都接收并返回 sample。"""

    def __init__(self, probability: float = 1.0, seed: int | None = None) -> None:
        self.probability = float(np.clip(probability, 0.0, 1.0))
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample: SampleDict, rng: np.random.Generator | None = None) -> SampleDict:
        """按概率执行增强，并保证输入输出结构一致。"""
        generator = rng or self.rng
        normalized = validate_sample(sample)
        if generator.random() > self.probability:
            return copy_sample(normalized)
        return self.apply(normalized, generator)

    @abstractmethod
    def apply(self, sample: SampleDict, rng: np.random.Generator) -> SampleDict:
        """执行具体增强逻辑。"""


def validate_sample(sample: SampleDict) -> SampleDict:
    """校验目标检测样本结构并规范 dtype。"""
    required = {"image", "bboxes", "labels"}
    missing = required.difference(sample)
    if missing:
        raise KeyError(f"sample 缺少字段: {sorted(missing)}")
    image = sample["image"]
    if not isinstance(image, np.ndarray):
        raise TypeError("image 必须是 numpy.ndarray")
    if image.ndim not in (2, 3):
        raise ValueError("image 必须是二维灰度图或三维彩色图")
    if image.size == 0:
        raise ValueError("image 不能为空")
    bboxes = np.asarray(sample["bboxes"], dtype=np.float32)
    if bboxes.ndim == 1 and bboxes.size == 0:
        bboxes = bboxes.reshape(0, 4)
    if bboxes.ndim != 2 or bboxes.shape[1] != 4:
        raise ValueError("bboxes 必须是 Nx4 的 xyxy 数组")
    labels = np.asarray(sample["labels"])
    if labels.ndim != 1:
        raise ValueError("labels 必须是一维数组")
    if len(labels) != len(bboxes):
        raise ValueError("labels 数量必须与 bboxes 数量一致")
    result = dict(sample)
    result["image"] = image
    result["bboxes"] = bboxes
    result["labels"] = labels
    return result


def copy_sample(sample: SampleDict) -> SampleDict:
    """复制样本，避免增强过程修改原始数据。"""
    copied = dict(sample)
    copied["image"] = sample["image"].copy()
    copied["bboxes"] = np.asarray(sample["bboxes"], dtype=np.float32).copy()
    copied["labels"] = np.asarray(sample["labels"]).copy()
    return copied


def build_sample(image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray, source: SampleDict) -> SampleDict:
    """构造增强后的样本并保留额外元信息。"""
    result = dict(source)
    result["image"] = image
    result["bboxes"] = bboxes.astype(np.float32, copy=False)
    result["labels"] = labels.copy()
    return result


def sample_image_size(sample: SampleDict) -> tuple[int, int]:
    """返回样本图像宽高。"""
    height, width = sample["image"].shape[:2]
    return width, height


def choose_value(
    rng: np.random.Generator,
    value: float | None,
    value_range: tuple[float, float],
) -> float:
    """从固定值或范围中选择一个增强参数。"""
    if value is not None:
        return float(value)
    low, high = value_range
    if low > high:
        raise ValueError("参数范围下界不能大于上界")
    return float(rng.uniform(low, high))


def choose_int_value(
    rng: np.random.Generator,
    value: int | None,
    value_range: tuple[int, int],
) -> int:
    """从固定整数或范围中选择一个增强参数。"""
    if value is not None:
        return int(value)
    low, high = value_range
    if low > high:
        raise ValueError("参数范围下界不能大于上界")
    return int(rng.integers(low, high + 1))
