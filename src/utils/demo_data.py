from __future__ import annotations

import cv2
import numpy as np

from src.utils.dataset import ImageClassificationDataset
from src.utils.random import get_rng


def create_demo_image(size: tuple[int, int] = (128, 128), seed: int | np.random.Generator | None = None) -> np.ndarray:
    """生成用于快速验证增强效果的工业风格示例图。"""
    rng = get_rng(seed)
    height, width = size
    base = np.full((height, width, 3), 42, dtype=np.uint8)
    gradient = np.linspace(0, 70, width, dtype=np.uint8)
    base += gradient.reshape(1, width, 1)
    center = (width // 2 + int(rng.integers(-8, 9)), height // 2 + int(rng.integers(-8, 9)))
    cv2.circle(base, center, min(height, width) // 5, (180, 180, 170), -1)
    cv2.rectangle(base, (width // 5, height // 5), (width // 5 * 2, height // 5 * 2), (90, 130, 180), -1)
    noise = rng.normal(0, 4, base.shape)
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def create_demo_dataset(
    samples_per_class: int = 16,
    image_size: tuple[int, int] = (96, 96),
    seed: int | np.random.Generator | None = None,
) -> ImageClassificationDataset:
    """生成可用于闭环搜索冒烟测试的合成分类数据集。"""
    rng = get_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    height, width = image_size
    for class_id in range(2):
        for _ in range(samples_per_class):
            image = np.full((height, width, 3), 38 + class_id * 20, dtype=np.uint8)
            if class_id == 0:
                center = (width // 2 + int(rng.integers(-6, 7)), height // 2 + int(rng.integers(-6, 7)))
                cv2.circle(image, center, width // 5, (185, 185, 175), -1)
            else:
                offset = int(rng.integers(-5, 6))
                cv2.rectangle(image, (width // 3 + offset, height // 3), (width // 3 * 2 + offset, height // 3 * 2), (95, 150, 210), -1)
            noise = rng.normal(0, 8, image.shape)
            images.append(np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8))
            labels.append(class_id)
    return ImageClassificationDataset(tuple(images), np.asarray(labels, dtype=np.int64))
