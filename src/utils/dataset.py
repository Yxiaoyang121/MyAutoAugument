from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.policies.policy import Policy
from src.utils.paths import resolve_project_path
from src.utils.random import get_rng


@dataclass(frozen=True)
class ImageClassificationDataset:
    """面向分类任务的轻量图像数据集。"""

    images: tuple[np.ndarray, ...]
    labels: np.ndarray

    def __post_init__(self) -> None:
        if len(self.images) != len(self.labels):
            raise ValueError("图像数量必须与标签数量一致")
        if len(self.images) == 0:
            raise ValueError("数据集不能为空")

    def split(
        self,
        validation_fraction: float = 0.2,
        seed: int | np.random.Generator | None = None,
    ) -> tuple["ImageClassificationDataset", "ImageClassificationDataset"]:
        """按固定随机种子划分训练集和验证集。"""
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("验证集比例必须位于 0 到 1 之间")
        rng = get_rng(seed)
        indices = np.arange(len(self.images))
        rng.shuffle(indices)
        validation_size = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices = indices[:validation_size]
        train_indices = indices[validation_size:]
        if len(train_indices) == 0:
            raise ValueError("训练集不能为空，请降低验证集比例或增加样本")
        return self.take(train_indices), self.take(validation_indices)

    def take(self, indices: Sequence[int]) -> "ImageClassificationDataset":
        """按索引抽取子集。"""
        selected_images = tuple(self.images[int(index)] for index in indices)
        selected_labels = self.labels[np.asarray(indices, dtype=int)]
        return ImageClassificationDataset(selected_images, selected_labels)

    def augment(self, policy: Policy, seed: int | np.random.Generator | None = None) -> "ImageClassificationDataset":
        """使用增强策略生成新的数据集。"""
        rng = get_rng(seed)
        images = tuple(policy.apply(image, rng=rng) for image in self.images)
        return ImageClassificationDataset(images, self.labels.copy())


def load_image_folder(root: str | Path, image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> ImageClassificationDataset:
    """从按类别分目录的数据集中读取图像。"""
    root_path = resolve_project_path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root_path}")
    images: list[np.ndarray] = []
    labels: list[int] = []
    class_dirs = sorted(path for path in root_path.iterdir() if path.is_dir())
    for label, class_dir in enumerate(class_dirs):
        image_paths = sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in image_extensions)
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            images.append(image)
            labels.append(label)
    if not images:
        raise ValueError("未读取到任何有效图像")
    return ImageClassificationDataset(tuple(images), np.asarray(labels, dtype=np.int64))
