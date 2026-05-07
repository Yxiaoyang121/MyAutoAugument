from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from AutoAugment.formats.yolo import load_yolo_labels
from AutoAugment.types import SampleDict

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def project_root() -> Path:
    """返回项目根目录。"""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """解析项目根目录下的相对路径。"""
    path = Path(path)
    if path.is_absolute():
        raise ValueError("数据路径必须是相对项目根目录的相对路径")
    return project_root() / path


class YoloDetectionDataset:
    """YOLO 目标检测数据集读取器，返回统一 sample 结构。"""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path | None = None,
        transform: Callable[[SampleDict], SampleDict] | None = None,
        image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    ) -> None:
        self.images_dir = resolve_project_path(images_dir)
        self.labels_dir = resolve_project_path(labels_dir) if labels_dir is not None else self.images_dir
        self.transform = transform
        self.image_paths = sorted(
            path for path in self.images_dir.rglob("*") if path.suffix.lower() in image_extensions
        )
        if not self.image_paths:
            raise ValueError(f"未找到图像文件: {self.images_dir}")

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> SampleDict:
        """读取单个 YOLO 样本。"""
        image_path = self.image_paths[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        height, width = image.shape[:2]
        label_path = self._label_path(image_path)
        labels, bboxes = load_yolo_labels(label_path, width, height)
        sample: dict[str, Any] = {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "image_path": str(image_path.relative_to(project_root())),
            "label_path": str(label_path.relative_to(project_root())) if label_path.exists() else None,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _label_path(self, image_path: Path) -> Path:
        """根据图像路径推导 YOLO 标签路径。"""
        relative = image_path.relative_to(self.images_dir)
        return self.labels_dir / relative.with_suffix(".txt")
