from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from AutoAugment.datasets.yolo_dataset import resolve_project_path
from AutoAugment.formats.coco import annotations_to_sample
from AutoAugment.types import SampleDict


class CocoDetectionDataset:
    """COCO 目标检测数据集读取器，返回统一 sample 结构。"""

    def __init__(
        self,
        annotation_file: str | Path,
        images_dir: str | Path,
        transform: Callable[[SampleDict], SampleDict] | None = None,
    ) -> None:
        self.annotation_file = resolve_project_path(annotation_file)
        self.images_dir = resolve_project_path(images_dir)
        self.transform = transform
        with self.annotation_file.open("r", encoding="utf-8") as file:
            self.coco = json.load(file)
        self.images = sorted(self.coco.get("images", []), key=lambda item: item["id"])
        self.annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.coco.get("annotations", []):
            self.annotations_by_image_id[int(annotation["image_id"])].append(annotation)
        if not self.images:
            raise ValueError("COCO 标注文件中没有 images")

    def __len__(self) -> int:
        """返回样本数量。"""
        return len(self.images)

    def __getitem__(self, index: int) -> SampleDict:
        """读取单个 COCO 样本。"""
        image_info = self.images[index]
        image_path = self.images_dir / image_info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        annotations = self.annotations_by_image_id.get(int(image_info["id"]), [])
        sample = annotations_to_sample(image, annotations)
        sample["image_id"] = int(image_info["id"])
        sample["image_path"] = str(image_path.relative_to(resolve_project_path(".")))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
