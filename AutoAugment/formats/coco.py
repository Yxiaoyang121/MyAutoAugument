from __future__ import annotations

from typing import Any

import numpy as np

from AutoAugment.bbox.convert import coco_to_xyxy, xyxy_to_coco


def annotations_to_sample(image: np.ndarray, annotations: list[dict[str, Any]]) -> dict[str, Any]:
    """将 COCO annotation 列表转换为检测 sample。"""
    labels = []
    coco_boxes = []
    for annotation in annotations:
        if annotation.get("iscrowd", 0):
            continue
        labels.append(int(annotation["category_id"]))
        coco_boxes.append(annotation["bbox"])
    if not coco_boxes:
        bboxes = np.zeros((0, 4), dtype=np.float32)
        label_array = np.zeros((0,), dtype=np.int64)
    else:
        bboxes = coco_to_xyxy(np.asarray(coco_boxes, dtype=np.float32))
        label_array = np.asarray(labels, dtype=np.int64)
    return {"image": image, "bboxes": bboxes, "labels": label_array}


def sample_to_coco_annotations(sample: dict[str, Any], image_id: int, start_annotation_id: int = 1) -> list[dict[str, Any]]:
    """将检测 sample 转换为 COCO annotation 列表。"""
    coco_boxes = xyxy_to_coco(sample["bboxes"])
    annotations = []
    for offset, (label, box) in enumerate(zip(sample["labels"], coco_boxes)):
        width = float(max(0.0, box[2]))
        height = float(max(0.0, box[3]))
        annotations.append(
            {
                "id": start_annotation_id + offset,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [float(box[0]), float(box[1]), width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
    return annotations
