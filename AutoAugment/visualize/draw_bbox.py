from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from AutoAugment.types import SampleDict


def draw_bboxes(
    sample: SampleDict,
    label_names: Mapping[int, str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """在图像副本上绘制 bbox 和标签。"""
    image = sample["image"].copy()
    labels = np.asarray(sample["labels"])
    for bbox, label in zip(sample["bboxes"], labels):
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if label_names is not None:
            text = label_names.get(int(label), str(int(label)))
            cv2.putText(image, text, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image
