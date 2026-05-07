from __future__ import annotations

import numpy as np


def bbox_iou(boxes_a: np.ndarray, boxes_b: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """计算两组 xyxy bbox 的两两 IoU。"""
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return (intersection / np.maximum(union, eps)).astype(np.float32)
