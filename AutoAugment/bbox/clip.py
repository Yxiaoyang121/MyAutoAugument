from __future__ import annotations

import numpy as np


def clip_bboxes(bboxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """将 xyxy bbox 裁剪到图像边界内。"""
    bboxes = np.asarray(bboxes, dtype=np.float32).copy()
    if bboxes.size == 0:
        return bboxes.reshape(0, 4)
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, width)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, width)
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, height)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, height)
    return bboxes


def valid_bbox_mask(
    bboxes: np.ndarray,
    min_width: float = 1.0,
    min_height: float = 1.0,
    min_area: float = 1.0,
) -> np.ndarray:
    """返回合法 bbox 掩码，要求宽高和面积均满足阈值。"""
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.size == 0:
        return np.zeros((0,), dtype=bool)
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    areas = widths * heights
    return (widths >= min_width) & (heights >= min_height) & (areas >= min_area)


def clip_filter_bboxes(
    bboxes: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
    min_width: float = 1.0,
    min_height: float = 1.0,
    min_area: float = 1.0,
    return_indices: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """裁剪 bbox 并同步过滤无效 bbox 与标签。"""
    clipped = clip_bboxes(bboxes, width, height)
    keep = valid_bbox_mask(clipped, min_width=min_width, min_height=min_height, min_area=min_area)
    filtered_bboxes = clipped[keep]
    filtered_labels = np.asarray(labels)[keep]
    if return_indices:
        return filtered_bboxes, filtered_labels, keep
    return filtered_bboxes, filtered_labels
