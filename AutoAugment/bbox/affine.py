from __future__ import annotations

import numpy as np


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """使用 2x3 仿射矩阵变换点集。"""
    points = np.asarray(points, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (2, 3):
        raise ValueError("仿射矩阵必须是 2x3")
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=1)
    return homogeneous @ matrix.T


def bbox_to_corners(bboxes: np.ndarray) -> np.ndarray:
    """将 Nx4 bbox 转为 Nx4x2 四角点。"""
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.size == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    return np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    )


def corners_to_bbox(corners: np.ndarray) -> np.ndarray:
    """将 Nx4x2 四角点重新包围为 xyxy bbox。"""
    corners = np.asarray(corners, dtype=np.float32)
    if corners.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    min_xy = corners.min(axis=1)
    max_xy = corners.max(axis=1)
    return np.concatenate([min_xy, max_xy], axis=1).astype(np.float32)


def apply_affine_to_bboxes(bboxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """对 bbox 四个角点做仿射变换并重新计算外接矩形。"""
    corners = bbox_to_corners(bboxes)
    if len(corners) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    flat = corners.reshape(-1, 2)
    transformed = transform_points(flat, matrix).reshape(-1, 4, 2)
    return corners_to_bbox(transformed)
