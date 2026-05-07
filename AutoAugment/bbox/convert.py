from __future__ import annotations

import numpy as np


def yolo_to_xyxy(yolo_boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """将 YOLO 归一化 xywh bbox 转为 xyxy。"""
    boxes = np.asarray(yolo_boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x_center = boxes[:, 0] * width
    y_center = boxes[:, 1] * height
    box_width = boxes[:, 2] * width
    box_height = boxes[:, 3] * height
    x1 = x_center - box_width / 2.0
    y1 = y_center - box_height / 2.0
    x2 = x_center + box_width / 2.0
    y2 = y_center + box_height / 2.0
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def xyxy_to_yolo(bboxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """将 xyxy bbox 转为 YOLO 归一化 xywh。"""
    boxes = np.asarray(bboxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x_center = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / width
    y_center = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / height
    box_width = (boxes[:, 2] - boxes[:, 0]) / width
    box_height = (boxes[:, 3] - boxes[:, 1]) / height
    return np.stack([x_center, y_center, box_width, box_height], axis=1).astype(np.float32)


def coco_to_xyxy(coco_boxes: np.ndarray) -> np.ndarray:
    """将 COCO xywh bbox 转为 xyxy。"""
    boxes = np.asarray(coco_boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def xyxy_to_coco(bboxes: np.ndarray) -> np.ndarray:
    """将 xyxy bbox 转为 COCO xywh。"""
    boxes = np.asarray(bboxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x = boxes[:, 0]
    y = boxes[:, 1]
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    return np.stack([x, y, width, height], axis=1).astype(np.float32)


def voc_to_xyxy(voc_boxes: np.ndarray) -> np.ndarray:
    """将 VOC bbox 转为 xyxy。"""
    return np.asarray(voc_boxes, dtype=np.float32).reshape(-1, 4)


def xyxy_to_voc(bboxes: np.ndarray) -> np.ndarray:
    """将 xyxy bbox 转为 VOC bbox。"""
    return np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
