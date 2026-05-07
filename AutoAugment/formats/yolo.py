from __future__ import annotations

from pathlib import Path

import numpy as np

from AutoAugment.bbox.convert import xyxy_to_yolo, yolo_to_xyxy


def load_yolo_labels(label_path: str | Path, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    """读取 YOLO txt 标注并转换为 labels 与 xyxy bbox。"""
    path = Path(label_path)
    if not path.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    labels: list[int] = []
    yolo_boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"YOLO 标注行至少需要 5 列: {line}")
        labels.append(int(float(parts[0])))
        yolo_boxes.append([float(value) for value in parts[1:5]])
    if not yolo_boxes:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    bboxes = yolo_to_xyxy(np.asarray(yolo_boxes, dtype=np.float32), image_width, image_height)
    return np.asarray(labels, dtype=np.int64), bboxes


def save_yolo_labels(label_path: str | Path, labels: np.ndarray, bboxes: np.ndarray, image_width: int, image_height: int) -> None:
    """将 labels 与 xyxy bbox 保存为 YOLO txt 标注。"""
    path = Path(label_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    yolo_boxes = xyxy_to_yolo(bboxes, image_width, image_height)
    lines = []
    for label, box in zip(labels, yolo_boxes):
        lines.append(f"{int(label)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")
