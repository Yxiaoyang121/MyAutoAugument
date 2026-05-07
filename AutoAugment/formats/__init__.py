"""目标检测标注格式转换入口。"""

from AutoAugment.formats.coco import annotations_to_sample, sample_to_coco_annotations
from AutoAugment.formats.yolo import load_yolo_labels, save_yolo_labels

__all__ = ["load_yolo_labels", "save_yolo_labels", "annotations_to_sample", "sample_to_coco_annotations"]
