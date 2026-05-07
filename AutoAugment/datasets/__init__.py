"""目标检测数据集读取入口。"""

from AutoAugment.datasets.coco_dataset import CocoDetectionDataset
from AutoAugment.datasets.yolo_dataset import YoloDetectionDataset

__all__ = ["YoloDetectionDataset", "CocoDetectionDataset"]
