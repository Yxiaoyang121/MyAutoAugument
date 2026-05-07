"""目标检测几何增强算子。"""

from AutoAugment.transforms.geometric.crop import RandomCrop
from AutoAugment.transforms.geometric.flip import HorizontalFlip, VerticalFlip
from AutoAugment.transforms.geometric.rotate import Rotate
from AutoAugment.transforms.geometric.scale import Scale
from AutoAugment.transforms.geometric.translate import Translate

__all__ = ["HorizontalFlip", "VerticalFlip", "Rotate", "Translate", "Scale", "RandomCrop"]
