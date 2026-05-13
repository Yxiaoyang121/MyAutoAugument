"""工业视觉目标检测数据增强框架。"""

from AutoAugment.pipelines.compose import Compose
from AutoAugment.policies import OperationSpec, Policy, SearchSpace, apply_policy, default_detection_search_space
from AutoAugment.search import (
    CommandEvaluator,
    DiagnosticPolicyUpdater,
    ProxyEvaluator,
    RandomSearch,
    YoloCommandEvaluator,
    YoloTrainValEvaluator,
)
from AutoAugment.transforms.geometric import HorizontalFlip, RandomCrop, Rotate, Scale, Translate, VerticalFlip

__all__ = [
    "Compose",
    "OperationSpec",
    "Policy",
    "SearchSpace",
    "apply_policy",
    "default_detection_search_space",
    "ProxyEvaluator",
    "CommandEvaluator",
    "YoloCommandEvaluator",
    "YoloTrainValEvaluator",
    "DiagnosticPolicyUpdater",
    "RandomSearch",
    "HorizontalFlip",
    "VerticalFlip",
    "Rotate",
    "Translate",
    "Scale",
    "RandomCrop",
]
