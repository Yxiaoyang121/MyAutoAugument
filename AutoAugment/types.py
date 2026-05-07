from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


class DetectionSample(TypedDict):
    """目标检测样本，统一使用图像、边界框和标签。"""

    image: np.ndarray
    bboxes: np.ndarray
    labels: np.ndarray


SampleDict = dict[str, Any]
