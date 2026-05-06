from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from src.augmentations.registry import apply_augmentation
from src.utils.random import get_rng


def _clip01(value: float) -> float:
    """将概率或强度裁剪到闭区间。"""
    return float(np.clip(value, 0.0, 1.0))


@dataclass(frozen=True)
class AugmentationStep:
    """单个增强步骤，包含算子名称、执行概率、强度和附加参数。"""

    name: str
    probability: float = 1.0
    strength: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "AugmentationStep":
        """返回概率和强度已归一化的步骤。"""
        return AugmentationStep(
            name=self.name,
            probability=_clip01(self.probability),
            strength=_clip01(self.strength),
            params=dict(self.params),
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为可保存的字典。"""
        step = self.normalized()
        return {
            "name": step.name,
            "probability": step.probability,
            "strength": step.strength,
            "params": dict(step.params),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AugmentationStep":
        """从字典恢复增强步骤。"""
        return AugmentationStep(
            name=str(data["name"]),
            probability=float(data.get("probability", 1.0)),
            strength=float(data.get("strength", 0.5)),
            params=dict(data.get("params", {})),
        ).normalized()


@dataclass(frozen=True)
class Policy:
    """增强策略，由多个增强步骤顺序组合而成。"""

    steps: tuple[AugmentationStep, ...]

    def __init__(self, steps: Iterable[AugmentationStep]):
        normalized_steps = tuple(step.normalized() for step in steps)
        object.__setattr__(self, "steps", normalized_steps)

    def apply(self, image: np.ndarray, rng: int | np.random.Generator | None = None) -> np.ndarray:
        """按策略顺序对单张图像执行增强。"""
        generator = get_rng(rng)
        result = image.copy()
        for step in self.steps:
            if generator.random() > step.probability:
                continue
            params = dict(step.params)
            params["strength"] = step.strength
            params["rng"] = generator
            result = apply_augmentation(step.name, result, params)
        return result

    def apply_batch(
        self,
        images: Iterable[np.ndarray],
        rng: int | np.random.Generator | None = None,
    ) -> tuple[np.ndarray, ...]:
        """对一批图像执行同一个策略。"""
        generator = get_rng(rng)
        return tuple(self.apply(image, rng=generator) for image in images)

    def to_dict(self) -> dict[str, Any]:
        """序列化策略。"""
        return {"steps": [step.to_dict() for step in self.steps]}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Policy":
        """从字典恢复策略。"""
        return Policy(AugmentationStep.from_dict(step) for step in data.get("steps", []))

    def names(self) -> tuple[str, ...]:
        """返回策略中的算子名称。"""
        return tuple(step.name for step in self.steps)
