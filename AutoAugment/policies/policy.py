from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.augmentations import apply_augmentation


@dataclass
class OperationSpec:
    name: str
    prob: float = 1.0
    strength: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip().lower()
        self.prob = float(np.clip(self.prob, 0.0, 1.0))
        self.strength = float(np.clip(self.strength, 0.0, 1.0))
        self.params = dict(self.params or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prob": self.prob,
            "strength": self.strength,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperationSpec":
        return cls(
            name=data["name"],
            prob=data.get("prob", 1.0),
            strength=data.get("strength", 1.0),
            params=data.get("params", {}) or {},
        )


@dataclass
class Policy:
    name: str
    operations: list[OperationSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "operations": [operation.to_dict() for operation in self.operations],
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        return cls(
            name=data.get("name", "policy"),
            operations=[OperationSpec.from_dict(item) for item in data.get("operations", [])],
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Policy":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to save YAML policies. Use .json instead.") from exc
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
            return
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to load YAML policies. Use .json instead.") from exc
            return cls.from_dict(yaml.safe_load(text))
        return cls.from_json(text)


def ensure_policy(policy: Policy | dict[str, Any]) -> Policy:
    if isinstance(policy, Policy):
        return policy
    return Policy.from_dict(policy)


def apply_policy(
    image: np.ndarray,
    labels: np.ndarray | None,
    bboxes: np.ndarray | None,
    policy: Policy | dict[str, Any],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = rng if rng is not None else np.random.default_rng()
    current_image = image.copy()
    current_labels = np.asarray(labels if labels is not None else [], dtype=np.int64).copy()
    current_bboxes = np.asarray(bboxes if bboxes is not None else np.zeros((0, 4)), dtype=np.float32).reshape(-1, 4).copy()
    resolved = ensure_policy(policy)
    for operation in resolved.operations:
        if generator.random() > operation.prob:
            continue
        current_image, current_labels, current_bboxes = apply_augmentation(
            operation.name,
            current_image,
            current_labels,
            current_bboxes,
            params=operation.params,
            strength=operation.strength,
            rng=generator,
        )
    return current_image, current_labels, current_bboxes


def apply_policy_to_sample(
    sample: dict[str, Any],
    policy: Policy | dict[str, Any],
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    image, labels, bboxes = apply_policy(sample["image"], sample["labels"], sample["bboxes"], policy, rng=rng)
    result = dict(sample)
    result["image"] = image
    result["labels"] = labels
    result["bboxes"] = bboxes
    return result
