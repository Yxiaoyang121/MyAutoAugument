from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.augmentations import list_augmentations
from AutoAugment.policies.policy import OperationSpec, Policy


@dataclass
class OperationSpace:
    name: str
    prob_range: tuple[float, float] = (0.3, 0.9)
    strength_range: tuple[float, float] = (0.1, 0.8)
    params: dict[str, Any] = field(default_factory=dict)

    def sample(self, rng: np.random.Generator) -> OperationSpec:
        prob = float(rng.uniform(*self.prob_range))
        strength = float(rng.uniform(*self.strength_range))
        return OperationSpec(name=self.name, prob=prob, strength=strength, params=dict(self.params))


@dataclass
class SearchSpace:
    operations: list[OperationSpace]
    operation_count_range: tuple[int, int] = (2, 4)
    allow_repeated_operations: bool = False
    name_prefix: str = "policy"
    operation_weights: dict[str, float] = field(default_factory=dict)
    forbidden_combinations: list[tuple[str, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.operations:
            raise ValueError("search space must contain at least one operation")
        low, high = self.operation_count_range
        if low <= 0 or high < low:
            raise ValueError("operation_count_range must be positive and ordered")
        registered = set(list_augmentations())
        missing = [operation.name for operation in self.operations if operation.name not in registered]
        if missing:
            raise ValueError(f"operations are not registered: {missing}")
        operation_names = {operation.name for operation in self.operations}
        unknown_weights = [name for name in self.operation_weights if name not in operation_names]
        if unknown_weights:
            raise ValueError(f"operation_weights contain unknown operations: {unknown_weights}")
        self.operation_weights = {name: float(weight) for name, weight in self.operation_weights.items() if float(weight) > 0}
        self.forbidden_combinations = [
            tuple(str(name).strip().lower() for name in combination if str(name).strip())
            for combination in self.forbidden_combinations
            if combination
        ]

    def sample_policy(self, rng: np.random.Generator | None = None, name: str | None = None) -> Policy:
        generator = rng if rng is not None else np.random.default_rng()
        low, high = self.operation_count_range
        count = int(generator.integers(low, high + 1))
        indices = self._sample_indices(generator, count)
        for _ in range(100):
            operations = [self.operations[int(index)].sample(generator) for index in indices]
            if not self._has_forbidden_combination([operation.name for operation in operations]):
                break
            indices = self._sample_indices(generator, count)
        else:
            raise RuntimeError("could not sample a policy without forbidden operation combinations")
        policy_name = name or f"{self.name_prefix}_{int(generator.integers(0, 1_000_000)):06d}"
        return Policy(name=policy_name, operations=operations)

    def _sample_indices(self, generator: np.random.Generator, count: int) -> np.ndarray:
        probabilities = self._sampling_probabilities()
        if not self.allow_repeated_operations:
            count = min(count, len(self.operations))
            return generator.choice(len(self.operations), size=count, replace=False, p=probabilities)
        return generator.choice(len(self.operations), size=count, replace=True, p=probabilities)

    def _sampling_probabilities(self) -> np.ndarray | None:
        if not self.operation_weights:
            return None
        weights = np.asarray([self.operation_weights.get(operation.name, 1.0) for operation in self.operations], dtype=np.float64)
        weights = np.maximum(weights, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            return None
        return weights / total

    def _has_forbidden_combination(self, operation_names: list[str]) -> bool:
        selected = set(operation_names)
        return any(set(combination).issubset(selected) for combination in self.forbidden_combinations)

    @classmethod
    def from_advisor_json(
        cls,
        path: str | Path,
        *,
        operation_count_range: tuple[int, int] | None = None,
        allow_repeated_operations: bool = False,
    ) -> "SearchSpace":
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        base = default_detection_search_space(
            operation_count_range=operation_count_range or (2, int(config.get("max_ops_per_policy", 4))),
            allow_repeated_operations=allow_repeated_operations,
        )
        base_by_name = {operation.name: operation for operation in base.operations}
        allowed = config.get("allowed_operations") or list(base_by_name)
        strength_ranges = config.get("strength_ranges") or {}
        prob_ranges = config.get("prob_ranges") or {}
        operations: list[OperationSpace] = []
        for raw_name in allowed:
            name = str(raw_name).strip().lower()
            if name not in base_by_name:
                raise ValueError(f"advisor search space contains unknown operation: {name}")
            template = base_by_name[name]
            operations.append(
                OperationSpace(
                    name=name,
                    prob_range=_range_from_config(prob_ranges.get(name), template.prob_range, "prob_ranges", name),
                    strength_range=_range_from_config(strength_ranges.get(name), template.strength_range, "strength_ranges", name),
                    params=dict(template.params),
                )
            )

        low, high = operation_count_range or base.operation_count_range
        if "max_ops_per_policy" in config:
            high = min(high, int(config["max_ops_per_policy"]))
            low = min(low, high)
        return cls(
            operations=operations,
            operation_count_range=(low, high),
            allow_repeated_operations=allow_repeated_operations,
            name_prefix=str(config.get("name_prefix", "advisor_policy")),
            operation_weights={str(key).strip().lower(): float(value) for key, value in (config.get("operation_weights") or {}).items()},
            forbidden_combinations=[
                tuple(str(item).strip().lower() for item in combination)
                for combination in (config.get("forbidden_combinations") or [])
            ],
        )


def load_search_space_from_json(
    path: str | Path,
    *,
    operation_count_range: tuple[int, int] | None = None,
    allow_repeated_operations: bool = False,
) -> SearchSpace:
    return SearchSpace.from_advisor_json(
        path,
        operation_count_range=operation_count_range,
        allow_repeated_operations=allow_repeated_operations,
    )


def _range_from_config(value: Any, default: tuple[float, float], field_name: str, operation_name: str) -> tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name}.{operation_name} must be a two-value range")
    low = float(value[0])
    high = float(value[1])
    if low < 0.0 or high < low:
        raise ValueError(f"{field_name}.{operation_name} must be non-negative and ordered")
    return (low, high)


def default_detection_search_space(
    *,
    operation_count_range: tuple[int, int] = (2, 4),
    allow_repeated_operations: bool = False,
) -> SearchSpace:
    operations = [
        OperationSpace("brightness", (0.3, 0.8), (0.1, 0.5), {"max_delta": 0.2}),
        OperationSpace("contrast", (0.3, 0.8), (0.1, 0.5), {"max_delta": 0.35}),
        OperationSpace("gamma", (0.3, 0.8), (0.1, 0.5), {"min_gamma": 0.75, "max_gamma": 1.35}),
        OperationSpace("gaussian_noise", (0.2, 0.7), (0.05, 0.35), {"max_std": 0.05}),
        OperationSpace("gaussian_blur", (0.2, 0.6), (0.05, 0.35), {"max_kernel": 5}),
        OperationSpace("motion_blur", (0.1, 0.5), (0.05, 0.3), {"max_kernel": 7}),
        OperationSpace("sharpen", (0.2, 0.6), (0.05, 0.35), {"amount": 0.8}),
        OperationSpace("clahe", (0.2, 0.6), (0.05, 0.4), {"max_clip_limit": 3.0}),
        OperationSpace("cutout", (0.1, 0.5), (0.05, 0.25), {"max_holes": 2, "max_fraction": 0.18}),
        OperationSpace("horizontal_flip", (0.3, 0.7), (1.0, 1.0), {}),
        OperationSpace("translate", (0.2, 0.7), (0.05, 0.4), {"max_translate": 0.08}),
        OperationSpace("scale", (0.2, 0.7), (0.05, 0.35), {"max_delta": 0.18}),
        OperationSpace("rotate", (0.2, 0.6), (0.05, 0.4), {"max_angle": 10.0}),
        OperationSpace(
            "affine",
            (0.1, 0.5),
            (0.05, 0.3),
            {"max_angle": 8.0, "max_translate": 0.06, "max_scale_delta": 0.12, "max_shear": 3.0},
        ),
    ]
    return SearchSpace(
        operations=operations,
        operation_count_range=operation_count_range,
        allow_repeated_operations=allow_repeated_operations,
    )
