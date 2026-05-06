from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.augmentations.registry import list_augmentations
from src.policies.policy import AugmentationStep, Policy
from src.utils.random import get_rng


def sample_random_policy(
    rng: int | np.random.Generator | None = None,
    operations: Sequence[str] | None = None,
    min_steps: int = 1,
    max_steps: int = 4,
    probability_range: tuple[float, float] = (0.2, 0.9),
    strength_range: tuple[float, float] = (0.1, 1.0),
) -> Policy:
    """随机采样增强策略。"""
    generator = get_rng(rng)
    operation_names = tuple(operations or list_augmentations())
    if not operation_names:
        raise ValueError("增强算子列表不能为空")
    if min_steps <= 0 or max_steps < min_steps:
        raise ValueError("策略步数范围不合法")
    step_count = int(generator.integers(min_steps, max_steps + 1))
    replace = step_count > len(operation_names)
    selected = generator.choice(operation_names, size=step_count, replace=replace)
    steps = [
        AugmentationStep(
            name=str(name),
            probability=float(generator.uniform(*probability_range)),
            strength=float(generator.uniform(*strength_range)),
        )
        for name in selected
    ]
    return Policy(steps)


def mutate_policy(
    policy: Policy,
    rng: int | np.random.Generator | None = None,
    operations: Sequence[str] | None = None,
    mutation_rate: float = 0.25,
    max_steps: int = 4,
) -> Policy:
    """对策略执行概率、强度、算子替换和步数变异。"""
    generator = get_rng(rng)
    operation_names = tuple(operations or list_augmentations())
    steps: list[AugmentationStep] = []
    for step in policy.steps:
        name = step.name
        probability = step.probability
        strength = step.strength
        params = dict(step.params)
        if generator.random() < mutation_rate:
            name = str(generator.choice(operation_names))
        if generator.random() < mutation_rate:
            probability = float(np.clip(probability + generator.normal(0.0, 0.15), 0.05, 1.0))
        if generator.random() < mutation_rate:
            strength = float(np.clip(strength + generator.normal(0.0, 0.18), 0.0, 1.0))
        if generator.random() >= mutation_rate or len(policy.steps) == 1:
            steps.append(AugmentationStep(name=name, probability=probability, strength=strength, params=params))
    if len(steps) < max_steps and generator.random() < mutation_rate:
        steps.append(
            AugmentationStep(
                name=str(generator.choice(operation_names)),
                probability=float(generator.uniform(0.2, 0.9)),
                strength=float(generator.uniform(0.1, 1.0)),
            )
        )
    if not steps:
        steps.append(
            AugmentationStep(
                name=str(generator.choice(operation_names)),
                probability=float(generator.uniform(0.2, 0.9)),
                strength=float(generator.uniform(0.1, 1.0)),
            )
        )
    return Policy(steps[:max_steps])


def crossover_policy(
    parent_a: Policy,
    parent_b: Policy,
    rng: int | np.random.Generator | None = None,
    max_steps: int = 4,
) -> Policy:
    """从两个父策略中交叉生成子策略。"""
    generator = get_rng(rng)
    combined = list(parent_a.steps) + list(parent_b.steps)
    if not combined:
        return sample_random_policy(generator, max_steps=max_steps)
    generator.shuffle(combined)
    child_size = int(generator.integers(1, min(max_steps, len(combined)) + 1))
    return Policy(combined[:child_size])
