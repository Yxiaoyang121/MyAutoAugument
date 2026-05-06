from __future__ import annotations

import numpy as np

from src.augmentations.registry import apply_augmentation, list_augmentations
from src.policies.policy import AugmentationStep, Policy
from src.search.feedback_loop import AutoAugmentOptimizer, OptimizationConfig
from src.utils.demo_data import create_demo_dataset, create_demo_image


def test_all_augmentations_keep_image_contract() -> None:
    """验证所有标准增强算子保持图像契约。"""
    image = create_demo_image(seed=1)
    rng = np.random.default_rng(7)
    for name in list_augmentations():
        output = apply_augmentation(name, image, {"strength": 0.6, "rng": rng})
        assert output.shape == image.shape
        assert output.dtype == np.uint8


def test_policy_application_is_reproducible() -> None:
    """验证固定种子下策略输出可复现。"""
    image = create_demo_image(seed=2)
    policy = Policy(
        [
            AugmentationStep("brightness", probability=1.0, strength=0.7),
            AugmentationStep("gaussian_noise", probability=1.0, strength=0.5),
            AugmentationStep("rotation", probability=1.0, strength=0.4),
        ]
    )
    first = policy.apply(image, rng=123)
    second = policy.apply(image, rng=123)
    assert np.array_equal(first, second)


def test_feedback_loop_runs_end_to_end() -> None:
    """验证策略搜索闭环可以端到端运行。"""
    dataset = create_demo_dataset(samples_per_class=5, seed=3)
    train_dataset, validation_dataset = dataset.split(validation_fraction=0.3, seed=3)
    config = OptimizationConfig(population_size=3, generations=2, elite_size=1, max_steps=2, seed=3)
    result = AutoAugmentOptimizer(train_dataset, validation_dataset, config).run()
    assert 0.0 <= result.best_score <= 1.0
    assert len(result.history) == 2
    assert result.best_policy.steps
