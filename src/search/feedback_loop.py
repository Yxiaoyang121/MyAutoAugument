from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from src.augmentations.registry import list_augmentations
from src.models.lightweight import LightweightPolicyEvaluator
from src.policies.policy import Policy
from src.search.evolutionary import EvolutionaryPolicySearch, SearchResult
from src.utils.dataset import ImageClassificationDataset


@dataclass(frozen=True)
class OptimizationConfig:
    """自动增强策略搜索配置。"""

    population_size: int = 8
    generations: int = 5
    elite_size: int = 2
    mutation_rate: float = 0.25
    min_steps: int = 1
    max_steps: int = 4
    seed: int = 42
    evaluator_repeats: int = 1
    operations: tuple[str, ...] | None = None


class AutoAugmentOptimizer:
    """工业图像自动增强闭环优化器。"""

    def __init__(
        self,
        train_dataset: ImageClassificationDataset,
        validation_dataset: ImageClassificationDataset,
        config: OptimizationConfig | None = None,
    ) -> None:
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = config or OptimizationConfig()

    def run(self, initial_policy: Policy | None = None) -> SearchResult:
        """执行 policy 到 score 再到 policy 更新的完整闭环。"""
        evaluator = LightweightPolicyEvaluator(
            train_dataset=self.train_dataset,
            validation_dataset=self.validation_dataset,
            seed=self.config.seed,
            repeats=self.config.evaluator_repeats,
        )
        operations: Sequence[str] = self.config.operations or list_augmentations()
        search = EvolutionaryPolicySearch(
            evaluator=evaluator.evaluate,
            operations=operations,
            population_size=self.config.population_size,
            generations=self.config.generations,
            elite_size=self.config.elite_size,
            mutation_rate=self.config.mutation_rate,
            min_steps=self.config.min_steps,
            max_steps=self.config.max_steps,
            seed=np.random.default_rng(self.config.seed),
        )
        return search.run(initial_policy=initial_policy)
