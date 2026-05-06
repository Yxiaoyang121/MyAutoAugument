from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from src.augmentations.registry import list_augmentations
from src.policies.policy import Policy
from src.policies.sampler import crossover_policy, mutate_policy, sample_random_policy
from src.utils.random import get_rng


@dataclass(frozen=True)
class Candidate:
    """带评分的候选增强策略。"""

    policy: Policy
    score: float


@dataclass(frozen=True)
class SearchResult:
    """搜索完成后的最佳策略与历史记录。"""

    best_policy: Policy
    best_score: float
    history: tuple[dict[str, float], ...]
    candidates: tuple[Candidate, ...]


class EvolutionaryPolicySearch:
    """简单进化搜索，用验证集奖励驱动策略优化。"""

    def __init__(
        self,
        evaluator: Callable[[Policy], float],
        operations: Sequence[str] | None = None,
        population_size: int = 8,
        generations: int = 5,
        elite_size: int = 2,
        mutation_rate: float = 0.25,
        min_steps: int = 1,
        max_steps: int = 4,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if population_size <= 0:
            raise ValueError("种群规模必须大于 0")
        if generations <= 0:
            raise ValueError("迭代轮数必须大于 0")
        self.evaluator = evaluator
        self.operations = tuple(operations or list_augmentations())
        self.population_size = population_size
        self.generations = generations
        self.elite_size = max(1, min(elite_size, population_size))
        self.mutation_rate = mutation_rate
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.rng = get_rng(seed)

    def run(self, initial_policy: Policy | None = None) -> SearchResult:
        """执行策略搜索并返回最佳策略。"""
        population = self._initial_population(initial_policy)
        history: list[dict[str, float]] = []
        best_candidate: Candidate | None = None
        ranked: list[Candidate] = []
        for generation in range(self.generations):
            ranked = self._evaluate_population(population)
            generation_best = ranked[0]
            if best_candidate is None or generation_best.score > best_candidate.score:
                best_candidate = generation_best
            scores = np.asarray([candidate.score for candidate in ranked], dtype=np.float32)
            history.append(
                {
                    "generation": float(generation),
                    "best_score": float(generation_best.score),
                    "mean_score": float(scores.mean()),
                }
            )
            if generation < self.generations - 1:
                population = self._next_population(ranked)
        if best_candidate is None:
            raise RuntimeError("搜索未产生有效候选策略")
        return SearchResult(
            best_policy=best_candidate.policy,
            best_score=best_candidate.score,
            history=tuple(history),
            candidates=tuple(ranked),
        )

    def _initial_population(self, initial_policy: Policy | None) -> list[Policy]:
        """构造初始策略种群。"""
        population: list[Policy] = []
        if initial_policy is not None:
            population.append(initial_policy)
        while len(population) < self.population_size:
            population.append(
                sample_random_policy(
                    rng=self.rng,
                    operations=self.operations,
                    min_steps=self.min_steps,
                    max_steps=self.max_steps,
                )
            )
        return population

    def _evaluate_population(self, population: Sequence[Policy]) -> list[Candidate]:
        """计算并排序当前种群。"""
        candidates = [Candidate(policy=policy, score=float(self.evaluator(policy))) for policy in population]
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def _next_population(self, ranked: Sequence[Candidate]) -> list[Policy]:
        """根据精英保留、交叉和变异生成下一代。"""
        elites = [candidate.policy for candidate in ranked[: self.elite_size]]
        pool = [candidate.policy for candidate in ranked[: max(self.elite_size, len(ranked) // 2)]]
        next_population = list(elites)
        while len(next_population) < self.population_size:
            parent_indices = self.rng.integers(0, len(pool), size=2)
            child = crossover_policy(pool[int(parent_indices[0])], pool[int(parent_indices[1])], rng=self.rng, max_steps=self.max_steps)
            child = mutate_policy(
                child,
                rng=self.rng,
                operations=self.operations,
                mutation_rate=self.mutation_rate,
                max_steps=self.max_steps,
            )
            next_population.append(child)
        return next_population
