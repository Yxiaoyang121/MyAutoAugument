from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search.feedback_loop import AutoAugmentOptimizer, OptimizationConfig
from src.utils.demo_data import create_demo_dataset


def parse_args() -> argparse.Namespace:
    """解析策略搜索示例参数。"""
    parser = argparse.ArgumentParser(description="运行自动增强策略搜索闭环")
    parser.add_argument("--generations", type=int, default=3, help="进化搜索迭代轮数")
    parser.add_argument("--population-size", type=int, default=6, help="策略种群规模")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def format_policy(result_policy) -> str:
    """将策略格式化为可读文本。"""
    return " -> ".join(
        f"{step.name}(p={step.probability:.2f}, strength={step.strength:.2f})"
        for step in result_policy.steps
    )


def main() -> None:
    """运行从策略到奖励再到策略更新的完整闭环。"""
    args = parse_args()
    dataset = create_demo_dataset(samples_per_class=18, seed=args.seed)
    train_dataset, validation_dataset = dataset.split(validation_fraction=0.25, seed=args.seed)
    config = OptimizationConfig(
        population_size=args.population_size,
        generations=args.generations,
        seed=args.seed,
        evaluator_repeats=1,
    )
    optimizer = AutoAugmentOptimizer(train_dataset, validation_dataset, config)
    result = optimizer.run()
    print("架构摘要: augmentations -> policies -> search -> models -> feedback loop")
    print(f"最佳验证准确率: {result.best_score:.4f}")
    print(f"最佳策略: {format_policy(result.best_policy)}")
    for item in result.history:
        print(
            f"第 {int(item['generation'])} 代: "
            f"最佳分数={item['best_score']:.4f}, 平均分数={item['mean_score']:.4f}"
        )


if __name__ == "__main__":
    main()
