from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.policies import Policy, apply_policy_to_sample
from AutoAugment.utils import (
    find_yolo_records,
    flatten_relative_stem,
    load_yolo_sample,
    resolve_output_dir,
    save_yolo_sample,
    write_apply_policy_readme,
    write_dataset_path_file,
    write_json_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a saved augmentation policy to a YOLO detection dataset.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root. Supports absolute or relative paths.")
    parser.add_argument("--policy", required=True, help="Path to policy json/yaml, usually best_policy.json.")
    parser.add_argument("--output", default=None, help="Output dataset directory. Defaults to outputs/runs/apply_policy/run_YYYYMMDD_HHMMSS.")
    parser.add_argument("--copies", type=int, default=3, help="Augmented copies to create per source image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--missing-label", choices=["empty", "skip", "error"], default="empty", help="How to handle missing txt labels.")
    parser.add_argument("--run-name", default=None, help="Custom default run directory name when --output is omitted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    policy_path = Path(args.policy)
    output = resolve_output_dir(args.output, "apply_policy", run_name=args.run_name)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    if args.copies <= 0:
        raise ValueError("--copies must be positive")

    output.mkdir(parents=True, exist_ok=True)
    policy = Policy.load(policy_path)
    policy.save(output / "applied_policy.json")
    write_dataset_path_file(output / "source_dataset.txt", supplied_path=args.dataset, resolved_path=dataset)
    write_json_file(
        output / "run_config.json",
        {
            "script": "examples/apply_policy_to_dataset.py",
            "args": vars(args),
            "source_dataset": str(dataset.resolve()),
            "policy": str(policy_path.resolve()),
            "output_dir": str(output.resolve()),
            "images_dir": str((output / "images").resolve()),
            "labels_dir": str((output / "labels").resolve()),
        },
    )
    write_apply_policy_readme(output, dataset_path=dataset, policy_path=policy_path)

    records = find_yolo_records(dataset, missing_label=args.missing_label)
    rng = np.random.default_rng(args.seed)
    total = 0
    for record in records:
        sample = load_yolo_sample(record)
        stem = flatten_relative_stem(record.relative_path)
        for copy_index in range(args.copies):
            augmented = apply_policy_to_sample(sample, policy, rng=rng)
            output_stem = f"{stem}_aug_{copy_index:03d}"
            image_path = output / "images" / f"{output_stem}{record.image_path.suffix.lower()}"
            label_path = output / "labels" / f"{output_stem}.txt"
            save_yolo_sample(augmented, image_path, label_path)
            total += 1
    write_apply_policy_readme(output, dataset_path=dataset, policy_path=policy_path)
    print("批量增强完成：")
    print(f"- 输入数据集：{dataset.resolve()}")
    print(f"- 使用策略：{policy_path.resolve()}")
    print(f"- 增强输出目录：{output.resolve()}")
    print(f"- 增强图片目录：{(output / 'images').resolve()}")
    print(f"- 增强标签目录：{(output / 'labels').resolve()}")
    print(f"- 说明文件：{(output / 'README.txt').resolve()}")
    print(f"- 增强样本数：{total}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
