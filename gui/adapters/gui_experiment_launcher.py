from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.policies import OperationSpace, Policy, SearchSpace, default_detection_search_space
from AutoAugment.search import ProxyEvaluator, RandomSearch, TrialResult, YoloTrainValEvaluator
from AutoAugment.utils import copy_yolo_records, resolve_yolo_train_val_records, write_json_file, write_policy_search_readme
from gui.adapters.dataset_adapter import DatasetAdapter
from gui.models.experiment_config import (
    ExperimentConfig,
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_FIXED_CATF,
    TRAINING_MODE_INTELLIGENT,
    TRAINING_MODE_LEGACY_SEARCH,
    TRAINING_MODE_NORMAL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动由 GUI 配置的 AutoAugment 后端实验。")
    parser.add_argument("--config", required=True, help="gui_experiment_config.json 的路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    output = Path(config.output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output)
    config.save(output / "gui_experiment_config.json")

    dataset = Path(config.dataset_path)
    if not dataset.is_absolute():
        dataset = PROJECT_ROOT / dataset
    if not dataset.exists():
        raise FileNotFoundError(f"数据集路径不存在: {dataset}")

    policy = Policy.from_dict(config.policy or {"name": "gui_policy", "operations": []})
    policy.save(output / "gui_policy.json")
    search_space = build_search_space(config, policy)
    write_json_file(output / "gui_search_space.json", search_space_to_dict(search_space))

    class_names = DatasetAdapter().inspect_yolo_dataset(dataset).class_names
    records = None
    trial_layout = "flat"
    training_profile = config.training_profile()
    context: dict[str, Any] = {
        "class_names": class_names or None,
        "workers": config.workers,
        "proxy_score_version": "dataset2_v1",
        "training_profile": training_profile,
        "catf_profile": training_profile.get("backend_profile") or training_profile.get("profile_id"),
        "sampler_only": bool(training_profile.get("sampler_only", False)),
        "weighted_index_list": bool(training_profile.get("weighted_index_list", False)),
        "sampled_distribution_changed": bool(training_profile.get("sampled_distribution_changed", False)),
    }
    if config.evaluator == "train_yolo":
        split = resolve_yolo_train_val_records(dataset, seed=config.seed, missing_label="empty")
        fixed_val_images = output / "val_fixed" / "images" / "val"
        fixed_val_labels = output / "val_fixed" / "labels" / "val"
        copy_yolo_records(split.val_records, fixed_val_images, fixed_val_labels)
        records = split.train_records
        trial_layout = "train_yolo"
        context.update(
            {
                "val_images_dir": str(fixed_val_images.resolve()),
                "val_labels_dir": str(fixed_val_labels.resolve()),
                "train_val_mode": split.mode,
                "original_train_images_dir": str(split.train_images_dir.resolve()),
                "original_train_labels_dir": str(split.train_labels_dir.resolve()),
                "original_val_images_dir": str(split.val_images_dir.resolve()),
                "original_val_labels_dir": str(split.val_labels_dir.resolve()),
            }
        )

    stage_config = {
        "search_stage": {
            "stage": "search_short_train" if config.evaluator == "train_yolo" else "proxy",
            "search_epochs": config.epochs if config.evaluator == "train_yolo" else None,
            "search_samples": config.samples,
            "workers": config.workers,
            "adaptive_policy": config.adaptive_policy,
            "diagnose_trial_errors": config.diagnose_trial_errors,
            "proxy_prefilter": config.proxy_prefilter,
            "source": "qt_gui",
            "run_mode": config.run_mode,
            "policy_source": policy_source_for_mode(config.run_mode),
            "training_profile": training_profile,
        },
        "final_stage": {
            "stage": "final_full_train",
            "status": "not_requested_from_gui",
        },
    }
    write_json_file(output / "stage_config.json", stage_config)
    write_json_file(output / "run_config.json", {"script": "gui/adapters/gui_experiment_launcher.py", "config": config.to_dict()})

    evaluator = build_evaluator(config)

    def report(result: TrialResult, best: TrialResult | None) -> None:
        best_score = result.score if best is None else best.score
        metrics = result.metrics or {}
        print(
            "试验={trial:03d} 分数={score:.6f} 最佳分数={best:.6f} "
            "已接受={accepted} 试验目录={trial_dir} 调整原因={reason}".format(
                trial=result.trial_index,
                score=result.score,
                best=best_score,
                accepted="是" if result.accepted else "否",
                trial_dir=result.trial_dir,
                reason=result.adjust_reason,
            ),
            flush=True,
        )
        print(
            "GUI_EVENT "
            + json.dumps(
                {
                    "type": "trial_complete",
                    "trial_index": result.trial_index,
                    "trial_number": result.trial_index + 1,
                    "total_trials": config.trials,
                    "score": result.score,
                    "best_score": best_score,
                    "accepted": result.accepted,
                    "trial_dir": str(result.trial_dir),
                    "adjust_reason": result.adjust_reason,
                    "metrics": {
                        "mAP50": metrics.get("yolo_map50"),
                        "mAP50_95": metrics.get("yolo_map50_95"),
                        "bbox_retention": metrics.get("bbox_retention_raw", metrics.get("bbox_retention")),
                        "bbox_valid_rate": metrics.get("bbox_valid_rate"),
                        "diversity_score": metrics.get("diversity_score"),
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    search = RandomSearch(
        dataset_root=dataset,
        output_dir=output,
        num_trials=1 if config.run_mode in {TRAINING_MODE_NORMAL, TRAINING_MODE_CUSTOM, TRAINING_MODE_FIXED_CATF} else max(1, config.trials),
        num_samples=config.samples if config.samples > 0 else None,
        seed=config.seed,
        search_space=search_space,
        evaluator=evaluator,
        keep_intermediate=True,
        missing_label="empty",
        records=records,
        trial_layout=trial_layout,
        context=context,
        on_trial=report,
        adaptive_policy=config.adaptive_policy,
        stage_config=stage_config,
        proxy_prefilter={
            "enabled": bool(config.proxy_prefilter),
            "hard_filter_profile": "dataset2_v1",
            "proxy_score_version": "dataset2_v1",
            "keep_candidate_artifacts": False,
            "artifact_mode": "metrics_only",
        },
    )
    results = search.run()
    best = max(results, key=lambda item: item.score)
    write_top_level_gui_outputs(output, results, best)
    write_policy_search_readme(output, dataset_path=dataset, evaluator=config.evaluator, metric=config.metric)
    print(
        "GUI_EVENT "
        + json.dumps(
            {
                "type": "experiment_finished",
                "total_trials": len(results),
                "best_trial": best.trial_index,
                "best_score": best.score,
                "output_dir": str(output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"完成 输出目录={output.resolve()} 最佳分数={best.score:.6f}", flush=True)


def build_search_space_from_policy(policy: Policy) -> SearchSpace:
    if not policy.operations:
        raise ValueError("手动策略训练需要先在策略页配置至少一个增强操作。")
    operations = [
        OperationSpace(
            name=operation.name,
            prob_range=(operation.prob, operation.prob),
            strength_range=(operation.strength, operation.strength),
            params=dict(operation.params),
        )
        for operation in policy.operations
    ]
    return SearchSpace(
        operations=operations,
        operation_count_range=(len(operations), len(operations)),
        allow_repeated_operations=False,
        name_prefix="gui_policy",
    )


def build_clean_yolo_search_space() -> SearchSpace:
    return SearchSpace(
        operations=[
            OperationSpace(
                name="brightness",
                prob_range=(0.0, 0.0),
                strength_range=(0.0, 0.0),
                params={"max_delta": 0.0},
            )
        ],
        operation_count_range=(1, 1),
        allow_repeated_operations=False,
        name_prefix="clean_yolo_default",
    )


def build_preserve_weak_image_only_catf_space() -> SearchSpace:
    return SearchSpace(
        operations=[
            OperationSpace("brightness", (0.15, 0.55), (0.05, 0.28), {"max_delta": 0.16}),
            OperationSpace("contrast", (0.15, 0.55), (0.05, 0.30), {"max_delta": 0.25}),
            OperationSpace("gamma", (0.15, 0.50), (0.05, 0.25), {"min_gamma": 0.85, "max_gamma": 1.20}),
            OperationSpace("gaussian_noise", (0.10, 0.40), (0.02, 0.18), {"max_std": 0.03}),
            OperationSpace("gaussian_blur", (0.08, 0.30), (0.02, 0.16), {"max_kernel": 3}),
            OperationSpace("motion_blur", (0.05, 0.25), (0.02, 0.14), {"max_kernel": 5}),
            OperationSpace("sharpen", (0.10, 0.45), (0.05, 0.24), {"amount": 0.45}),
            OperationSpace("clahe", (0.10, 0.45), (0.05, 0.28), {"max_clip_limit": 2.0}),
        ],
        operation_count_range=(1, 3),
        allow_repeated_operations=False,
        name_prefix="preserve_weak_image_only_catf",
    )


def build_fixed_catf_v2_ablation_space() -> SearchSpace:
    return SearchSpace(
        operations=[
            OperationSpace("brightness", (0.45, 0.45), (0.25, 0.25), {"max_delta": 0.2}),
            OperationSpace("contrast", (0.45, 0.45), (0.25, 0.25), {"max_delta": 0.35}),
            OperationSpace("gaussian_blur", (0.20, 0.20), (0.12, 0.12), {"max_kernel": 5}),
            OperationSpace("horizontal_flip", (0.50, 0.50), (1.0, 1.0), {}),
        ],
        operation_count_range=(4, 4),
        allow_repeated_operations=False,
        name_prefix="fixed_catf_v2_ablation",
    )


def build_search_space(config: ExperimentConfig, policy: Policy) -> SearchSpace:
    if config.run_mode == TRAINING_MODE_NORMAL:
        return build_clean_yolo_search_space()
    if config.run_mode == TRAINING_MODE_CUSTOM:
        return build_search_space_from_policy(policy)
    if config.run_mode == TRAINING_MODE_INTELLIGENT:
        return build_preserve_weak_image_only_catf_space()
    if config.run_mode == TRAINING_MODE_FIXED_CATF:
        return build_fixed_catf_v2_ablation_space()
    if config.run_mode == TRAINING_MODE_LEGACY_SEARCH:
        return default_detection_search_space(operation_count_range=(2, 4), allow_repeated_operations=False)
    raise ValueError(f"未知实验运行模式: {config.run_mode}")


def policy_source_for_mode(run_mode: str) -> str:
    if run_mode == TRAINING_MODE_NORMAL:
        return "clean_yolo_default"
    if run_mode == TRAINING_MODE_CUSTOM:
        return "custom_policy"
    if run_mode == TRAINING_MODE_INTELLIGENT:
        return "preserve_weak_image_only_catf"
    if run_mode == TRAINING_MODE_FIXED_CATF:
        return "fixed_catf_v2_ablation"
    if run_mode == TRAINING_MODE_LEGACY_SEARCH:
        return "legacy_auto_search_space"
    return "unknown"


def build_evaluator(config: ExperimentConfig):
    if config.evaluator == "train_yolo":
        return YoloTrainValEvaluator(
            metric=config.metric,
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            model=config.model,
            workers=config.workers,
            diagnose_val_errors=config.diagnose_trial_errors,
        )
    return ProxyEvaluator(score_version="dataset2_v1")


def search_space_to_dict(search_space: SearchSpace) -> dict[str, Any]:
    return {
        "operations": [
            {
                "name": operation.name,
                "prob_range": list(operation.prob_range),
                "strength_range": list(operation.strength_range),
                "params": dict(operation.params),
            }
            for operation in search_space.operations
        ],
        "operation_count_range": list(search_space.operation_count_range),
        "allow_repeated_operations": search_space.allow_repeated_operations,
        "operation_weights": search_space.operation_weights,
        "forbidden_combinations": [list(item) for item in search_space.forbidden_combinations],
    }


def write_top_level_gui_outputs(output: Path, results: list[TrialResult], best: TrialResult) -> None:
    trials = [result.to_dict() for result in results]
    summary = {
        "total_trials": len(results),
        "best_trial": best.trial_index,
        "best_score": best.score,
        "best_policy": best.policy.to_dict(),
        "output_dir": str(output.resolve()),
    }
    metrics = {
        "trial_index": [result.trial_index for result in results],
        "score": [result.score for result in results],
        "mAP50": [result.metrics.get("yolo_map50") for result in results],
        "mAP50_95": [result.metrics.get("yolo_map50_95") for result in results],
        "proxy_score": [result.metrics.get("proxy_score", result.score) for result in results],
        "bbox_retention": [result.metrics.get("bbox_retention_raw", result.metrics.get("bbox_retention")) for result in results],
        "bbox_valid_rate": [result.metrics.get("bbox_valid_rate") for result in results],
        "diversity_score": [result.metrics.get("diversity_score") for result in results],
    }
    write_json_file(output / "summary.json", summary)
    write_json_file(output / "trial_record.json", {"trials": trials})
    write_json_file(output / "metrics.json", metrics)
    write_json_file(output / "policy.json", best.policy.to_dict())
    write_json_file(output / "diagnosis.json", best.diagnosis or {})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
