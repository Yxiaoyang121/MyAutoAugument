from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.policies import SearchSpace, default_detection_search_space
from AutoAugment.search import (
    CommandEvaluator,
    ProxyEvaluator,
    RandomSearch,
    TrialResult,
    YoloCommandEvaluator,
    YoloTrainValEvaluator,
)
from AutoAugment.utils import (
    copy_yolo_records,
    resolve_output_dir,
    resolve_yolo_train_val_records,
    write_dataset_path_file,
    write_json_file,
    write_policy_search_readme,
)


FINAL_FULL_DATASET_NOT_IMPLEMENTED = (
    "final_full_dataset=true was requested, but final full-dataset training is not implemented."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random policy search for a YOLO detection dataset.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root. Supports absolute or relative paths.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to outputs/runs/policy_search/run_YYYYMMDD_HHMMSS.")
    parser.add_argument("--trials", type=int, default=10, help="Number of random policies to evaluate.")
    parser.add_argument("--samples", type=int, default=20, help="Images sampled per trial. Use <=0 for all images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--evaluator", choices=["proxy", "command", "yolo", "train_yolo"], default="proxy", help="Evaluator backend.")
    parser.add_argument("--command-template", default=None, help="Generic external evaluator command template.")
    parser.add_argument("--score-regex", default=r"score\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", help="Regex used by CommandEvaluator to parse the score.")
    parser.add_argument("--yolo-command", default=None, help="YOLO validation command template used by --evaluator yolo.")
    parser.add_argument("--metric", choices=["map50", "map50_95"], default="map50", help="YOLO metric used as score.")
    parser.add_argument("--dataset-yaml", default=None, help="Optional existing YOLO data yaml for --evaluator yolo.")
    parser.add_argument("--class-names", default=None, help="Comma-separated class names for generated data.yaml.")
    parser.add_argument("--timeout", type=int, default=None, help="External command timeout in seconds.")
    parser.add_argument("--hybrid-proxy-weight", type=float, default=0.0, help="Blend YOLO score with ProxyEvaluator score.")
    parser.add_argument("--train-command", default=None, help="YOLO train command template for --evaluator train_yolo.")
    parser.add_argument("--val-command", default=None, help="YOLO val command template for --evaluator train_yolo.")
    parser.add_argument("--predict-command", default=None, help="YOLO predict command template for per-trial diagnostics in train_yolo mode.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model used by --evaluator train_yolo.")
    parser.add_argument("--epochs", type=int, default=10, help="YOLO train epochs used by --evaluator train_yolo.")
    parser.add_argument("--search-epochs", type=int, default=None, help="Short-train epochs for policy search. Defaults to --epochs.")
    parser.add_argument("--final-epochs", type=int, default=None, help="Final full-train epochs recorded for reproducibility.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size used by --evaluator train_yolo.")
    parser.add_argument("--batch", type=int, default=8, help="YOLO batch size used by --evaluator train_yolo.")
    parser.add_argument("--workers", type=int, default=0, help="YOLO dataloader workers for train/val/predict commands.")
    parser.add_argument("--search-samples", type=int, default=None, help="Images sampled per search trial. Defaults to --samples.")
    parser.add_argument("--search-subset", default=None, help="Optional search subset name or manifest path recorded in run_config.")
    parser.add_argument("--final-full-dataset", action="store_true", help="Record that final validation/full training should use the full dataset.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze YOLO backbone during short-train search.")
    parser.add_argument("--no-adaptive-policy", action="store_true", help="Disable diagnosis-driven policy updates and use legacy random sampling.")
    parser.add_argument("--diagnose-trial-errors", action="store_true", help="Run YOLO predict plus validation error analyzer after each train_yolo trial.")
    parser.add_argument("--diagnosis-conf", type=float, default=0.25, help="Confidence threshold for per-trial diagnostic YOLO predict.")
    parser.add_argument("--diagnosis-iou", type=float, default=0.5, help="NMS and matching IoU used by per-trial diagnostics.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for simple images/labels datasets in train_yolo mode.")
    parser.add_argument("--train-images", default=None, help="Explicit train images directory for train_yolo mode.")
    parser.add_argument("--train-labels", default=None, help="Explicit train labels directory for train_yolo mode.")
    parser.add_argument("--val-images", default=None, help="Explicit validation images directory for train_yolo mode.")
    parser.add_argument("--val-labels", default=None, help="Explicit validation labels directory for train_yolo mode.")
    parser.add_argument("--run-name", default=None, help="Custom default run directory name when --output is omitted.")
    parser.add_argument("--policy-ops-min", type=int, default=2, help="Minimum operations in a sampled policy.")
    parser.add_argument("--policy-ops-max", type=int, default=4, help="Maximum operations in a sampled policy.")
    parser.add_argument("--search-space-json", default=None, help="Optional advisor_search_space.json from diagnostics.")
    parser.add_argument("--missing-label", choices=["empty", "skip", "error"], default="empty", help="How to handle images without txt labels.")
    parser.add_argument("--no-keep-intermediate", action="store_true", help="Delete non-best trial image/label directories after scoring.")
    parser.add_argument("--proxy-prefilter", action="store_true", help="Enable pre-training proxy candidate policy filtering.")
    parser.add_argument("--candidate-policies", type=int, default=24, help="Candidate policies sampled before each trial when proxy prefilter is enabled.")
    parser.add_argument("--proxy-eval-samples", type=int, default=96, help="Samples used to proxy-evaluate each candidate policy.")
    parser.add_argument("--proxy-top-k", type=int, default=1, help="Number of top proxy candidates recorded. Only the best one is trained.")
    parser.add_argument("--proxy-score-version", default="dataset2_v1", help="Proxy score formula version.")
    parser.add_argument("--proxy-hard-filter-profile", default="dataset2_v1", help="Proxy hard filter threshold profile.")
    parser.add_argument("--keep-proxy-candidate-artifacts", action="store_true", help="Retain proxy candidate augmented images/labels for debugging.")
    parser.add_argument(
        "--proxy-candidate-artifact-mode",
        choices=["metrics_only", "selected_only", "all"],
        default="metrics_only",
        help="Candidate artifact retention mode. Defaults to metrics_only.",
    )
    parser.add_argument("--max-proxy-artifact-gb", type=float, default=5.0, help="Maximum retained proxy artifact size before forcing metrics_only mode.")
    return parser.parse_args()


def build_evaluator(args: argparse.Namespace):
    if args.evaluator == "proxy":
        return ProxyEvaluator()
    if args.evaluator == "command":
        if not args.command_template:
            raise ValueError("--command-template is required when --evaluator command is used")
        return CommandEvaluator(args.command_template, score_regex=args.score_regex, timeout=args.timeout)
    if args.evaluator == "train_yolo":
        search_epochs = getattr(args, "search_epochs", None) or args.epochs
        return YoloTrainValEvaluator(
            train_command_template=args.train_command,
            val_command_template=args.val_command,
            metric=args.metric,
            epochs=search_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            model=args.model,
            timeout=args.timeout,
            hybrid_proxy_weight=args.hybrid_proxy_weight,
            freeze_backbone=getattr(args, "freeze_backbone", False),
            diagnose_val_errors=getattr(args, "diagnose_trial_errors", False),
            predict_command_template=getattr(args, "predict_command", None),
            diagnosis_conf=getattr(args, "diagnosis_conf", 0.25),
            diagnosis_iou=getattr(args, "diagnosis_iou", 0.5),
            workers=getattr(args, "workers", 0),
        )
    if not args.yolo_command:
        raise ValueError("--yolo-command is required when --evaluator yolo is used")
    return YoloCommandEvaluator(
        args.yolo_command,
        metric=args.metric,
        dataset_yaml=args.dataset_yaml,
        class_names=parse_class_names(args.class_names),
        timeout=args.timeout,
        hybrid_proxy_weight=args.hybrid_proxy_weight,
        workers=getattr(args, "workers", 0),
    )


def parse_class_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or None


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    output = resolve_output_dir(args.output, "policy_search", run_name=args.run_name)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset}")
    if args.policy_ops_min <= 0 or args.policy_ops_max < args.policy_ops_min:
        raise ValueError("--policy-ops-min/--policy-ops-max must be positive and ordered")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.candidate_policies <= 0:
        raise ValueError("--candidate-policies must be positive")
    if args.proxy_eval_samples <= 0:
        raise ValueError("--proxy-eval-samples must be positive")
    if args.proxy_top_k <= 0:
        raise ValueError("--proxy-top-k must be positive")
    if args.max_proxy_artifact_gb <= 0:
        raise ValueError("--max-proxy-artifact-gb must be positive")
    ensure_final_stage_supported(args)

    output.mkdir(parents=True, exist_ok=True)
    class_names = parse_class_names(args.class_names)
    effective_samples = args.search_samples if args.search_samples is not None else args.samples
    effective_search_epochs = args.search_epochs if args.search_epochs is not None else args.epochs
    final_epochs = args.final_epochs if args.final_epochs is not None else args.epochs
    proxy_prefilter_config = {
        "enabled": args.proxy_prefilter,
        "candidate_policies": args.candidate_policies,
        "proxy_eval_samples": args.proxy_eval_samples,
        "proxy_top_k": args.proxy_top_k,
        "proxy_score_version": args.proxy_score_version,
        "hard_filter_profile": args.proxy_hard_filter_profile,
        "keep_candidate_artifacts": args.keep_proxy_candidate_artifacts,
        "artifact_mode": args.proxy_candidate_artifact_mode,
        "max_proxy_artifact_gb": args.max_proxy_artifact_gb,
    }
    stage_config = {
        "search_stage": {
            "stage": "search_short_train" if args.evaluator == "train_yolo" else args.evaluator,
            "search_epochs": effective_search_epochs if args.evaluator == "train_yolo" else None,
            "search_samples": effective_samples,
            "search_subset": args.search_subset,
            "freeze_backbone": args.freeze_backbone,
            "workers": args.workers,
            "adaptive_policy": not args.no_adaptive_policy,
            "diagnose_trial_errors": args.diagnose_trial_errors,
            "proxy_prefilter": proxy_prefilter_config,
        },
        "final_stage": {
            "stage": "final_full_train",
            "final_epochs": final_epochs,
            "final_full_dataset": args.final_full_dataset,
            "workers": args.workers,
            "status": "not_requested",
        },
    }
    train_val_split = None
    train_records = None
    trial_layout = "flat"
    search_context = {"class_names": class_names, "workers": args.workers}
    if args.evaluator == "train_yolo":
        train_val_split = resolve_yolo_train_val_records(
            dataset,
            train_images=args.train_images,
            train_labels=args.train_labels,
            val_images=args.val_images,
            val_labels=args.val_labels,
            val_ratio=args.val_ratio,
            seed=args.seed,
            missing_label=args.missing_label,
        )
        fixed_val_images = output / "val_fixed" / "images" / "val"
        fixed_val_labels = output / "val_fixed" / "labels" / "val"
        copy_yolo_records(train_val_split.val_records, fixed_val_images, fixed_val_labels)
        train_records = train_val_split.train_records
        trial_layout = "train_yolo"
        search_context.update(
            {
                "val_images_dir": str(fixed_val_images.resolve()),
                "val_labels_dir": str(fixed_val_labels.resolve()),
                "train_val_mode": train_val_split.mode,
                "original_train_images_dir": str(train_val_split.train_images_dir.resolve()),
                "original_train_labels_dir": str(train_val_split.train_labels_dir.resolve()),
                "original_val_images_dir": str(train_val_split.val_images_dir.resolve()),
                "original_val_labels_dir": str(train_val_split.val_labels_dir.resolve()),
            }
        )

    write_dataset_path_file(output / "input_dataset.txt", supplied_path=args.dataset, resolved_path=dataset)
    write_json_file(
        output / "run_config.json",
        {
            "script": "examples/run_policy_search.py",
            "args": vars(args),
            "input_dataset": str(dataset.resolve()),
            "output_dir": str(output.resolve()),
            "trial_dir": str((output / "trials").resolve()),
            "data_yaml_generation": _data_yaml_generation_mode(args),
            "search_space_json": str(Path(args.search_space_json).resolve()) if args.search_space_json else None,
            "stage_config": stage_config,
            "adaptive_policy": not args.no_adaptive_policy,
            "train_val_split": None
            if train_val_split is None
            else {
                "mode": train_val_split.mode,
                "train_count": len(train_val_split.train_records),
                "val_count": len(train_val_split.val_records),
                "fixed_val_images_dir": str((output / "val_fixed" / "images" / "val").resolve()),
                "fixed_val_labels_dir": str((output / "val_fixed" / "labels" / "val").resolve()),
            },
        },
    )
    write_json_file(output / "stage_config.json", stage_config)
    write_policy_search_readme(
        output,
        dataset_path=dataset,
        evaluator=args.evaluator,
        metric=args.metric if args.evaluator in {"yolo", "train_yolo"} else None,
        search_space_json=args.search_space_json,
    )

    search_space = build_search_space(args)
    evaluator = build_evaluator(args)
    initial_diagnostic_context = load_initial_diagnostic_context(args.search_space_json)
    if initial_diagnostic_context:
        search_context.update(initial_diagnostic_context)

    def report(result: TrialResult, best: TrialResult | None) -> None:
        best_score = best.score if best is not None else result.score
        metric_name = result.metrics.get("yolo_metric_name", args.evaluator)
        print(
            "trial_index={trial:03d} policy={policy} score={score:.6f} metric={metric} "
            "best_score={best:.6f} trial_dir={trial_dir} train_dataset={train_dataset} "
            "data_yaml={data_yaml} best_pt={best_pt} accepted={accepted} adjust_reason={adjust_reason}".format(
                trial=result.trial_index,
                policy=result.policy.name,
                score=result.score,
                metric=metric_name,
                best=best_score,
                trial_dir=result.trial_dir,
                train_dataset=result.trial_dir / "dataset" if args.evaluator == "train_yolo" else "",
                data_yaml=result.metrics.get("dataset_yaml", ""),
                best_pt=result.metrics.get("best_pt", ""),
                accepted=result.accepted,
                adjust_reason=result.adjust_reason,
            )
        )

    search = RandomSearch(
        dataset_root=dataset,
        output_dir=output,
        num_trials=args.trials,
        num_samples=effective_samples,
        seed=args.seed,
        search_space=search_space,
        evaluator=evaluator,
        keep_intermediate=not args.no_keep_intermediate,
        missing_label=args.missing_label,
        records=train_records,
        trial_layout=trial_layout,
        context=search_context,
        on_trial=report,
        adaptive_policy=not args.no_adaptive_policy,
        stage_config=stage_config,
        proxy_prefilter=proxy_prefilter_config,
    )
    results = search.run()
    best = max(results, key=lambda item: item.score)
    write_policy_search_readme(
        output,
        dataset_path=dataset,
        evaluator=args.evaluator,
        metric=args.metric if args.evaluator in {"yolo", "train_yolo"} else None,
        search_space_json=args.search_space_json,
    )
    print("Policy search completed:")
    print(f"- Input dataset: {dataset.resolve()}")
    print(f"- Search output dir: {output.resolve()}")
    print(f"- Trial dir: {(output / 'trials').resolve()}")
    print(f"- Best policy: {(output / 'best_policy.json').resolve()}")
    print(f"- Trials CSV: {(output / 'trials.csv').resolve()}")
    print(f"- Policy history: {(output / 'policy_history.jsonl').resolve()}")
    print(f"- Stage config: {(output / 'stage_config.json').resolve()}")
    print(f"- README: {(output / 'README.txt').resolve()}")
    if args.search_space_json:
        print(f"- Advisor search space: {Path(args.search_space_json).resolve()}")
    print(f"- Best score: {best.score:.6f}")
    if args.evaluator == "train_yolo":
        print("- train_yolo note: every trial trains a model; start with --trials 8 --search-samples 50 --search-epochs 5.")


def _data_yaml_generation_mode(args: argparse.Namespace) -> str:
    if args.evaluator == "train_yolo":
        return "train_val_per_trial"
    if args.evaluator == "yolo" and args.dataset_yaml is None:
        return "auto_per_trial"
    return "not_auto_generated"


def ensure_final_stage_supported(args: argparse.Namespace) -> None:
    if getattr(args, "final_full_dataset", False):
        raise NotImplementedError(FINAL_FULL_DATASET_NOT_IMPLEMENTED)


def build_search_space(args: argparse.Namespace):
    operation_count_range = (args.policy_ops_min, args.policy_ops_max)
    if args.search_space_json:
        path = Path(args.search_space_json)
        if not path.exists():
            raise FileNotFoundError(f"--search-space-json does not exist: {path}")
        return SearchSpace.from_advisor_json(
            path,
            operation_count_range=operation_count_range,
            allow_repeated_operations=False,
        )
    return default_detection_search_space(
        operation_count_range=operation_count_range,
        allow_repeated_operations=False,
    )


def load_initial_diagnostic_context(search_space_json: str | None) -> dict[str, object]:
    if not search_space_json:
        return {}
    path = Path(search_space_json)
    context: dict[str, object] = {}
    parent = path.parent
    summary_path = parent / "error_summary.json"
    advice_path = parent / "augmentation_advice.json"
    if summary_path.exists():
        context["trial_error_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        context["initial_error_summary_path"] = str(summary_path.resolve())
    if advice_path.exists():
        context["initial_advice"] = json.loads(advice_path.read_text(encoding="utf-8"))
        context["initial_advice_path"] = str(advice_path.resolve())
    return context


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
