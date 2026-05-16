from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostic_pipeline import (
    build_final_augmented_dataset,
    generate_candidate_policies,
    append_strategy_memory,
    memory_guided_rerank,
    run_baseline_training,
    run_error_diagnosis,
    run_final_training,
    run_proxy_evaluation,
    run_short_training_selector,
    run_validation_prediction,
    write_dry_run_diagnosis,
    write_experiment_report,
    write_metric_consistency_audit,
)
from AutoAugment.diagnostic_pipeline.common import write_json, write_markdown
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml
from AutoAugment.utils import resolve_yolo_train_val_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-error diagnostic-driven augmentation pipeline for YOLO defect detection."
    )
    parser.add_argument("--dataset-root", required=True, help="YOLO dataset root.")
    parser.add_argument("--data-yaml", required=True, help="YOLO data.yaml path.")
    parser.add_argument("--output-dir", default="outputs/diagnostic_aug_pipeline_smoke")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--short-epochs", type=int, default=5)
    parser.add_argument("--final-epochs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-short-train", action="store_true")
    parser.add_argument("--skip-final-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-weights", default=None, help="Existing baseline best.pt used when --skip-baseline is set.")
    parser.add_argument("--predictions-dir", default=None, help="Existing YOLO prediction labels directory for diagnosis.")
    parser.add_argument("--proxy-samples", type=int, default=32, help="Number of training images used per proxy evaluation.")
    parser.add_argument("--augment-repeat", type=int, default=1, help="Augmented copies per training image for final dataset.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds for each YOLO command.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_root = Path(args.dataset_root)
    data_yaml = Path(args.data_yaml)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names_from_data_yaml(data_yaml) if data_yaml.exists() else {}
    split = resolve_or_plan_split(dataset_root, args.seed, dry_run=args.dry_run)
    run_config = {
        "script": "scripts/run_diagnostic_augmentation_pipeline.py",
        "args": vars(args),
        "dataset_root": str(dataset_root.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "output_dir": str(output.resolve()),
        "class_names": class_names,
        "train_count": len(split["train_records"]),
        "val_count": len(split["val_records"]),
        "framework_positioning": "validation_error_diagnostic_driven_augmentation",
        "not_yolo_network_structure_modification": True,
    }
    write_json(output / "run_config.json", run_config)

    baseline_record = None
    if args.skip_baseline:
        baseline_record = write_skipped_baseline(output / "baseline", args)
    else:
        baseline_record = run_baseline_training(
            data_yaml=data_yaml,
            output_dir=output / "baseline",
            dataset_root=dataset_root,
            model=args.model,
            epochs=args.baseline_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )

    baseline_weights = resolve_baseline_weights(args, baseline_record, output)
    prediction_record = run_validation_prediction(
        weights=baseline_weights,
        val_images_dir=split["val_images_dir"],
        val_labels_dir=split["val_labels_dir"],
        output_dir=output / "validation_prediction",
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        dry_run=args.dry_run,
        existing_predictions_dir=args.predictions_dir,
        timeout=args.timeout,
    )

    if args.dry_run:
        diagnosis = write_dry_run_diagnosis(output / "diagnosis")
    else:
        diagnosis = run_error_diagnosis(
            val_images_dir=split["val_images_dir"],
            val_labels_dir=split["val_labels_dir"],
            predictions_dir=prediction_record["predictions_dir"],
            output_dir=output / "diagnosis",
            class_names=class_names,
            match_iou=args.match_iou if args.match_iou is not None else args.iou,
        )
    metric_audit = write_metric_consistency_audit(
        output_dir=output / "metric_consistency",
        baseline_record=baseline_record,
        prediction_record=prediction_record,
        diagnosis=diagnosis,
        conf=args.conf,
        iou=args.iou,
        match_iou=args.match_iou if args.match_iou is not None else args.iou,
    )

    policies_payload = generate_candidate_policies(
        diagnosis,
        output_dir=output / "policies",
        seed=args.seed,
    )
    proxy_payload = run_proxy_evaluation(
        policies_payload=policies_payload,
        train_records=split["train_records"],
        output_dir=output / "proxy",
        seed=args.seed,
        proxy_samples=args.proxy_samples,
        dry_run=args.dry_run,
        class_names=class_names,
    )
    proxy_payload = memory_guided_rerank(
        diagnosis=diagnosis,
        proxy_payload=proxy_payload,
        output_dir=output / "strategy_memory",
    )
    short_payload = run_short_training_selector(
        proxy_payload=proxy_payload,
        train_records=split["train_records"],
        val_records=split["val_records"],
        output_dir=output / "short_training",
        data_yaml=data_yaml,
        model=args.model,
        epochs=args.short_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        top_k=args.top_k,
        class_names=class_names,
        seed=args.seed,
        dry_run=args.dry_run,
        skip_training=args.skip_short_train,
        timeout=args.timeout,
    )
    strategy_memory_record = append_strategy_memory(
        diagnosis=diagnosis,
        policies_payload=policies_payload,
        proxy_payload=proxy_payload,
        short_payload=short_payload,
        output_dir=output / "strategy_memory",
        dry_run=args.dry_run,
    )
    dataset_report = build_final_augmented_dataset(
        selected_policy=short_payload["selected_policy"],
        train_records=split["train_records"],
        val_records=split["val_records"],
        output_dir=output / "dataset_builder",
        class_names=class_names,
        seed=args.seed,
        augment_repeat=args.augment_repeat,
        dry_run=args.dry_run,
    )
    final_payload = run_final_training(
        data_yaml=dataset_report["data_yaml"],
        output_dir=output / "final_training",
        model=args.model,
        epochs=args.final_epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        dry_run=args.dry_run,
        skip_training=args.skip_final_train,
        timeout=args.timeout,
    )
    report = write_experiment_report(
        output_dir=output / "report",
        baseline_record=baseline_record,
        diagnosis=diagnosis,
        proxy_payload=proxy_payload,
        short_training_payload=short_payload,
        final_training_payload=final_payload,
    )
    write_pipeline_summary(
        output,
        args,
        baseline_record,
        prediction_record,
        diagnosis,
        policies_payload,
        proxy_payload,
        short_payload,
        dataset_report,
        final_payload,
        report,
        metric_audit,
        strategy_memory_record,
    )
    print_plan(output, args)


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    for name in ["baseline_epochs", "short_epochs", "final_epochs", "top_k", "batch", "imgsz"]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.proxy_samples <= 0:
        raise ValueError("--proxy-samples must be positive")
    if args.augment_repeat < 0:
        raise ValueError("--augment-repeat must be non-negative")


def resolve_or_plan_split(dataset_root: Path, seed: int, *, dry_run: bool) -> dict[str, Any]:
    if dataset_root.exists():
        split = resolve_yolo_train_val_records(dataset_root, seed=seed)
        return {
            "train_records": split.train_records,
            "val_records": split.val_records,
            "val_images_dir": split.val_images_dir,
            "val_labels_dir": split.val_labels_dir,
            "mode": split.mode,
        }
    if not dry_run:
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    return {
        "train_records": [],
        "val_records": [],
        "val_images_dir": dataset_root / "images" / "val",
        "val_labels_dir": dataset_root / "labels" / "val",
        "mode": "dry_run_missing_dataset",
    }


def write_skipped_baseline(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = args.baseline_weights or output_dir / "train_runs" / "baseline" / "weights" / "best.pt"
    record = {
        "stage": "baseline_training",
        "status": "skipped",
        "dry_run": args.dry_run,
        "best_pt": str(Path(weights).resolve()),
        "model": args.model,
        "epochs": args.baseline_epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "metrics": {},
    }
    write_json(output_dir / "baseline_metrics.json", record)
    return record


def resolve_baseline_weights(args: argparse.Namespace, baseline_record: dict[str, Any], output: Path) -> Path:
    if args.baseline_weights:
        return Path(args.baseline_weights)
    candidate = baseline_record.get("best_pt")
    if candidate:
        return Path(candidate)
    fallback = output / "baseline" / "train_runs" / "baseline" / "weights" / "best.pt"
    if not args.dry_run and not fallback.exists():
        raise FileNotFoundError("No baseline weights available. Run baseline training or pass --baseline-weights.")
    return fallback


def write_pipeline_summary(
    output: Path,
    args: argparse.Namespace,
    baseline_record: dict[str, Any],
    prediction_record: dict[str, Any],
    diagnosis: dict[str, Any],
    policies_payload: dict[str, Any],
    proxy_payload: dict[str, Any],
    short_payload: dict[str, Any],
    dataset_report: dict[str, Any],
    final_payload: dict[str, Any],
    report: dict[str, Any],
    metric_audit: dict[str, Any],
    strategy_memory_record: dict[str, Any],
) -> None:
    summary = {
        "dry_run": args.dry_run,
        "stages": {
            "baseline": baseline_record.get("status"),
            "validation_prediction": prediction_record.get("status"),
            "diagnosis": diagnosis.get("status"),
            "metric_consistency_audit": metric_audit.get("status"),
            "policy_mapping": policies_payload.get("status"),
            "proxy": proxy_payload.get("status"),
            "strategy_memory": strategy_memory_record.get("status"),
            "short_training": short_payload.get("status"),
            "dataset_builder": dataset_report.get("status"),
            "final_training": final_payload.get("status"),
            "report": report.get("status"),
        },
        "key_outputs": {
            "diagnosis": str((output / "diagnosis" / "diagnosis.json").resolve()),
            "metric_consistency_audit": str((output / "metric_consistency" / "metric_consistency_audit.md").resolve()),
            "candidate_policies": str((output / "policies" / "candidate_policies.json").resolve()),
            "policy_update_report": str((output / "policies" / "policy_update_report.md").resolve()),
            "proxy_metrics": str((output / "proxy" / "proxy_metrics.json").resolve()),
            "proxy_ranking": str((output / "proxy" / "proxy_ranking.json").resolve()),
            "copy_paste_filter_audit": str((output / "proxy" / "copy_paste_filter_audit.md").resolve()),
            "memory_guided_ranking": str((output / "strategy_memory" / "memory_guided_ranking.json").resolve()),
            "strategy_memory_report": str((output / "strategy_memory" / "strategy_memory_report.md").resolve()),
            "selected_policy": str((output / "short_training" / "selected_policy.json").resolve()),
            "dataset_build_report": str((output / "dataset_builder" / "dataset_build_report.json").resolve()),
            "experiment_summary": str((output / "report" / "experiment_summary.md").resolve()),
        },
    }
    write_json(output / "pipeline_summary.json", summary)
    write_markdown(
        output / "README.txt",
        [
            "Diagnostic Augmentation Pipeline Run",
            "",
            f"Dry run: {args.dry_run}",
            f"Dataset root: {Path(args.dataset_root).resolve()}",
            f"Data YAML: {Path(args.data_yaml).resolve()}",
            f"Output: {output.resolve()}",
            "",
            "Key outputs:",
            f"- diagnosis.json: {summary['key_outputs']['diagnosis']}",
            f"- metric_consistency_audit.md: {summary['key_outputs']['metric_consistency_audit']}",
            f"- candidate_policies.json: {summary['key_outputs']['candidate_policies']}",
            f"- policy_update_report.md: {summary['key_outputs']['policy_update_report']}",
            f"- proxy_metrics.json: {summary['key_outputs']['proxy_metrics']}",
            f"- copy_paste_filter_audit.md: {summary['key_outputs']['copy_paste_filter_audit']}",
            f"- memory_guided_ranking.json: {summary['key_outputs']['memory_guided_ranking']}",
            f"- selected_policy.json: {summary['key_outputs']['selected_policy']}",
            f"- experiment_summary.md: {summary['key_outputs']['experiment_summary']}",
        ],
    )


def print_plan(output: Path, args: argparse.Namespace) -> None:
    print("Diagnostic augmentation pipeline completed." if not args.dry_run else "Diagnostic augmentation pipeline dry-run completed.")
    print(f"- Output: {output.resolve()}")
    print(f"- Diagnosis: {(output / 'diagnosis' / 'diagnosis.json').resolve()}")
    print(f"- Candidate policies: {(output / 'policies' / 'candidate_policies.json').resolve()}")
    print(f"- Proxy ranking: {(output / 'proxy' / 'proxy_ranking.json').resolve()}")
    print(f"- Selected policy: {(output / 'short_training' / 'selected_policy.json').resolve()}")
    print(f"- Experiment summary: {(output / 'report' / 'experiment_summary.md').resolve()}")
    if args.dry_run:
        print("")
        print("Planned commands were written under baseline/, validation_prediction/, short_training/trials/, and final_training/.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
