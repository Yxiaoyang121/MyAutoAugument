from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostic_pipeline import run_error_diagnosis, run_validation_prediction  # noqa: E402
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402


PYTHON_EXE = Path(r"D:\Anaconda\envs\pytorch\python.exe")
DEFAULT_DATA = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "outputs/experiments"
DEFAULT_RUN_ID = "yolo_default_feedback_aug_50ep"
REFERENCE_METRICS = (
    PROJECT_ROOT
    / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/diagnosis_constrained_metrics.json"
)

METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
CONSTRAINT_DROP_LIMIT = -0.01
OFFICIAL_YOLO_DEFAULT_AUG = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.4,
    "close_mosaic": 10,
}
CUSTOM_OP_LIMITS = {
    "clahe": (0.0, 0.35, 0.0, 0.45),
    "gamma": (0.0, 0.35, 0.0, 0.45),
    "sharpen_mild": (0.0, 0.30, 0.0, 0.40),
    "local_contrast": (0.0, 0.30, 0.0, 0.40),
    "cutout_safe": (0.0, 0.25, 0.0, 0.35),
}


@dataclass
class StageRecord:
    stage_index: int
    epochs: int
    global_epoch_start: int
    global_epoch_end: int
    run_dir: Path
    input_checkpoint: str
    train_command: str
    train_success: bool
    train_error: str | None
    train_wall_seconds: float
    weights_last: str | None
    weights_best: str | None
    val_metrics: dict[str, Any]
    diagnosis: dict[str, Any]
    constraints: dict[str, Any]
    policy_before: dict[str, Any]
    policy_after: dict[str, Any] | None
    adjustments: list[dict[str, Any]]


def main() -> None:
    args = parse_args()
    output_dir = (Path(args.project) / args.run_id).resolve()
    configure_environment(output_dir)
    prepare_dirs(output_dir)

    reference = load_reference_metrics(Path(args.reference_metrics))
    policy_state = initial_policy_state()
    write_json(output_dir / "configs/experiment_config.json", build_experiment_config(args, output_dir, reference))
    write_json(output_dir / "configs/official_yolo_default_aug.json", OFFICIAL_YOLO_DEFAULT_AUG)

    stage_lengths = build_stage_lengths(args.epochs, args.feedback_interval)
    current_model: str | Path = args.model
    global_epoch = 0
    history: list[dict[str, Any]] = []
    stages: list[StageRecord] = []
    train_success = True
    final_metrics: dict[str, Any] = {}

    for stage_index, stage_epochs in enumerate(stage_lengths):
        stage_before = deepcopy(policy_state)
        stage_record = run_stage(
            args=args,
            output_dir=output_dir,
            stage_index=stage_index,
            stage_epochs=stage_epochs,
            global_epoch=global_epoch,
            model=current_model,
            policy_state=stage_before,
            reference_metrics=reference["metrics"],
        )
        stages.append(stage_record)
        final_metrics = stage_record.val_metrics or final_metrics
        train_success = train_success and stage_record.train_success
        if not stage_record.train_success:
            break

        next_epoch = global_epoch + stage_epochs
        if stage_index < len(stage_lengths) - 1 and next_epoch >= int(args.feedback_start_epoch):
            policy_state, adjustments = update_feedback_policy(
                policy_state=policy_state,
                diagnosis=stage_record.diagnosis,
                metrics=stage_record.val_metrics,
                reference_metrics=reference["metrics"],
            )
            stage_record.policy_after = deepcopy(policy_state)
            stage_record.adjustments = adjustments
            history.append(
                build_history_record(
                    stage_record=stage_record,
                    profile=args.feedback_profile,
                    adjustments=adjustments,
                )
            )
            write_json(output_dir / "configs" / f"stage_{stage_index:02d}_updated_policy_state.json", policy_state)

        current_model = Path(stage_record.weights_last) if stage_record.weights_last else current_model
        global_epoch = next_epoch

    write_policy_history(output_dir / "reports", history, policy_state)
    constraint_scoring = build_constraint_scoring(final_metrics, reference["metrics"])
    final_payload = build_final_payload(args, output_dir, reference, stages, history, policy_state, final_metrics, constraint_scoring)
    write_outputs(output_dir, final_payload)

    if not args.skip_doc_update:
        update_state_docs(final_payload)
        export_project_snapshot()

    print(json.dumps(final_payload["summary"], ensure_ascii=False, indent=2))
    if not train_success:
        raise RuntimeError("one or more feedback smoke stages failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with Ultralytics YOLO default augmentation plus diagnosis feedback.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--feedback-interval", type=int, default=5)
    parser.add_argument("--feedback-start-epoch", type=int, default=5)
    parser.add_argument("--feedback-profile", default="industrial")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--reference-metrics", default=str(REFERENCE_METRICS))
    parser.add_argument("--skip-doc-update", action="store_true")
    return parser.parse_args()


def configure_environment(output_dir: Path) -> None:
    scripts_dir = PYTHON_EXE.parent / "Scripts"
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    yolo_config = output_dir / "configs" / "ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())


def prepare_dirs(output_dir: Path) -> None:
    for name in ["configs", "logs", "reports", "stages"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def build_experiment_config(args: argparse.Namespace, output_dir: Path, reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "entrypoint": "scripts/train_yolo_default_with_feedback.py",
        "model": args.model,
        "data": str(Path(args.data)),
        "epochs": int(args.epochs),
        "feedback_interval": int(args.feedback_interval),
        "feedback_start_epoch": int(args.feedback_start_epoch),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": str(args.device),
        "seed": int(args.seed),
        "base_augmentation": "Ultralytics YOLO default augmentation remains enabled.",
        "custom_yolo_like_replacement": False,
        "disabled_custom_ops": ["mosaic4", "randaugment_like", "copy_paste"],
        "constraint_reference": reference,
        "constraint_limits": {
            "precision_min_delta": CONSTRAINT_DROP_LIMIT,
            "map50_min_delta": CONSTRAINT_DROP_LIMIT,
            "map50_95_min_delta": CONSTRAINT_DROP_LIMIT,
        },
    }


def initial_policy_state() -> dict[str, Any]:
    return {
        "official_yolo_aug": deepcopy(OFFICIAL_YOLO_DEFAULT_AUG),
        "custom_industrial_policy": {
            "policy_id": "yolo_default_feedback_extra_industrial",
            "notes": [
                "Ultralytics YOLO default augmentation is the base.",
                "No custom mosaic4 or randaugment_like is used.",
                "copy_paste is disabled pending constrained object-bank design.",
            ],
            "operations": [
                op_payload("clahe", 0.0, 0.0),
                op_payload("gamma", 0.0, 0.0),
                op_payload("sharpen_mild", 0.0, 0.0, params={"amount": 0.6}),
                op_payload("local_contrast", 0.0, 0.0),
                op_payload(
                    "cutout_safe",
                    0.0,
                    0.0,
                    params={"max_holes": 2, "max_fraction": 0.12, "max_bbox_overlap": 0.05},
                ),
            ],
        },
    }


def op_payload(name: str, prob: float, strength: float, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    min_prob, max_prob, min_strength, max_strength = CUSTOM_OP_LIMITS[name]
    return {
        "name": name,
        "prob": prob,
        "base_prob": prob,
        "min_prob": min_prob,
        "max_prob": max_prob,
        "strength": strength,
        "base_strength": strength,
        "min_strength": min_strength,
        "max_strength": max_strength,
        "params": params or {},
    }


def build_stage_lengths(total_epochs: int, interval: int) -> list[int]:
    total = max(1, int(total_epochs))
    step = max(1, int(interval))
    lengths: list[int] = []
    remaining = total
    while remaining:
        current = min(step, remaining)
        lengths.append(current)
        remaining -= current
    return lengths


def run_stage(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    stage_index: int,
    stage_epochs: int,
    global_epoch: int,
    model: str | Path,
    policy_state: dict[str, Any],
    reference_metrics: dict[str, Any],
) -> StageRecord:
    stage_name = f"stage_{stage_index:02d}"
    stage_run_dir = output_dir / "stages" / stage_name
    stage_policy = training_policy(policy_state)
    policy_path = output_dir / "configs" / f"{stage_name}_custom_policy.json"
    config_path = output_dir / "configs" / f"{stage_name}_config.json"
    yolo_overrides = changed_yolo_overrides(policy_state["official_yolo_aug"])
    write_json(policy_path, stage_policy)
    write_json(
        config_path,
        {
            "stage_index": stage_index,
            "stage_epochs": stage_epochs,
            "global_epoch_start": global_epoch,
            "global_epoch_end": global_epoch + stage_epochs,
            "input_model": str(model),
            "base_augmentation": "Ultralytics YOLO default enabled",
            "yolo_aug_overrides": yolo_overrides,
            "custom_policy": stage_policy,
        },
    )
    command = build_stage_command(args, output_dir, stage_name, model, stage_epochs, policy_path, yolo_overrides)
    write_text(output_dir / "logs" / f"{stage_name}.train.command.txt", subprocess.list2cmdline(command))

    print(f"[feedback-stage:start] {stage_name} epochs={stage_epochs} model={model}")
    start = time.time()
    with (output_dir / "logs" / f"{stage_name}.train.stdout.log").open("w", encoding="utf-8", errors="replace") as stdout:
        with (output_dir / "logs" / f"{stage_name}.train.stderr.log").open("w", encoding="utf-8", errors="replace") as stderr:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout,
                stderr=stderr,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=os.environ.copy(),
            )
    wall = time.time() - start
    write_text(output_dir / "logs" / f"{stage_name}.train.wall_seconds.txt", f"{wall:.3f}")
    write_text(output_dir / "logs" / f"{stage_name}.train.exitcode", str(completed.returncode))
    success = completed.returncode == 0
    error = None if success else f"stage command failed with exit code {completed.returncode}"

    last_pt = stage_run_dir / "train" / "weights" / "last.pt"
    best_pt = stage_run_dir / "train" / "weights" / "best.pt"
    stats = read_json(stage_run_dir / "reports" / "online_aug_stats.json") if (stage_run_dir / "reports" / "online_aug_stats.json").exists() else {}
    val_metrics = compact_metrics(stats.get("val_metrics", {}))
    diagnosis: dict[str, Any] = {}
    if success and last_pt.exists():
        diagnosis = run_stage_diagnosis(args, stage_run_dir, last_pt)
    constraints = evaluate_constraints(metric_delta(val_metrics, reference_metrics))
    write_json(
        stage_run_dir / "reports" / "stage_constraint_status.json",
        {"reference": "YOLO default seed=42", "metrics": val_metrics, "constraints": constraints},
    )
    print(f"[feedback-stage:done] {stage_name} success={success} wall_seconds={wall:.1f}")
    return StageRecord(
        stage_index=stage_index,
        epochs=stage_epochs,
        global_epoch_start=global_epoch,
        global_epoch_end=global_epoch + stage_epochs,
        run_dir=stage_run_dir,
        input_checkpoint=str(model),
        train_command=subprocess.list2cmdline(command),
        train_success=success,
        train_error=error,
        train_wall_seconds=wall,
        weights_last=str(last_pt.resolve()) if last_pt.exists() else None,
        weights_best=str(best_pt.resolve()) if best_pt.exists() else None,
        val_metrics=val_metrics,
        diagnosis=diagnosis,
        constraints=constraints,
        policy_before=policy_state,
        policy_after=None,
        adjustments=[],
    )


def build_stage_command(
    args: argparse.Namespace,
    output_dir: Path,
    stage_name: str,
    model: str | Path,
    stage_epochs: int,
    policy_path: Path,
    yolo_overrides: dict[str, Any],
) -> list[str]:
    command = [
        str(PYTHON_EXE),
        str(PROJECT_ROOT / "scripts/train_yolo_online_aug.py"),
        "--model",
        str(model),
        "--data",
        str(args.data),
        "--policy",
        str(policy_path),
        "--epochs",
        str(stage_epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--run-id",
        stage_name,
        "--project",
        str(output_dir / "stages"),
        "--no-disable-yolo-aug",
        "--save-preview",
        "--preview-count",
        str(args.preview_count),
        "--skip-doc-update",
    ]
    if yolo_overrides:
        command.extend(["--yolo-aug-overrides", json.dumps(yolo_overrides, ensure_ascii=False, sort_keys=True)])
    return command


def training_policy(policy_state: dict[str, Any]) -> dict[str, Any]:
    policy = deepcopy(policy_state["custom_industrial_policy"])
    policy["operations"] = [
        op for op in policy.get("operations", []) if float(op.get("prob", 0.0) or 0.0) > 0.0
    ]
    blocked = {"mosaic4", "randaugment_like", "copy_paste", "online_copy_paste", "class_balanced_copy_paste"}
    for op in policy["operations"]:
        name = str(op.get("name", ""))
        if name in blocked:
            raise ValueError(f"{name} is not allowed in YOLO-default feedback mode")
    return policy


def changed_yolo_overrides(current: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, default in OFFICIAL_YOLO_DEFAULT_AUG.items():
        value = current.get(key, default)
        if value != default:
            overrides[key] = value
    return overrides


def run_stage_diagnosis(args: argparse.Namespace, stage_run_dir: Path, weights: Path) -> dict[str, Any]:
    val_images, val_labels = resolve_val_image_label_dirs(Path(args.data))
    class_names = load_class_names_from_data_yaml(args.data)
    prediction = run_validation_prediction(
        weights=weights,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        output_dir=stage_run_dir / "diagnosis_prediction",
        imgsz=int(args.imgsz),
        workers=int(args.workers),
        device=str(args.device),
        conf=0.25,
        iou=0.5,
        dry_run=False,
    )
    diagnosis = run_error_diagnosis(
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        predictions_dir=prediction["predictions_dir"],
        output_dir=stage_run_dir / "diagnosis",
        class_names=class_names,
        match_iou=0.5,
        localization_weak_iou=0.3,
    )
    write_markdown(stage_run_dir / "reports" / "diagnosis_report.md", build_stage_diagnosis_report(diagnosis))
    return diagnosis


def update_feedback_policy(
    *,
    policy_state: dict[str, Any],
    diagnosis: dict[str, Any],
    metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updated = deepcopy(policy_state)
    adjustments: list[dict[str, Any]] = []
    delta = metric_delta(metrics, reference_metrics)
    precision_drop = (delta.get("precision") or 0.0) < CONSTRAINT_DROP_LIMIT
    factor = 0.5 if precision_drop else 1.0
    flags = feedback_flags(diagnosis, metrics, reference_metrics)

    if flags["recall_low"] or flags["fn_high"]:
        adjustments.extend(adjust_yolo(updated, "hsv_v", 0.03 * factor, 0.25, 0.55, "recall_low_or_fn_high"))
        adjustments.extend(adjust_yolo(updated, "translate", 0.01 * factor, 0.04, 0.14, "recall_low_or_fn_high"))
        adjustments.extend(adjust_yolo(updated, "scale", 0.05 * factor, 0.30, 0.65, "recall_low_or_fn_high"))
        for name in ["clahe", "gamma", "sharpen_mild"]:
            adjustments.extend(adjust_custom(updated, name, 0.05 * factor, 0.04 * factor, "recall_low_or_fn_high"))

    if flags["precision_low"] or flags["fp_high"]:
        adjustments.extend(adjust_yolo(updated, "hsv_v", -0.04, 0.25, 0.55, "precision_low_or_fp_high"))
        for name in ["clahe", "gamma", "local_contrast"]:
            adjustments.extend(adjust_custom(updated, name, -0.04, -0.03, "precision_low_or_fp_high"))
        adjustments.extend(adjust_yolo(updated, "erasing", 0.04, 0.25, 0.50, "precision_low_hard_negative_like"))
        adjustments.extend(adjust_custom(updated, "cutout_safe", 0.04, 0.03, "precision_low_hard_negative_like"))

    if flags["map50_high_map95_low"]:
        adjustments.extend(adjust_yolo(updated, "translate", -0.02, 0.04, 0.14, "map50_high_map95_low_reduce_geometry"))
        adjustments.extend(adjust_yolo(updated, "scale", -0.05, 0.30, 0.65, "map50_high_map95_low_reduce_geometry"))
        adjustments.extend(adjust_yolo(updated, "mosaic", -0.10, 0.60, 1.0, "map50_high_map95_low_reduce_mosaic"))
        updated["official_yolo_aug"]["close_mosaic"] = 10
        for name in ["sharpen_mild", "local_contrast"]:
            adjustments.extend(adjust_custom(updated, name, 0.05, 0.04, "map50_high_map95_low"))

    if flags["low_contrast_fn_high"]:
        for name in ["clahe", "gamma", "local_contrast", "sharpen_mild"]:
            adjustments.extend(adjust_custom(updated, name, 0.05 * factor, 0.04 * factor, "low_contrast_fn_high_monitor_fp"))

    return updated, adjustments


def feedback_flags(diagnosis: dict[str, Any], metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> dict[str, bool]:
    global_diag = diagnosis.get("global", {}) or {}
    vector = diagnosis.get("diagnosis_vector", {}) or {}
    issues = {str(item.get("type")) for item in diagnosis.get("issues", []) if isinstance(item, dict)}
    delta = metric_delta(metrics, reference_metrics)
    fn = int(global_diag.get("fn", 0) or 0)
    fp = int(global_diag.get("fp", 0) or 0)
    tp = int(global_diag.get("tp", 0) or 0)
    low_contrast_score = float((vector.get("low_contrast_score") or {}).get("score", 0.0) or 0.0)
    return {
        "recall_low": (delta.get("recall") or 0.0) < -0.01,
        "fn_high": fn > 0 and fn / max(1, tp + fn) > 0.18,
        "precision_low": (delta.get("precision") or 0.0) < CONSTRAINT_DROP_LIMIT,
        "fp_high": fp > 0 and fp / max(1, tp + fp) > 0.25,
        "map50_high_map95_low": metrics.get("map50") is not None
        and metrics.get("map50_95") is not None
        and float(metrics["map50"]) - float(metrics["map50_95"]) > 0.18,
        "low_contrast_fn_high": low_contrast_score > 0.15 or "low_contrast_missed_defect" in issues,
    }


def adjust_yolo(state: dict[str, Any], key: str, delta: float, lower: float, upper: float, reason: str) -> list[dict[str, Any]]:
    aug = state["official_yolo_aug"]
    before = float(aug.get(key, OFFICIAL_YOLO_DEFAULT_AUG[key]))
    after = clip(before + delta, lower, upper)
    if after == before:
        return []
    aug[key] = after
    return [{"scope": "official_yolo_aug", "name": key, "field": "value", "before": before, "after": after, "reason": reason}]


def adjust_custom(state: dict[str, Any], name: str, prob_delta: float, strength_delta: float, reason: str) -> list[dict[str, Any]]:
    op = find_custom_op(state, name)
    min_prob, max_prob, min_strength, max_strength = CUSTOM_OP_LIMITS[name]
    changes: list[dict[str, Any]] = []
    before_prob = float(op.get("prob", 0.0) or 0.0)
    after_prob = clip(before_prob + prob_delta, min_prob, max_prob)
    if after_prob != before_prob:
        op["prob"] = after_prob
        changes.append({"scope": "custom_industrial_policy", "name": name, "field": "prob", "before": before_prob, "after": after_prob, "reason": reason})
    before_strength = float(op.get("strength", 0.0) or 0.0)
    after_strength = clip(before_strength + strength_delta, min_strength, max_strength)
    if after_strength != before_strength:
        op["strength"] = after_strength
        changes.append({"scope": "custom_industrial_policy", "name": name, "field": "strength", "before": before_strength, "after": after_strength, "reason": reason})
    return changes


def find_custom_op(state: dict[str, Any], name: str) -> dict[str, Any]:
    for op in state["custom_industrial_policy"].setdefault("operations", []):
        if op.get("name") == name:
            return op
    op = op_payload(name, 0.0, 0.0)
    state["custom_industrial_policy"]["operations"].append(op)
    return op


def build_history_record(*, stage_record: StageRecord, profile: str, adjustments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage_index": stage_record.stage_index,
        "profile": profile,
        "trigger_metrics": stage_record.val_metrics,
        "diagnosis_global": stage_record.diagnosis.get("global", {}),
        "diagnosis_vector": stage_record.diagnosis.get("diagnosis_vector", {}),
        "constraints": stage_record.constraints,
        "adjustments": adjustments,
        "policy_before": stage_record.policy_before,
        "policy_after": stage_record.policy_after,
        "copy_paste_status": "disabled_not_allowed_by_feedback_constraints",
    }


def evaluate_constraints(delta: dict[str, float | None]) -> dict[str, Any]:
    failures: list[str] = []
    if delta.get("precision") is not None and float(delta["precision"]) < CONSTRAINT_DROP_LIMIT:
        failures.append("precision_drop_gt_0.01")
    if delta.get("map50") is not None and float(delta["map50"]) < CONSTRAINT_DROP_LIMIT:
        failures.append("map50_drop_gt_0.01")
    if delta.get("map50_95") is not None and float(delta["map50_95"]) < CONSTRAINT_DROP_LIMIT:
        failures.append("map50_95_drop_gt_0.01")
    return {"passed": not failures, "constraint_failed": bool(failures), "failures": failures}


def build_constraint_scoring(final_metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> dict[str, Any]:
    delta = metric_delta(final_metrics, reference_metrics)
    constraints = evaluate_constraints(delta)
    recall_improved = (delta.get("recall") or 0.0) > 0.0
    return {
        "reference": "YOLO default seed=42",
        "final_metrics": final_metrics,
        "delta_vs_yolo_default_seed42": delta,
        "constraints": constraints,
        "final_strategy_accepted": bool(constraints["passed"] and recall_improved),
        "reason": "accepted_recall_gain_under_constraints"
        if constraints["passed"] and recall_improved
        else "constraint_failed_or_no_recall_gain",
    }


def build_final_payload(
    args: argparse.Namespace,
    output_dir: Path,
    reference: dict[str, Any],
    stages: list[StageRecord],
    history: list[dict[str, Any]],
    policy_state: dict[str, Any],
    final_metrics: dict[str, Any],
    constraint_scoring: dict[str, Any],
) -> dict[str, Any]:
    comparisons = build_reference_comparisons(final_metrics, reference)
    return {
        "run_id": args.run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "mode": "yolo_default_with_diagnosis_feedback",
        "base_augmentation": "Ultralytics YOLO default augmentation enabled for every stage.",
        "custom_yolo_like_base_status": "engineering_exploration_only_not_mainline",
        "fixed_augmented_dataset_generated": False,
        "model": args.model,
        "data": str(Path(args.data)),
        "epochs": int(args.epochs),
        "feedback_interval": int(args.feedback_interval),
        "stage_count": len(stages),
        "reference": reference,
        "stages": [stage_to_dict(stage) for stage in stages],
        "policy_history_count": len(history),
        "final_policy_state": policy_state,
        "final_metrics": final_metrics,
        "constraint_scoring": constraint_scoring,
        "comparisons": comparisons,
        "summary": {
            "entrypoint": "scripts/train_yolo_default_with_feedback.py",
            "smoke_mode": int(args.epochs) < 50,
            "train_success": all(stage.train_success for stage in stages),
            "stage_count": len(stages),
            "policy_history_updates": len(history),
            "yolo_default_enabled": True,
            "custom_mosaic4_used": False,
            "custom_randaugment_like_used": False,
            "fixed_augmented_dataset_generated": False,
            "final_metrics": final_metrics,
            "constraints": constraint_scoring["constraints"],
            "policy_history": str((output_dir / "reports/policy_history.json").resolve()),
            "constraint_scoring": str((output_dir / "reports/constraint_scoring.json").resolve()),
        },
    }


def stage_to_dict(stage: StageRecord) -> dict[str, Any]:
    payload = deepcopy(stage.__dict__)
    payload["run_dir"] = str(stage.run_dir)
    return payload


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    write_json(output_dir / "reports/final_metrics.json", payload)
    write_json(output_dir / "reports/constraint_scoring.json", payload["constraint_scoring"])
    write_json(output_dir / "reports/stage_metrics.json", build_stage_metrics_payload(payload))
    write_json(output_dir / "reports/reference_comparison.json", payload["comparisons"])
    report = build_report_markdown(payload)
    compare = build_compare_markdown(payload)
    write_markdown(output_dir / "reports/final_report.md", report)
    write_markdown(output_dir / "reports/compare_with_yolo_default_baseline_diagaug_random.md", compare)
    write_markdown(output_dir / "reports/yolo_default_feedback_smoke_report.md", report)


def build_stage_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": payload["run_id"],
        "stage_count": payload["stage_count"],
        "stages": [
            {
                "stage_index": stage["stage_index"],
                "start_epoch": stage["global_epoch_start"],
                "end_epoch": stage["global_epoch_end"],
                "epochs": stage["epochs"],
                "run_dir": stage["run_dir"],
                "input_checkpoint": stage["input_checkpoint"],
                "output_checkpoint": stage.get("weights_last") or stage.get("weights_best"),
                "train_command": stage["train_command"],
                "val_command": str(Path(stage["run_dir"]) / "configs" / "val_command.txt"),
                "val_metrics": stage["val_metrics"],
                "diagnosis_summary": str(Path(stage["run_dir"]) / "diagnosis" / "diagnosis_summary.md"),
                "feedback_update": stage["adjustments"],
                "old_policy": stage["policy_before"],
                "new_policy": stage["policy_after"],
                "constraint_status": stage["constraints"],
            }
            for stage in payload["stages"]
        ],
    }


def write_policy_history(history_dir: Path, history: list[dict[str, Any]], latest_policy: dict[str, Any]) -> None:
    write_json(history_dir / "policy_history.json", {"history": history, "latest_policy": latest_policy})
    lines = ["# YOLO Default Feedback Policy History", "", "| stage | adjustments | constraints |", "|---:|---:|---|"]
    for record in history:
        lines.append(
            f"| {record['stage_index']} | {len(record.get('adjustments', []))} | "
            f"{'failed' if record.get('constraints', {}).get('constraint_failed') else 'passed'} |"
        )
    lines.extend(["", "## Adjustments", ""])
    for record in history:
        lines.append(f"### Stage {record['stage_index']}")
        for adjustment in record.get("adjustments", []):
            lines.append(
                f"- `{adjustment['scope']}.{adjustment['name']}` {adjustment['field']}: "
                f"{float(adjustment['before']):.4f} -> {float(adjustment['after']):.4f} ({adjustment['reason']})"
            )
        if not record.get("adjustments"):
            lines.append("- No adjustment.")
    write_markdown(history_dir / "policy_history.md", "\n".join(lines))
    with (history_dir / "policy_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage_index", "scope", "name", "field", "before", "after", "reason"])
        writer.writeheader()
        for record in history:
            for adjustment in record.get("adjustments", []):
                writer.writerow({"stage_index": record["stage_index"], **adjustment})


def build_stage_diagnosis_report(diagnosis: dict[str, Any]) -> str:
    global_diag = diagnosis.get("global", {})
    issues = ", ".join(str(item.get("type")) for item in diagnosis.get("issues", []) if isinstance(item, dict))
    return "\n".join(
        [
            "# Stage Diagnosis",
            "",
            f"- TP: `{global_diag.get('tp', 0)}`",
            f"- FP: `{global_diag.get('fp', 0)}`",
            f"- FN: `{global_diag.get('fn', 0)}`",
            f"- Localization weak: `{global_diag.get('localization_weak', 0)}`",
            f"- Issues: `{issues}`",
        ]
    )


def build_report_markdown(payload: dict[str, Any]) -> str:
    scoring = payload["constraint_scoring"]
    is_smoke = bool(payload["summary"].get("smoke_mode"))
    title = "YOLO Default With Diagnosis Feedback Smoke" if is_smoke else "YOLO Default With Diagnosis Feedback 50 Epoch"
    final = payload.get("final_metrics", {})
    deltas = scoring.get("delta_vs_yolo_default_seed42", {})
    up, down = adjustment_summary(payload.get("policy_history_count", 0), payload.get("stages", []))
    stage_worse = detect_stage_degradation(payload.get("stages", []))
    lines = [
        f"# {title}",
        "",
        "- Base augmentation: Ultralytics YOLO default augmentation remains enabled.",
        "- Custom YOLO-like `mosaic4` / `randaugment_like`: `false`.",
        "- Fixed augmented dataset generated: `false`.",
        f"- Stage count: `{payload['stage_count']}`",
        f"- Policy history updates: `{payload['policy_history_count']}`",
        f"- Feedback 50 epoch completed: `{str((not is_smoke) and all(stage.get('train_success') for stage in payload.get('stages', []))).lower()}`",
        "",
        "## Final Metrics",
        "",
        "| Precision | Recall | mAP50 | mAP50-95 |",
        "|---:|---:|---:|---:|",
        metric_row(final),
        "",
        "## Constraint Scoring",
        "",
        f"- Accepted final strategy: `{str(scoring.get('final_strategy_accepted')).lower()}`",
        f"- Failures: `{', '.join(scoring.get('constraints', {}).get('failures', []))}`",
        f"- Delta vs YOLO default seed=42: P `{fmt(deltas.get('precision'), signed=True)}`, R `{fmt(deltas.get('recall'), signed=True)}`, mAP50 `{fmt(deltas.get('map50'), signed=True)}`, mAP50-95 `{fmt(deltas.get('map50_95'), signed=True)}`",
        "",
        "## Required Answers",
        "",
        f"- 1. Feedback 50 epoch success: `{str((not is_smoke) and all(stage.get('train_success') for stage in payload.get('stages', []))).lower()}`",
        f"- 2. Final P/R/mAP50/mAP50-95: `{fmt(final.get('precision'))}/{fmt(final.get('recall'))}/{fmt(final.get('map50'))}/{fmt(final.get('map50_95'))}`",
        f"- 3. Exceeds YOLO default: `{str(all((deltas.get(key) or 0.0) > 0.0 for key in METRIC_KEYS)).lower()}`",
        f"- 4. Recall improved without breaking P/mAP: `{str(scoring.get('final_strategy_accepted')).lower()}`",
        f"- 5. Increased parameters: `{', '.join(up) if up else 'none'}`",
        f"- 6. Decreased parameters: `{', '.join(down) if down else 'none'}`",
        f"- 7. Policy updates stable: `{str(not stage_worse['severe']).lower()}`",
        f"- 8. Stage degradation observed: `{str(stage_worse['observed']).lower()}`",
        f"- 9. Constraint failed: `{str(scoring.get('constraints', {}).get('constraint_failed')).lower()}`",
        f"- 10. Worthy as paper main method now: `{str(scoring.get('final_strategy_accepted')).lower()}`",
        "",
        "## Stage Summary",
        "",
        "| stage | epochs | P | R | mAP50 | mAP50-95 | adjustments |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in payload["stages"]:
        metrics = stage.get("val_metrics", {})
        lines.append(
            f"| {stage['stage_index']} | {stage['epochs']} | {fmt(metrics.get('precision'))} | "
            f"{fmt(metrics.get('recall'))} | {fmt(metrics.get('map50'))} | {fmt(metrics.get('map50_95'))} | "
            f"{len(stage.get('adjustments', []))} |"
        )
    lines.extend(["", "## Reference Comparison", "", comparison_table(payload["comparisons"])])
    return "\n".join(lines) + "\n"


def build_compare_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# YOLO Default Feedback vs References",
        "",
        "- Reference for constraints: YOLO default seed=42.",
        "- Constraint rule: fail if Precision, mAP50, or mAP50-95 drops by more than 0.01.",
        "",
        comparison_table(payload["comparisons"]),
        "",
        "## Conclusion",
        "",
        f"- Constraint failed: `{str(payload['constraint_scoring']['constraints'].get('constraint_failed')).lower()}`",
        f"- Final strategy accepted: `{str(payload['constraint_scoring'].get('final_strategy_accepted')).lower()}`",
    ]
    return "\n".join(lines) + "\n"


def adjustment_summary(_history_count: int, stages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    increased: list[str] = []
    decreased: list[str] = []
    for stage in stages:
        for adj in stage.get("adjustments", []):
            name = f"{adj.get('scope')}.{adj.get('name')}.{adj.get('field')}"
            before = float(adj.get("before", 0.0))
            after = float(adj.get("after", 0.0))
            if after > before and name not in increased:
                increased.append(name)
            if after < before and name not in decreased:
                decreased.append(name)
    return increased, decreased


def detect_stage_degradation(stages: list[dict[str, Any]]) -> dict[str, Any]:
    observed = False
    severe = False
    previous: dict[str, Any] | None = None
    drops: list[dict[str, Any]] = []
    for stage in stages:
        current = stage.get("val_metrics", {})
        if previous:
            delta = metric_delta(current, previous)
            bad = {
                key: value
                for key, value in delta.items()
                if value is not None and key in {"precision", "map50", "map50_95"} and value < -0.02
            }
            if bad:
                observed = True
                drops.append({"stage_index": stage.get("stage_index"), "delta": bad})
            if any(value < -0.05 for value in bad.values()):
                severe = True
        previous = current
    return {"observed": observed, "severe": severe, "drops": drops}


def comparison_table(comparisons: dict[str, Any]) -> str:
    lines = [
        "| reference | P | R | mAP50 | mAP50-95 | dP | dR | d_mAP50 | d_mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in comparisons.items():
        metrics = item.get("metrics", {})
        delta = item.get("delta", {})
        lines.append(
            f"| {label} | {fmt(metrics.get('precision'))} | {fmt(metrics.get('recall'))} | "
            f"{fmt(metrics.get('map50'))} | {fmt(metrics.get('map50_95'))} | "
            f"{fmt(delta.get('precision'), signed=True)} | {fmt(delta.get('recall'), signed=True)} | "
            f"{fmt(delta.get('map50'), signed=True)} | {fmt(delta.get('map50_95'), signed=True)} |"
        )
    return "\n".join(lines)


def metric_row(metrics: dict[str, Any]) -> str:
    return f"| {fmt(metrics.get('precision'))} | {fmt(metrics.get('recall'))} | {fmt(metrics.get('map50'))} | {fmt(metrics.get('map50_95'))} |"


def build_reference_comparisons(final_metrics: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    refs = {
        "YOLO default seed=42": reference["metrics"],
        "baseline no aug": load_reference_file(
            PROJECT_ROOT / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json"
        ),
        "offline DiagAug": load_reference_file(
            PROJECT_ROOT / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_metrics.json"
        ),
        "offline random external": load_reference_file(
            PROJECT_ROOT / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/random_external_aug_50ep_metrics.json"
        ),
    }
    return {
        name: {"metrics": metrics, "delta": metric_delta(final_metrics, metrics)}
        for name, metrics in refs.items()
        if metrics
    }


def load_reference_metrics(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    for row in payload.get("rows", []):
        if row.get("key") == "yolo_default":
            return {"source": str(path), "key": "yolo_default", "metrics": compact_metrics(row.get("metrics", {}))}
    raise ValueError(f"could not find yolo_default row in {path}")


def load_reference_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = read_json(path)
    metrics = (
        payload.get("final_metrics")
        or payload.get("validation", {}).get("overall")
        or payload.get("final", {}).get("overall")
        or payload.get("overall")
        or {}
    )
    return compact_metrics(metrics)


def resolve_val_image_label_dirs(data_yaml: Path) -> tuple[Path, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig")) or {}
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val_value = Path(str(data["val"]))
    val_images = val_value if val_value.is_absolute() else root / val_value
    try:
        rel_parts = list(val_images.relative_to(root).parts)
        if rel_parts and rel_parts[0] == "images":
            val_labels = root.joinpath("labels", *rel_parts[1:])
        else:
            val_labels = val_images.parent.parent / "labels" / val_images.name
    except ValueError:
        parts = list(val_images.parts)
        if "images" in parts:
            index = parts.index("images")
            parts[index] = "labels"
            val_labels = Path(*parts)
        else:
            val_labels = val_images.parent.parent / "labels" / val_images.name
    return val_images, val_labels


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in ["images", "instances", *METRIC_KEYS]}


def metric_delta(metrics: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        ref_value = reference.get(key)
        delta[key] = None if value is None or ref_value is None else float(value) - float(ref_value)
    return delta


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    write_text(path, text.rstrip() + "\n")


def update_marked_section(path: Path, marker: str, content: str) -> None:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    block = f"{start}\n{content.rstrip()}\n{end}"
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in original and end in original:
        before = original.split(start, 1)[0].rstrip()
        after = original.split(end, 1)[1].lstrip()
        text = f"{before}\n{block}\n{after}".rstrip() + "\n"
    else:
        text = original.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def update_state_docs(payload: dict[str, Any]) -> None:
    is_smoke = bool(payload["summary"].get("smoke_mode"))
    scope = "2-stage smoke; no formal 50 epoch run in this step." if is_smoke else "formal 50 epoch segmented feedback run."
    marker = "YOLO_DEFAULT_FEEDBACK_AUG_SMOKE" if is_smoke else "YOLO_DEFAULT_FEEDBACK_AUG_50EP_FULL"
    content = "\n".join(
        [
            "## YOLO Default Feedback Augmentation",
            "",
            "- Entrypoint: `scripts/train_yolo_default_with_feedback.py`.",
            "- Base: Ultralytics YOLO default augmentation remains enabled; custom YOLO-like `mosaic4` and `randaugment_like` are not used.",
            f"- Scope: {scope}",
            f"- Output: `outputs/experiments/{payload['run_id']}/`",
            f"- Stage count: `{payload['stage_count']}`",
            f"- Policy history updates: `{payload['policy_history_count']}`",
            f"- Fixed augmented dataset generated: `{str(payload['fixed_augmented_dataset_generated']).lower()}`",
            f"- Final P/R/mAP50/mAP50-95: `{fmt(payload['final_metrics'].get('precision'))}/{fmt(payload['final_metrics'].get('recall'))}/{fmt(payload['final_metrics'].get('map50'))}/{fmt(payload['final_metrics'].get('map50_95'))}`",
            f"- Constraint accepted: `{str(payload['constraint_scoring'].get('final_strategy_accepted')).lower()}`",
            f"- Report: `outputs/experiments/{payload['run_id']}/reports/final_report.md`",
            f"- Policy history: `outputs/experiments/{payload['run_id']}/reports/policy_history.json`",
        ]
    )
    for name in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
        update_marked_section(PROJECT_ROOT / name, marker, content)


def export_project_snapshot() -> None:
    subprocess.run([str(PYTHON_EXE), str(PROJECT_ROOT / "scripts/export_project_snapshot.py")], cwd=str(PROJECT_ROOT), check=False)


if __name__ == "__main__":
    main()
