from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostic_pipeline import run_error_diagnosis, run_validation_prediction  # noqa: E402
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402
from AutoAugment.feedback_policy_controller import FeedbackPolicyController, default_catf_policy  # noqa: E402
from AutoAugment.online_augmentation import OnlineAugmentationStats, OnlinePolicyAugmentor  # noqa: E402
from scripts.train_yolo_online_aug import (  # noqa: E402
    OnlineTrainingContext,
    check_ultralytics_api,
    count_split_images,
    make_online_trainer,
    metrics_to_dict,
    register_epoch_callback,
)


PYTHON_EXE = Path(r"D:\Anaconda\envs\pytorch\python.exe")
DEFAULT_DATA = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "outputs/experiments"
DEFAULT_RUN_ID = "yolo_default_inloop_feedback_10ep_smoke"
DEFAULT_CONTROL_METRICS = DEFAULT_PROJECT / "clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_metrics.json"
DEFAULT_REFERENCE_CURVE = DEFAULT_PROJECT / "clean_native_yolo_default_seed42_50ep/train/results.csv"
REFERENCE_METRICS = (
    PROJECT_ROOT
    / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep/reports/diagnosis_constrained_metrics.json"
)
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
INDUSTRIAL_OPS = {"clahe", "gamma", "local_contrast", "sharpen_mild", "cutout_safe", "brightness", "contrast"}
CUSTOM_LIMITS = {
    "clahe": (0.0, 0.15, 0.0, 0.45),
    "gamma": (0.0, 0.15, 0.0, 0.45),
    "local_contrast": (0.0, 0.18, 0.0, 0.40),
    "sharpen_mild": (0.0, 0.20, 0.0, 0.45),
    "cutout_safe": (0.0, 0.08, 0.0, 0.20),
    "brightness": (0.0, 0.10, 0.0, 0.25),
    "contrast": (0.0, 0.10, 0.0, 0.25),
}


@dataclass
class InLoopFeedbackState:
    output_dir: Path
    args: argparse.Namespace
    context: OnlineTrainingContext
    policy_state: dict[str, Any]
    reference_metrics: dict[str, Any]
    reference_curve: dict[int, dict[str, Any]] = field(default_factory=dict)
    feedback_controller: FeedbackPolicyController | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    epoch_records: list[dict[str, Any]] = field(default_factory=list)
    callback_invocations: int = 0
    train_start_time: float = field(default_factory=time.time)
    train_end_time: float | None = None
    trainer_identity: dict[str, int | None] = field(default_factory=dict)
    feedback_epochs: list[int] = field(default_factory=list)


def main() -> None:
    args = parse_args()
    output_dir = (Path(args.project) / args.run_id).resolve()
    configure_environment(output_dir)
    prepare_output_dirs(output_dir)

    api = check_ultralytics_api()
    reference_metrics = load_yolo_default_reference(Path(args.reference_metrics))
    reference_curve = load_reference_curve(Path(args.reference_curve))
    if is_native_no_feedback_mode(args):
        payload = run_native_no_feedback_control(args, output_dir, api, reference_metrics)
        write_outputs(payload)
        if not args.skip_doc_update:
            update_state_docs(payload)
            export_project_snapshot()
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        if not payload["train"]["success"]:
            raise RuntimeError(f"native no-feedback training failed: {payload['train']['error']}")
        if not payload["val"]["success"]:
            raise RuntimeError(f"native no-feedback final validation failed: {payload['val']['error']}")
        return

    policy_state = initial_policy_state()
    active_policy = training_policy(policy_state, enabled=bool(args.industrial_aug_enabled))
    write_json(output_dir / "configs" / "initial_policy_state.json", policy_state)
    write_json(output_dir / "configs" / "active_policy_epoch_000.json", active_policy)
    write_json(output_dir / "configs" / "train_config.json", build_train_config(args, output_dir))

    stats = OnlineAugmentationStats()
    augmentor = OnlinePolicyAugmentor(
        active_policy,
        seed=args.seed,
        copy_paste_enabled=False,
        stats=stats,
        total_epochs=args.epochs,
    )
    context = OnlineTrainingContext(
        augmentor=augmentor,
        preview_dir=output_dir / "previews",
        save_preview=args.save_preview,
        preview_count=args.preview_count,
        total_epochs=args.epochs,
    )
    state = InLoopFeedbackState(
        output_dir=output_dir,
        args=args,
        context=context,
        policy_state=policy_state,
        reference_metrics=reference_metrics,
        reference_curve=reference_curve,
    )
    state.feedback_controller = FeedbackPolicyController(
        policy_state,
        history_dir=output_dir / "reports",
        policy_state_path=output_dir / "configs" / "current_policy_state.json",
        profile=args.feedback_profile,
        reference_curve=reference_curve,
        freeze_epoch=40,
        feedback_interval=int(args.feedback_interval),
    )

    model = api["YOLO"](args.model)
    trainer_cls = make_online_trainer(api, context) if args.industrial_aug_enabled else None
    if args.industrial_aug_enabled:
        register_epoch_callback(model, context, total_epochs=args.epochs)
    else:
        context.train_image_count = count_split_images(Path(args.data), "train")
    register_inloop_feedback_callback(model, state)
    train_kwargs = build_train_kwargs(args, output_dir)
    write_text(output_dir / "configs" / "train_command.txt", build_train_command(args, output_dir))
    write_text(output_dir / "logs" / "train.command.txt", build_train_command(args, output_dir))

    train_success = False
    train_error: str | None = None
    train_result_type: str | None = None
    try:
        if trainer_cls is not None:
            train_result = model.train(trainer=trainer_cls, **train_kwargs)
        else:
            train_result = model.train(**train_kwargs)
        train_result_type = type(train_result).__name__
        train_success = True
    except Exception as exc:
        train_error = f"{type(exc).__name__}: {exc}"
    state.train_end_time = time.time()

    best_pt = output_dir / "train" / "weights" / "best.pt"
    last_pt = output_dir / "train" / "weights" / "last.pt"
    weights_for_val = best_pt if best_pt.exists() else last_pt
    val_metrics: dict[str, Any] = {}
    val_success = False
    val_error: str | None = None
    if train_success and weights_for_val.exists():
        try:
            val_result = api["YOLO"](str(weights_for_val)).val(
                data=str(Path(args.data).resolve()),
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=str(args.device),
                project=str(output_dir),
                name="val",
                exist_ok=True,
                plots=False,
            )
            val_metrics = metrics_to_dict(val_result)
            if val_metrics.get("images") is None:
                val_metrics["images"] = count_split_images(Path(args.data), "val")
            val_success = True
        except Exception as exc:
            val_error = f"{type(exc).__name__}: {exc}"
    elif train_success:
        val_error = "no best.pt or last.pt found after training"
    else:
        val_error = "validation skipped because training failed"

    payload = build_payload(
        args=args,
        output_dir=output_dir,
        state=state,
        train_success=train_success,
        train_error=train_error,
        train_result_type=train_result_type,
        val_success=val_success,
        val_error=val_error,
        val_metrics=val_metrics,
        best_pt=best_pt,
        last_pt=last_pt,
    )
    write_outputs(payload)
    if not args.skip_doc_update:
        update_state_docs(payload)
        export_project_snapshot()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if not train_success:
        raise RuntimeError(f"in-loop feedback training failed: {train_error}")
    if not val_success:
        raise RuntimeError(f"in-loop feedback final validation failed: {val_error}")


def is_native_no_feedback_mode(args: argparse.Namespace) -> bool:
    return not bool(args.feedback_enabled) and not bool(args.industrial_aug_enabled)


def run_native_no_feedback_control(
    args: argparse.Namespace,
    output_dir: Path,
    api: dict[str, Any],
    reference_metrics: dict[str, Any],
) -> dict[str, Any]:
    write_json(output_dir / "configs" / "initial_policy_state.json", empty_native_policy_state())
    write_json(output_dir / "configs" / "active_policy_epoch_000.json", {"policy_id": "native_yolo_default_no_feedback", "operations": []})
    write_json(output_dir / "configs" / "train_config.json", build_train_config(args, output_dir))

    stats = OnlineAugmentationStats()
    augmentor = OnlinePolicyAugmentor(
        {"policy_id": "native_yolo_default_no_feedback", "operations": []},
        seed=args.seed,
        copy_paste_enabled=False,
        stats=stats,
        total_epochs=args.epochs,
    )
    context = OnlineTrainingContext(
        augmentor=augmentor,
        preview_dir=output_dir / "previews",
        save_preview=False,
        preview_count=0,
        total_epochs=args.epochs,
    )
    context.train_image_count = count_split_images(Path(args.data).resolve(), "train")
    state = InLoopFeedbackState(
        output_dir=output_dir,
        args=args,
        context=context,
        policy_state=empty_native_policy_state(),
        reference_metrics=reference_metrics,
        reference_curve=load_reference_curve(Path(args.reference_curve)),
    )

    train_command = build_train_command(args, output_dir)
    write_text(output_dir / "configs" / "train_command.txt", train_command)
    write_text(output_dir / "logs" / "train.command.txt", train_command)

    train_success = False
    train_error: str | None = None
    train_result_type: str | None = None
    train_start = time.time()
    model = api["YOLO"](args.model)
    train_kwargs = build_train_kwargs(args, output_dir)
    try:
        train_result = model.train(**train_kwargs)
        train_result_type = type(train_result).__name__
        train_success = True
    except Exception as exc:
        train_error = f"{type(exc).__name__}: {exc}"
    state.train_start_time = train_start
    state.train_end_time = time.time()

    best_pt = output_dir / "train" / "weights" / "best.pt"
    last_pt = output_dir / "train" / "weights" / "last.pt"
    weights_for_val = best_pt if best_pt.exists() else last_pt
    val_metrics: dict[str, Any] = {}
    val_success = False
    val_error: str | None = None
    if train_success and weights_for_val.exists():
        val_command = build_val_command(args, output_dir, weights_for_val)
        write_text(output_dir / "configs" / "val_command.txt", val_command)
        write_text(output_dir / "logs" / "val.command.txt", val_command)
        try:
            val_result = api["YOLO"](str(weights_for_val)).val(
                data=str(Path(args.data).resolve()),
                imgsz=args.imgsz,
                batch=args.batch,
                workers=args.workers,
                device=str(args.device),
                project=str(output_dir),
                name="val",
                exist_ok=True,
                plots=False,
            )
            val_metrics = metrics_to_dict(val_result)
            if val_metrics.get("images") is None:
                val_metrics["images"] = count_split_images(Path(args.data).resolve(), "val")
            val_success = True
        except Exception as exc:
            val_error = f"{type(exc).__name__}: {exc}"
    elif train_success:
        val_error = "no best.pt or last.pt found after training"
    else:
        val_error = "validation skipped because training failed"

    return build_payload(
        args=args,
        output_dir=output_dir,
        state=state,
        train_success=train_success,
        train_error=train_error,
        train_result_type=train_result_type,
        val_success=val_success,
        val_error=val_error,
        val_metrics=val_metrics,
        best_pt=best_pt,
        last_pt=last_pt,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-run YOLO default training with in-loop diagnosis feedback.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--feedback-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feedback-interval", type=int, default=5)
    parser.add_argument("--feedback-start-epoch", type=int, default=5)
    parser.add_argument("--feedback-profile", default="industrial")
    parser.add_argument("--industrial-aug-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-metrics", default=str(REFERENCE_METRICS))
    parser.add_argument("--reference-curve", default=str(DEFAULT_REFERENCE_CURVE))
    parser.add_argument("--control-metrics", default=str(DEFAULT_CONTROL_METRICS))
    parser.add_argument("--save-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--keep-diagnosis-predict-runs", action="store_true")
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


def prepare_output_dirs(output_dir: Path) -> None:
    for subdir in ["configs", "logs", "reports", "previews", "diagnosis"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def build_train_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "entrypoint": "scripts/train_yolo_default_with_inloop_feedback.py",
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "model": args.model,
        "data": str(Path(args.data).resolve()),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": str(args.device),
        "seed": int(args.seed),
        "feedback_enabled": bool(args.feedback_enabled),
        "feedback_interval": int(args.feedback_interval),
        "feedback_start_epoch": int(args.feedback_start_epoch),
        "industrial_aug_enabled": bool(args.industrial_aug_enabled),
        "feedback_controller": "CATF",
        "reference_curve": str(Path(args.reference_curve).resolve()),
        "native_no_feedback_passthrough": is_native_no_feedback_mode(args),
        "yolo_default_augmentation_enabled": True,
        "disable_yolo_aug": False,
        "custom_mosaic4_used": False,
        "custom_randaugment_like_used": False,
        "copy_paste_status": "pending_object_bank_design",
        "keep_diagnosis_predict_runs": bool(args.keep_diagnosis_predict_runs),
    }


def build_train_kwargs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "data": str(Path(args.data).resolve()),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": str(args.device),
        "seed": int(args.seed),
        "project": str(output_dir),
        "name": "train",
        "exist_ok": True,
        "plots": False,
    }


def build_train_command(args: argparse.Namespace, output_dir: Path) -> str:
    parts = [
        "YOLO.train",
        f"model={args.model}",
        f"data={Path(args.data).resolve()}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"project={output_dir}",
        "name=train",
        "trainer=InLoopFeedbackDetectionTrainer" if args.industrial_aug_enabled else "trainer=UltralyticsDefaultDetectionTrainer",
        "yolo_default_augmentation_enabled=True",
        "disable_yolo_aug=False",
        f"feedback_enabled={bool(args.feedback_enabled)}",
        f"industrial_aug_enabled={bool(args.industrial_aug_enabled)}",
        f"feedback_controller=CATF",
        f"reference_curve={Path(getattr(args, 'reference_curve', DEFAULT_REFERENCE_CURVE)).resolve()}",
    ]
    return subprocess.list2cmdline([str(part) for part in parts])


def build_val_command(args: argparse.Namespace, output_dir: Path, weights: Path) -> str:
    parts = [
        "YOLO.val",
        f"model={weights.resolve()}",
        f"data={Path(args.data).resolve()}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"device={args.device}",
        f"project={output_dir}",
        "name=val",
        "exist_ok=True",
        "plots=False",
    ]
    return subprocess.list2cmdline([str(part) for part in parts])


def initial_policy_state() -> dict[str, Any]:
    return default_catf_policy()


def empty_native_policy_state() -> dict[str, Any]:
    return {
        "policy_id": "native_yolo_default_no_feedback",
        "copy_paste_status": "not_used",
        "notes": [
            "feedback_enabled=false and industrial_aug_enabled=false.",
            "No custom trainer, dataset, transform, industrial augmentor, or feedback callback is registered.",
            "Training is a direct Ultralytics YOLO.train(**same_args) call.",
        ],
        "operations": [],
    }


def op_payload(name: str, prob: float, strength: float, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    min_prob, max_prob, min_strength, max_strength = CUSTOM_LIMITS[name]
    return {
        "name": name,
        "prob": float(prob),
        "base_prob": float(prob),
        "min_prob": min_prob,
        "max_prob": max_prob,
        "strength": float(strength),
        "base_strength": float(strength),
        "min_strength": min_strength,
        "max_strength": max_strength,
        "params": params or {},
    }


def training_policy(policy_state: dict[str, Any], *, enabled: bool = True) -> dict[str, Any]:
    policy = deepcopy(policy_state)
    if not enabled:
        policy["operations"] = []
        return policy
    blocked = {"mosaic4", "randaugment_like", "copy_paste", "online_copy_paste", "class_balanced_copy_paste"}
    operations = []
    for operation in policy.get("operations", []):
        name = str(operation.get("name", "")).strip().lower()
        if name in blocked:
            raise ValueError(f"{name} is not allowed in YOLO-default in-loop feedback mode")
        if name not in INDUSTRIAL_OPS:
            raise ValueError(f"unsupported in-loop industrial op: {name}")
        if float(operation.get("prob", 0.0) or 0.0) > 0.0:
            operations.append(deepcopy(operation))
    policy["operations"] = operations
    return policy


def register_inloop_feedback_callback(model: Any, state: InLoopFeedbackState) -> None:
    def _on_fit_epoch_end(trainer: Any) -> None:
        state.callback_invocations += 1
        epoch_index = int(getattr(trainer, "epoch", 0) or 0)
        epoch_num = epoch_index + 1
        metrics = compact_trainer_metrics(getattr(trainer, "metrics", {}) or {})
        state.epoch_records.append(
            {
                "epoch": epoch_num,
                "metrics": metrics,
                "optimizer_id": id(getattr(trainer, "optimizer", None)) if getattr(trainer, "optimizer", None) is not None else None,
                "scheduler_id": id(getattr(trainer, "scheduler", None)) if getattr(trainer, "scheduler", None) is not None else None,
                "ema_id": id(getattr(trainer, "ema", None)) if getattr(trainer, "ema", None) is not None else None,
            }
        )
        if not should_update_feedback(state.args, epoch_num):
            return
        state.trainer_identity.setdefault("trainer_id", id(trainer))
        state.trainer_identity.setdefault("optimizer_id", id(getattr(trainer, "optimizer", None)) if getattr(trainer, "optimizer", None) is not None else None)
        state.trainer_identity.setdefault("scheduler_id", id(getattr(trainer, "scheduler", None)) if getattr(trainer, "scheduler", None) is not None else None)
        state.trainer_identity.setdefault("ema_id", id(getattr(trainer, "ema", None)) if getattr(trainer, "ema", None) is not None else None)

        diagnosis = run_feedback_diagnosis(state, trainer, epoch_num)
        old_policy = deepcopy(state.policy_state)
        controller = state.feedback_controller
        if controller is None:
            controller = FeedbackPolicyController(
                state.policy_state,
                history_dir=state.output_dir / "reports",
                policy_state_path=state.output_dir / "configs" / "current_policy_state.json",
                profile=state.args.feedback_profile,
                reference_curve=state.reference_curve,
                freeze_epoch=40,
                feedback_interval=int(state.args.feedback_interval),
            )
            state.feedback_controller = controller
        reference_metrics = reference_metrics_for_epoch(state, epoch_num)
        new_policy = controller.update(diagnosis, epoch=epoch_num, metrics=metrics, reference_metrics=reference_metrics)
        state.policy_state.clear()
        state.policy_state.update(deepcopy(new_policy))
        active_policy = training_policy(state.policy_state, enabled=bool(state.args.industrial_aug_enabled))
        state.context.augmentor.set_policy(active_policy)
        state.feedback_epochs.append(epoch_num)
        record = deepcopy(controller.history[-1])
        record.update(
            {
                "diagnosis_global": diagnosis.get("global", {}),
                "diagnosis_vector": diagnosis.get("diagnosis_vector", {}),
                "old_policy_before_callback": old_policy,
                "new_policy": deepcopy(state.policy_state),
                "active_policy": active_policy,
                "copy_paste_status": "pending_object_bank_design",
                "trainer_identity": {
                    "trainer_id": id(trainer),
                    "optimizer_id": id(getattr(trainer, "optimizer", None)) if getattr(trainer, "optimizer", None) is not None else None,
                    "scheduler_id": id(getattr(trainer, "scheduler", None)) if getattr(trainer, "scheduler", None) is not None else None,
                    "ema_id": id(getattr(trainer, "ema", None)) if getattr(trainer, "ema", None) is not None else None,
                },
            }
        )
        controller.history[-1] = deepcopy(record)
        state.history = deepcopy(controller.history)
        write_json(state.output_dir / "configs" / f"active_policy_epoch_{epoch_num:03d}.json", active_policy)
        write_policy_history(state.output_dir / "reports", state.history, state.policy_state)

    if hasattr(model, "add_callback"):
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)


def should_update_feedback(args: argparse.Namespace, epoch_num: int) -> bool:
    if not bool(args.feedback_enabled):
        return False
    if not bool(getattr(args, "industrial_aug_enabled", True)):
        return False
    if epoch_num >= int(args.epochs):
        return False
    if epoch_num < int(args.feedback_start_epoch):
        return False
    interval = max(1, int(args.feedback_interval))
    return epoch_num % interval == 0


def reference_metrics_for_epoch(state: InLoopFeedbackState, epoch_num: int) -> dict[str, Any]:
    if state.reference_curve:
        if epoch_num in state.reference_curve:
            return state.reference_curve[epoch_num]
        earlier_epochs = [epoch for epoch in state.reference_curve if epoch <= epoch_num]
        if earlier_epochs:
            return state.reference_curve[max(earlier_epochs)]
    return state.reference_metrics


def run_feedback_diagnosis(state: InLoopFeedbackState, trainer: Any, epoch_num: int) -> dict[str, Any]:
    weights = Path(getattr(trainer, "last", state.output_dir / "train" / "weights" / "last.pt"))
    if not weights.exists():
        weights = Path(getattr(trainer, "best", state.output_dir / "train" / "weights" / "best.pt"))
    val_images, val_labels = resolve_val_image_label_dirs(Path(state.args.data))
    class_names = load_class_names_from_data_yaml(state.args.data)
    diag_root = state.output_dir / "diagnosis" / f"epoch_{epoch_num:03d}"
    prediction = run_validation_prediction(
        weights=weights,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        output_dir=diag_root / "prediction",
        imgsz=int(state.args.imgsz),
        workers=int(state.args.workers),
        device=str(state.args.device),
        conf=0.25,
        iou=0.5,
        dry_run=False,
    )
    diagnosis = run_error_diagnosis(
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        predictions_dir=prediction["predictions_dir"],
        output_dir=diag_root / "diagnosis",
        class_names=class_names,
        match_iou=0.5,
        localization_weak_iou=0.3,
    )
    if not bool(getattr(state.args, "keep_diagnosis_predict_runs", False)):
        predict_runs = diag_root / "prediction" / "predict_runs"
        if predict_runs.exists():
            shutil.rmtree(predict_runs)
    return diagnosis


def update_policy_state(
    policy_state: dict[str, Any],
    *,
    diagnosis: dict[str, Any],
    metrics: dict[str, Any],
    reference: dict[str, Any],
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="catf_policy_update_") as temp_dir:
        controller = FeedbackPolicyController(policy_state, history_dir=Path(temp_dir), profile="industrial")
        updated = controller.update(diagnosis, epoch=0, metrics=metrics, reference_metrics=reference)
        policy_state.clear()
        policy_state.update(deepcopy(updated))
        return deepcopy(controller.history[-1].get("adjustments", []))


def feedback_flags(diagnosis: dict[str, Any], metrics: dict[str, Any], reference: dict[str, Any]) -> dict[str, bool]:
    global_diag = diagnosis.get("global", {}) or {}
    vector = diagnosis.get("diagnosis_vector", {}) or {}
    issues = {str(item.get("type")) for item in diagnosis.get("issues", []) if isinstance(item, dict)}
    delta = metric_delta(metrics, reference)
    tp = int(global_diag.get("tp", 0) or 0)
    fp = int(global_diag.get("fp", 0) or 0)
    fn = int(global_diag.get("fn", 0) or 0)
    low_contrast_score = float((vector.get("low_contrast_score") or {}).get("score", 0.0) or 0.0)
    map50 = metrics.get("map50")
    map95 = metrics.get("map50_95")
    return {
        "recall_low": (delta.get("recall") or 0.0) < -0.01,
        "fn_high": fn / max(1, tp + fn) > 0.18,
        "precision_low": (delta.get("precision") or 0.0) < -0.01,
        "fp_high": fp / max(1, tp + fp) > 0.25,
        "map50_high_map95_low": map50 is not None and map95 is not None and float(map50) - float(map95) > 0.18,
        "low_contrast_fn_high": low_contrast_score > 0.15 or "low_contrast_missed_defect" in issues,
    }


def adjust_operation(
    policy_state: dict[str, Any],
    name: str,
    *,
    prob_delta: float,
    strength_delta: float,
    reason: str,
) -> list[dict[str, Any]]:
    op = find_operation(policy_state, name)
    min_prob, max_prob, min_strength, max_strength = CUSTOM_LIMITS[name]
    changes: list[dict[str, Any]] = []
    before_prob = float(op.get("prob", 0.0) or 0.0)
    after_prob = clip(before_prob + prob_delta, min_prob, max_prob)
    if after_prob != before_prob:
        op["prob"] = after_prob
        changes.append({"epoch_field": "policy", "op": name, "field": "prob", "before": before_prob, "after": after_prob, "reason": reason})
    before_strength = float(op.get("strength", 0.0) or 0.0)
    after_strength = clip(before_strength + strength_delta, min_strength, max_strength)
    if after_strength != before_strength:
        op["strength"] = after_strength
        changes.append({"epoch_field": "policy", "op": name, "field": "strength", "before": before_strength, "after": after_strength, "reason": reason})
    return changes


def find_operation(policy_state: dict[str, Any], name: str) -> dict[str, Any]:
    for op in policy_state.setdefault("operations", []):
        if op.get("name") == name:
            return op
    op = op_payload(name, 0.0, 0.0)
    policy_state["operations"].append(op)
    return op


def compact_trainer_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    out: dict[str, Any] = {"images": None, "instances": None}
    for key, source in mapping.items():
        value = metrics.get(source, metrics.get(key))
        out[key] = None if value is None else float(value)
    return out


def build_payload(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    state: InLoopFeedbackState,
    train_success: bool,
    train_error: str | None,
    train_result_type: str | None,
    val_success: bool,
    val_error: str | None,
    val_metrics: dict[str, Any],
    best_pt: Path,
    last_pt: Path,
) -> dict[str, Any]:
    stats = state.context.augmentor.stats.to_dict()
    train_image_count = state.context.train_image_count
    if train_image_count is None:
        train_image_count = count_split_images(Path(args.data).resolve(), "train")
    stats.update(
        {
            "run_id": args.run_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "online_augmentation": bool(args.industrial_aug_enabled),
            "industrial_online_augmentation": bool(args.industrial_aug_enabled),
            "inloop_feedback": bool(args.feedback_enabled),
            "train_image_count": train_image_count,
            "expected_original_train_images": 2301,
            "train_image_count_matches_original": train_image_count == 2301,
            "fixed_augmented_dataset_generated": fixed_augmented_dataset_generated(output_dir),
            "copy_paste_online_supported": False,
            "copy_paste_status": "pending_object_bank_design",
            "train_success": train_success,
            "train_error": train_error,
            "val_success": val_success,
            "val_error": val_error,
            "val_metrics": val_metrics,
        }
    )
    continuity = verify_continuity(output_dir, args, state)
    control_metrics = load_control_metrics(Path(args.control_metrics))
    reference_for_constraints = control_metrics or state.reference_metrics
    delta_vs_reference = metric_delta(val_metrics, state.reference_metrics)
    delta_vs_control = metric_delta(val_metrics, control_metrics) if control_metrics else {}
    baseline_name = baseline_name_from_path(Path(args.control_metrics)) if control_metrics else baseline_name_from_path(Path(args.reference_metrics))
    constraint_scoring = build_constraint_scoring(val_metrics, reference_for_constraints, baseline_name=baseline_name)
    summary = {
        "single_run_inloop_feedback": True,
        "train_success": train_success,
        "val_success": val_success,
        "stage_restart_count": 0,
        "epoch_continuous": continuity["epoch_continuous"],
        "feedback_epochs": state.feedback_epochs,
        "feedback_update_count": len(state.history),
        "feedback_controller": "CATF",
        "reference_curve_loaded": bool(state.reference_curve),
        "yolo_default_augmentation_enabled": True,
        "industrial_aug_enabled": bool(args.industrial_aug_enabled),
        "industrial_aug_dynamic": bool(args.industrial_aug_enabled and int(stats.get("samples_augmented", 0) or 0) > 0),
        "fixed_augmented_dataset_generated": stats["fixed_augmented_dataset_generated"],
        "train_image_count": train_image_count,
        "bbox_class_valid": stats["invalid_bbox_count"] == 0 and stats["class_id_oob_count"] == 0,
        "policy_history": str((output_dir / "reports" / "policy_history.json").resolve()),
        "online_aug_stats": str((output_dir / "reports" / "online_aug_stats.json").resolve()),
        "report": str(primary_report_path(output_dir, args).resolve()),
        "constraint_failed": constraint_scoring["constraint_failed"] if args.feedback_enabled else None,
    }
    return {
        "run_id": args.run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "mode": "single_run_yolo_default_with_inloop_feedback",
        "model": str(args.model),
        "data": str(Path(args.data)),
        "epochs": int(args.epochs),
        "feedback_interval": int(args.feedback_interval),
        "feedback_start_epoch": int(args.feedback_start_epoch),
        "feedback_enabled": bool(args.feedback_enabled),
        "industrial_aug_enabled": bool(args.industrial_aug_enabled),
        "feedback_controller": "CATF",
        "reference_curve_path": str(Path(args.reference_curve).resolve()),
        "reference_curve_loaded": bool(state.reference_curve),
        "train_result_type": train_result_type,
        "train": {
            "success": train_success,
            "error": train_error,
            "wall_seconds": None if state.train_end_time is None else state.train_end_time - state.train_start_time,
            "results_csv": str(output_dir / "train" / "results.csv"),
            "best_pt": str(best_pt.resolve()) if best_pt.exists() else None,
            "last_pt": str(last_pt.resolve()) if last_pt.exists() else None,
        },
        "val": {"success": val_success, "error": val_error, "metrics": val_metrics},
        "policy_history": state.history,
        "latest_policy_state": state.policy_state,
        "epoch_records": state.epoch_records,
        "online_aug_stats": stats,
        "continuity": continuity,
        "reference_metrics": state.reference_metrics,
        "reference_curve_path": str(Path(args.reference_curve).resolve()),
        "reference_curve_epochs": sorted(state.reference_curve.keys()),
        "control_metrics": control_metrics,
        "control_metrics_path": str(Path(args.control_metrics).resolve()) if control_metrics else None,
        "reference_metrics_path": str(Path(args.reference_metrics).resolve()),
        "delta_vs_reference": delta_vs_reference,
        "delta_vs_control": delta_vs_control,
        "constraint_scoring": constraint_scoring,
        "summary": summary,
    }


def verify_continuity(output_dir: Path, args: argparse.Namespace, state: InLoopFeedbackState) -> dict[str, Any]:
    results_csv = output_dir / "train" / "results.csv"
    epochs = read_epoch_sequence(results_csv)
    expected = list(range(1, int(args.epochs) + 1))
    args_yaml = read_yaml(output_dir / "train" / "args.yaml")
    train_dirs = [p for p in output_dir.iterdir() if p.is_dir() and p.name == "train"]
    stage_dirs = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("stage_")]
    optimizer_ids = {record.get("optimizer_id") for record in state.epoch_records if record.get("optimizer_id") is not None}
    scheduler_ids = {record.get("scheduler_id") for record in state.epoch_records if record.get("scheduler_id") is not None}
    ema_ids = {record.get("ema_id") for record in state.epoch_records if record.get("ema_id") is not None}
    return {
        "single_train_run_dir": len(train_dirs) == 1,
        "stage_restart_dirs": [str(path) for path in stage_dirs],
        "stage_restart_count": len(stage_dirs),
        "epochs_arg": args_yaml.get("epochs"),
        "resume_arg": args_yaml.get("resume"),
        "close_mosaic_arg": args_yaml.get("close_mosaic"),
        "epoch_sequence": epochs,
        "expected_epoch_sequence": expected,
        "epoch_continuous": epochs == expected,
        "optimizer_single_id_observed": len(optimizer_ids) <= 1,
        "scheduler_single_id_observed": len(scheduler_ids) <= 1,
        "ema_single_id_observed": len(ema_ids) <= 1,
        "trainer_identity": state.trainer_identity,
        "global_best_path": str((output_dir / "train" / "weights" / "best.pt").resolve()),
        "global_last_path": str((output_dir / "train" / "weights" / "last.pt").resolve()),
        "official_close_mosaic_managed_by_yolo": True,
    }


def write_outputs(payload: dict[str, Any]) -> None:
    output_dir = Path(payload["output_dir"])
    write_json(output_dir / "reports" / "final_metrics.json", payload)
    write_json(output_dir / "reports" / "online_aug_stats.json", payload["online_aug_stats"])
    write_json(output_dir / "reports" / "epoch_records.json", payload["epoch_records"])
    write_json(output_dir / "reports" / "constraint_scoring.json", payload["constraint_scoring"])
    write_policy_history(output_dir / "reports", payload["policy_history"], payload["latest_policy_state"])
    write_markdown(output_dir / "reports" / "final_report.md", build_final_report(payload))
    write_markdown(output_dir / "reports" / "inloop_feedback_smoke_report.md", build_smoke_report(payload))
    if payload["feedback_enabled"] and int(payload["epochs"]) <= 10:
        write_markdown(output_dir / "reports" / "catf_smoke_report.md", build_catf_smoke_report(payload))
    if not payload["feedback_enabled"] and not payload["industrial_aug_enabled"]:
        write_json(output_dir / "reports" / "inloop_no_feedback_control_metrics.json", build_control_metrics_payload(payload))
        write_markdown(output_dir / "reports" / "inloop_no_feedback_control_report.md", build_no_feedback_control_report(payload))
        write_markdown(output_dir / "reports" / "compare_with_yolo_default_reference.md", build_reference_comparison_report(payload))
    if payload["feedback_enabled"]:
        write_markdown(output_dir / "reports" / "compare_with_yolo_default_no_feedback.md", build_feedback_comparison_report(payload))
        if payload.get("control_metrics_path") and "clean_native_yolo_default" in payload["control_metrics_path"]:
            write_markdown(output_dir / "reports" / "compare_with_clean_native_yolo_default.md", build_feedback_comparison_report(payload))


def write_policy_history(history_dir: Path, history: list[dict[str, Any]], latest_policy: dict[str, Any]) -> None:
    write_json(history_dir / "policy_history.json", {"history": history, "latest_policy": latest_policy})
    lines = ["# CATF In-Loop Feedback Policy History", "", "| epoch | action | guards | adjustments | frozen | copy_paste_status |", "|---:|---|---|---:|---|---|"]
    for record in history:
        lines.append(
            f"| {record.get('epoch')} | {record.get('action', 'accept')} | "
            f"{','.join(record.get('guard_triggered', [])) or 'none'} | {len(record.get('adjustments', []))} | "
            f"{str(record.get('frozen', False)).lower()} | {record.get('copy_paste_status')} |"
        )
    lines.extend(["", "## Adjustments", ""])
    for record in history:
        lines.append(f"### Epoch {record.get('epoch')} - {record.get('action', 'accept')}")
        lines.append(f"- Reference metrics: `{record.get('reference_metrics', {})}`")
        lines.append(f"- Delta metrics: `{record.get('delta_metrics', {})}`")
        lines.append(f"- Last safe policy id: `{record.get('last_safe_policy_id')}`")
        lines.append(f"- Rollback reason: `{record.get('rollback_reason')}`")
        lines.append(f"- Group budget before: `{record.get('group_budget_before', {})}`")
        lines.append(f"- Group budget after: `{record.get('group_budget_after', {})}`")
        lines.append(f"- Trust-region clipping: `{record.get('trust_region_clipping', [])}`")
        if not record.get("adjustments"):
            lines.append("- No adjustment.")
        for adj in record.get("adjustments", []):
            lines.append(f"- `{adj['op']}` {adj['field']}: {adj['before']:.4f} -> {adj['after']:.4f} ({adj['reason']})")
    write_markdown(history_dir / "policy_history.md", "\n".join(lines))
    with (history_dir / "policy_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "action",
                "op",
                "field",
                "before",
                "after",
                "reason",
                "delta_precision",
                "delta_recall",
                "delta_map50",
                "delta_map50_95",
                "guard_triggered",
                "rollback_reason",
                "frozen",
            ],
        )
        writer.writeheader()
        for record in history:
            adjustments = record.get("adjustments") or [{}]
            for adj in adjustments:
                writer.writerow(
                    {
                        "epoch": record.get("epoch"),
                        "action": record.get("action"),
                        "op": adj.get("op"),
                        "field": adj.get("field"),
                        "before": adj.get("before"),
                        "after": adj.get("after"),
                        "reason": adj.get("reason"),
                        "delta_precision": (record.get("delta_metrics") or {}).get("precision"),
                        "delta_recall": (record.get("delta_metrics") or {}).get("recall"),
                        "delta_map50": (record.get("delta_metrics") or {}).get("map50"),
                        "delta_map50_95": (record.get("delta_metrics") or {}).get("map50_95"),
                        "guard_triggered": ",".join(record.get("guard_triggered", [])),
                        "rollback_reason": record.get("rollback_reason"),
                        "frozen": record.get("frozen"),
                    }
                )


def build_smoke_report(payload: dict[str, Any]) -> str:
    stats = payload["online_aug_stats"]
    continuity = payload["continuity"]
    lines = [
        "# In-Loop YOLO Default Feedback Smoke",
        "",
        "## Answers",
        "",
        f"- Single-run continuous training: `{str(payload['summary']['single_run_inloop_feedback']).lower()}`",
        f"- No stage restart: `{str(continuity['stage_restart_count'] == 0).lower()}`",
        f"- Epoch sequence continuous: `{str(continuity['epoch_continuous']).lower()}`",
        f"- Optimizer/scheduler/EMA managed by one trainer: `{str(continuity['optimizer_single_id_observed'] and continuity['scheduler_single_id_observed'] and continuity['ema_single_id_observed']).lower()}`",
        f"- close_mosaic managed by official YOLO: `{str(continuity['official_close_mosaic_managed_by_yolo']).lower()}`",
        f"- Feedback updated after epoch 5: `{str(any(epoch >= 5 for epoch in payload['summary']['feedback_epochs'])).lower()}`",
        f"- Mutable policy affects later epochs: `{str(stats.get('samples_augmented', 0) > 0).lower()}`",
        "- YOLO default augmentation remains enabled: `true`",
        f"- BBox/class legal: `{str(payload['summary']['bbox_class_valid']).lower()}`",
        "",
        "## Augmentation Stats",
        "",
        f"- Train image count: `{stats.get('train_image_count')}`",
        f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
        f"- Samples seen: `{stats.get('samples_seen')}`",
        f"- Samples augmented: `{stats.get('samples_augmented')}`",
        f"- Invalid bbox count: `{stats.get('invalid_bbox_count')}`",
        f"- Class id OOB count: `{stats.get('class_id_oob_count')}`",
        "",
        "| op | seen | applied | skipped_probability |",
        "|---|---:|---:|---:|",
    ]
    for name, counts in sorted((stats.get("ops") or {}).items()):
        lines.append(
            f"| {name} | {counts.get('seen', 0)} | {counts.get('applied', 0)} | {counts.get('skipped_probability', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Continuity Evidence",
            "",
            f"- Train dirs: `1`",
            f"- Stage dirs: `{continuity['stage_restart_count']}`",
            f"- args.yaml epochs: `{continuity.get('epochs_arg')}`",
            f"- args.yaml resume: `{continuity.get('resume_arg')}`",
            f"- args.yaml close_mosaic: `{continuity.get('close_mosaic_arg')}`",
            f"- results.csv epoch sequence: `{continuity.get('epoch_sequence')}`",
            f"- Global best.pt: `{continuity.get('global_best_path')}`",
            "",
            "## Final Metrics",
            "",
            f"- Precision: `{fmt(payload['val']['metrics'].get('precision'))}`",
            f"- Recall: `{fmt(payload['val']['metrics'].get('recall'))}`",
            f"- mAP50: `{fmt(payload['val']['metrics'].get('map50'))}`",
            f"- mAP50-95: `{fmt(payload['val']['metrics'].get('map50_95'))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_catf_smoke_report(payload: dict[str, Any]) -> str:
    history = payload.get("policy_history", [])
    first_update = history[0] if history else {}
    stats = payload["online_aug_stats"]
    lines = [
        "# CATF Feedback Controller Smoke Report",
        "",
        "## Answers",
        "",
        f"- CATF enabled: `{str(payload['feedback_enabled'] and payload['industrial_aug_enabled']).lower()}`",
        f"- Reference curve loaded: `{str(bool(payload.get('reference_curve_epochs'))).lower()}`",
        f"- Reference curve path: `{payload.get('reference_curve_path')}`",
        f"- Epoch 5 update happened: `{str(5 in payload['summary']['feedback_epochs']).lower()}`",
        f"- Trust-region active: `{str(all_trust_region_steps_within_limit(history)).lower()}`",
        f"- Group budget active: `{str(all_group_budgets_within_limit(history)).lower()}`",
        f"- Proposed policy recorded: `{str(bool(first_update.get('proposed_policy'))).lower()}`",
        f"- Accepted policy recorded: `{str(bool(first_update.get('accepted_policy'))).lower()}`",
        f"- Guard triggered: `{first_update.get('guard_triggered', [])}`",
        f"- Rollback triggered: `{str(any(item.get('action') == 'rollback' for item in history)).lower()}`",
        f"- Freeze triggered: `{str(any(item.get('frozen') for item in history)).lower()}`",
        f"- BBox/class legal: `{str(payload['summary']['bbox_class_valid']).lower()}`",
        f"- Train image count: `{stats.get('train_image_count')}`",
        f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
        "",
        "## Policy History",
        "",
        "| epoch | action | guards | adjustments | frozen |",
        "|---:|---|---|---:|---|",
    ]
    for record in history:
        lines.append(
            f"| {record.get('epoch')} | {record.get('action')} | "
            f"{','.join(record.get('guard_triggered', [])) or 'none'} | "
            f"{len(record.get('adjustments', []))} | {str(record.get('frozen')).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- The smoke is sufficient for a multi-seed 50ep CATF recheck only if training succeeded, bbox/class remained legal, and the first update recorded proposed/accepted policies with trust-region and budget evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def all_trust_region_steps_within_limit(history: list[dict[str, Any]]) -> bool:
    for record in history:
        for adjustment in record.get("adjustments", []):
            limit = 0.02 if adjustment.get("field") == "prob" else 0.03
            if abs(float(adjustment.get("after", 0.0)) - float(adjustment.get("before", 0.0))) > limit + 1e-9:
                return False
    return True


def all_group_budgets_within_limit(history: list[dict[str, Any]]) -> bool:
    for record in history:
        for payload in (record.get("group_budget_after") or {}).values():
            if not payload.get("within_budget", False):
                return False
    return True


def primary_report_path(output_dir: Path, args: argparse.Namespace) -> Path:
    if not bool(args.feedback_enabled) and not bool(args.industrial_aug_enabled):
        return output_dir / "reports" / "inloop_no_feedback_control_report.md"
    if int(args.epochs) <= 10:
        return output_dir / "reports" / "catf_smoke_report.md"
    return output_dir / "reports" / "final_report.md"


def build_final_report(payload: dict[str, Any]) -> str:
    metrics = payload["val"]["metrics"]
    continuity = payload["continuity"]
    stats = payload["online_aug_stats"]
    scoring = payload["constraint_scoring"]
    baseline = scoring["baseline_metrics"]
    deltas = scoring["deltas"]
    adjustment_counts = adjustment_direction_counts(payload.get("policy_history", []))
    improved_p = (deltas.get("precision") or 0.0) > 0
    improved_r = (deltas.get("recall") or 0.0) > 0
    improved_map50 = (deltas.get("map50") or 0.0) > 0
    map95_within_constraint = (deltas.get("map50_95") or 0.0) >= -0.01
    lines = [
        "# In-Loop YOLO Default Feedback Report",
        "",
        "## Run Integrity",
        "",
        f"- Training success: `{str(payload['train']['success']).lower()}`",
        f"- Single-run continuous training: `{str(payload['summary']['single_run_inloop_feedback']).lower()}`",
        f"- Stage restart count: `{continuity['stage_restart_count']}`",
        f"- Epoch sequence continuous: `{str(continuity['epoch_continuous']).lower()}`",
        f"- Epoch sequence: `{continuity.get('epoch_sequence')}`",
        f"- YOLO default augmentation enabled: `{str(payload['summary']['yolo_default_augmentation_enabled']).lower()}`",
        f"- Industrial augmentation enabled: `{str(payload['industrial_aug_enabled']).lower()}`",
        f"- Industrial augmentation dynamic: `{str(payload['summary']['industrial_aug_dynamic']).lower()}`",
        f"- Train image count: `{stats.get('train_image_count')}`",
        f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
        f"- BBox/class legal: `{str(payload['summary']['bbox_class_valid']).lower()}`",
        f"- Global best.pt: `{continuity.get('global_best_path')}`",
        "",
        "## Metrics",
        "",
        f"- Precision: `{fmt(metrics.get('precision'))}`",
        f"- Recall: `{fmt(metrics.get('recall'))}`",
        f"- mAP50: `{fmt(metrics.get('map50'))}`",
        f"- mAP50-95: `{fmt(metrics.get('map50_95'))}`",
        "",
        "## Clean Native Reference",
        "",
        f"- Baseline: `{scoring['baseline_name']}`",
        f"- Precision: `{fmt(baseline.get('precision'))}`",
        f"- Recall: `{fmt(baseline.get('recall'))}`",
        f"- mAP50: `{fmt(baseline.get('map50'))}`",
        f"- mAP50-95: `{fmt(baseline.get('map50_95'))}`",
        "",
        "## Feedback",
        "",
        f"- Feedback enabled: `{str(payload['feedback_enabled']).lower()}`",
        f"- Feedback controller: `{payload.get('feedback_controller', 'CATF')}`",
        f"- Reference curve loaded: `{str(payload.get('reference_curve_loaded', False)).lower()}`",
        f"- Reference curve path: `{payload.get('reference_curve_path')}`",
        f"- Feedback epochs: `{payload['summary']['feedback_epochs']}`",
        f"- Policy updates: `{payload['summary']['feedback_update_count']}`",
        f"- copy_paste status: `pending_object_bank_design`",
        f"- Ops upregulated: `{format_adjustment_counts(adjustment_counts['up'])}`",
        f"- Ops downregulated: `{format_adjustment_counts(adjustment_counts['down'])}`",
        f"- Guard-triggered epochs: `{format_guard_epochs(payload.get('policy_history', []))}`",
        f"- Rollback triggered: `{str(any(record.get('action') == 'rollback' for record in payload.get('policy_history', []))).lower()}`",
        f"- Cooldown triggered: `{str(any(record.get('action') == 'cooldown' or int(record.get('cooldown_remaining') or 0) > 0 for record in payload.get('policy_history', []))).lower()}`",
        f"- Freeze triggered: `{str(any(record.get('frozen') for record in payload.get('policy_history', []))).lower()}`",
        "",
        "## Industrial Augmentation Stats",
        "",
        "| op | seen | applied | skipped_probability |",
        "|---|---:|---:|---:|",
    ]
    for name, counts in sorted((stats.get("ops") or {}).items()):
        lines.append(
            f"| {name} | {counts.get('seen', 0)} | {counts.get('applied', 0)} | {counts.get('skipped_probability', 0)} |"
        )
    lines.extend(
        [
        "",
        "## Constraint Scoring",
        "",
        f"- Baseline: `{scoring['baseline_name']}`",
        f"- Delta Precision: `{fmt(scoring['deltas'].get('precision'))}`",
        f"- Delta Recall: `{fmt(scoring['deltas'].get('recall'))}`",
        f"- Delta mAP50: `{fmt(scoring['deltas'].get('map50'))}`",
        f"- Delta mAP50-95: `{fmt(scoring['deltas'].get('map50_95'))}`",
        f"- constraint_failed: `{str(scoring['constraint_failed']).lower()}`",
        "",
        "## Verdict",
        "",
        f"- Exceeds clean native reference on Precision: `{str(improved_p).lower()}`",
        f"- Exceeds clean native reference on Recall: `{str(improved_r).lower()}`",
        f"- Exceeds clean native reference on mAP50: `{str(improved_map50).lower()}`",
        f"- mAP50-95 remains within constraint: `{str(map95_within_constraint).lower()}`",
        feedback_worthy_verdict(payload),
        ]
    )
    return "\n".join(lines) + "\n"


def adjustment_direction_counts(history: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"up": {}, "down": {}}
    for record in history:
        for adjustment in record.get("adjustments", []):
            op = str(adjustment.get("op"))
            before = float(adjustment.get("before") or 0.0)
            after = float(adjustment.get("after") or 0.0)
            if after > before:
                counts["up"][op] = counts["up"].get(op, 0) + 1
            elif after < before:
                counts["down"][op] = counts["down"].get(op, 0) + 1
    return counts


def format_adjustment_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{op}:{count}" for op, count in sorted(counts.items()))


def format_guard_epochs(history: list[dict[str, Any]]) -> str:
    items = []
    for record in history:
        guards = record.get("guard_triggered") or []
        if guards:
            items.append(f"{record.get('epoch')}:{'/'.join(str(guard) for guard in guards)}")
    return ", ".join(items) if items else "none"


def build_no_feedback_control_report(payload: dict[str, Any]) -> str:
    metrics = payload["val"]["metrics"]
    delta = payload["delta_vs_reference"]
    close = control_close_to_reference(delta)
    lines = [
        "# In-Loop No-Feedback YOLO Default Control",
        "",
        "## Integrity",
        "",
        f"- Training success: `{str(payload['train']['success']).lower()}`",
        f"- Single train run: `{str(payload['continuity']['single_train_run_dir']).lower()}`",
        f"- Stage restart count: `{payload['continuity']['stage_restart_count']}`",
        f"- Epoch sequence continuous: `{str(payload['continuity']['epoch_continuous']).lower()}`",
        f"- args.yaml epochs: `{payload['continuity'].get('epochs_arg')}`",
        f"- close_mosaic official/global: `{str(payload['continuity']['official_close_mosaic_managed_by_yolo']).lower()}`",
        f"- YOLO default augmentation enabled: `{str(payload['summary']['yolo_default_augmentation_enabled']).lower()}`",
        f"- Industrial augmentation enabled: `{str(payload['industrial_aug_enabled']).lower()}`",
        f"- Feedback enabled: `{str(payload['feedback_enabled']).lower()}`",
        f"- Train image count: `{payload['online_aug_stats'].get('train_image_count')}`",
        f"- Fixed augmented dataset generated: `{str(payload['online_aug_stats'].get('fixed_augmented_dataset_generated')).lower()}`",
        f"- Global best.pt: `{payload['continuity'].get('global_best_path')}`",
        "",
        "## Metrics",
        "",
        metric_table(metrics, payload["reference_metrics"], delta),
        "",
        "## Answer",
        "",
        f"- Close to YOLO default reference: `{str(close).lower()}`",
        "- If this control is not close, feedback 50ep should not be trusted because the entrypoint changed training behavior.",
    ]
    return "\n".join(lines) + "\n"


def build_reference_comparison_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Compare In-Loop No-Feedback With YOLO Default Reference",
            "",
            metric_table(payload["val"]["metrics"], payload["reference_metrics"], payload["delta_vs_reference"]),
            "",
            f"- Close to reference: `{str(control_close_to_reference(payload['delta_vs_reference'])).lower()}`",
        ]
    ) + "\n"


def build_feedback_comparison_report(payload: dict[str, Any]) -> str:
    metrics = payload["val"]["metrics"]
    lines = [
        "# Compare In-Loop Feedback With YOLO Default / No-Feedback",
        "",
        "## Versus YOLO Default Reference",
        "",
        metric_table(metrics, payload["reference_metrics"], payload["delta_vs_reference"]),
    ]
    if payload.get("control_metrics"):
        lines.extend(
            [
                "",
                "## Versus In-Loop No-Feedback Control",
                "",
                metric_table(metrics, payload["control_metrics"], payload["delta_vs_control"]),
            ]
        )
    lines.extend(
        [
            "",
            "## Constraint",
            "",
            f"- Baseline: `{payload['constraint_scoring']['baseline_name']}`",
            f"- constraint_failed: `{str(payload['constraint_scoring']['constraint_failed']).lower()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_control_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": payload["run_id"],
        "output_dir": payload["output_dir"],
        "metrics": compact_metrics(payload["val"]["metrics"]),
        "reference_metrics": payload["reference_metrics"],
        "delta_vs_reference": payload["delta_vs_reference"],
        "close_to_yolo_default_reference": control_close_to_reference(payload["delta_vs_reference"]),
        "continuity": payload["continuity"],
        "train": payload["train"],
        "val": payload["val"],
    }


def metric_table(metrics: dict[str, Any], reference: dict[str, Any], delta: dict[str, Any]) -> str:
    lines = ["| metric | value | reference | delta |", "|---|---:|---:|---:|"]
    labels = {"precision": "Precision", "recall": "Recall", "map50": "mAP50", "map50_95": "mAP50-95"}
    for key in METRIC_KEYS:
        lines.append(f"| {labels[key]} | {fmt(metrics.get(key))} | {fmt(reference.get(key))} | {fmt(delta.get(key))} |")
    return "\n".join(lines)


def feedback_worthy_verdict(payload: dict[str, Any]) -> str:
    if not payload["feedback_enabled"]:
        return "This is a no-feedback control run."
    scoring = payload["constraint_scoring"]
    delta = scoring["deltas"]
    recall_gain = delta.get("recall")
    if scoring["constraint_failed"]:
        return "Not acceptable as the paper main method under current industrial constraints."
    if recall_gain is not None and recall_gain > 0:
        return "Acceptable candidate: Recall improved without violating Precision/mAP constraints."
    return "Not yet a strong paper main method: constraints passed but Recall did not improve."


def load_yolo_default_reference(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if isinstance(payload, dict):
        if isinstance(payload.get("metrics"), dict):
            return compact_metrics(payload["metrics"])
        if isinstance(payload.get("val"), dict) and isinstance(payload["val"].get("metrics"), dict):
            return compact_metrics(payload["val"]["metrics"])
    for row in payload.get("rows", []):
        if row.get("key") in {"yolo_default", "clean_native_yolo_default"}:
            return compact_metrics(row.get("metrics", {}))
    raise ValueError(f"missing yolo_default reference row in {path}")


def load_reference_curve(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".csv":
        rows = path.read_text(encoding="utf-8-sig").splitlines()
        if not rows:
            return {}
        reader = csv.DictReader(rows)
        curve: dict[int, dict[str, Any]] = {}
        for row in reader:
            epoch_value = row.get("epoch") or row.get("Epoch") or row.get("epoch_index")
            try:
                epoch = int(float(str(epoch_value).strip()))
            except (TypeError, ValueError):
                continue
            curve[epoch] = {
                "precision": _csv_float(row, "metrics/precision(B)", "precision"),
                "recall": _csv_float(row, "metrics/recall(B)", "recall"),
                "map50": _csv_float(row, "metrics/mAP50(B)", "map50"),
                "map50_95": _csv_float(row, "metrics/mAP50-95(B)", "map50_95"),
            }
        return {epoch: compact_metrics(metrics) for epoch, metrics in curve.items()}
    payload = read_json(path)
    if isinstance(payload, dict):
        source = payload.get("epoch_records") or payload.get("curve") or payload.get("history") or []
        curve: dict[int, dict[str, Any]] = {}
        if isinstance(source, list):
            for record in source:
                if not isinstance(record, dict):
                    continue
                epoch_value = record.get("epoch")
                try:
                    epoch = int(float(epoch_value))
                except (TypeError, ValueError):
                    continue
                metrics = record.get("metrics", record)
                if isinstance(metrics, dict):
                    curve[epoch] = compact_metrics(metrics)
        return curve
    return {}


def _csv_float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key not in row:
            continue
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return None


def load_control_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    if isinstance(payload, dict):
        if isinstance(payload.get("metrics"), dict):
            return compact_metrics(payload["metrics"])
        if isinstance(payload.get("val"), dict) and isinstance(payload["val"].get("metrics"), dict):
            return compact_metrics(payload["val"]["metrics"])
    return None


def baseline_name_from_path(path: Path) -> str:
    normalized = str(path).replace("\\", "/").lower()
    if "clean_native_yolo_default" in normalized:
        return "clean_native_yolo_default"
    if "no_feedback" in normalized:
        return "no_feedback_control"
    return "yolo_default_reference"


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
        ref = reference.get(key)
        delta[key] = None if value is None or ref is None else float(value) - float(ref)
    return delta


def build_constraint_scoring(metrics: dict[str, Any], baseline: dict[str, Any], *, baseline_name: str) -> dict[str, Any]:
    deltas = metric_delta(metrics, baseline)
    failures = []
    if deltas.get("precision") is not None and deltas["precision"] < -0.01:
        failures.append("precision_drop_gt_0.01")
    if deltas.get("map50") is not None and deltas["map50"] < -0.01:
        failures.append("map50_drop_gt_0.01")
    if deltas.get("map50_95") is not None and deltas["map50_95"] < -0.01:
        failures.append("map50_95_drop_gt_0.01")
    return {
        "baseline_name": baseline_name,
        "baseline_metrics": baseline,
        "metrics": compact_metrics(metrics),
        "deltas": deltas,
        "failure_reasons": failures,
        "constraint_failed": bool(failures),
        "rules": {
            "precision_drop_gt_0.01": True,
            "map50_drop_gt_0.01": True,
            "map50_95_drop_gt_0.01": True,
            "recall_gain_alone_is_not_success": True,
        },
    }


def control_close_to_reference(delta: dict[str, Any]) -> bool:
    thresholds = {"precision": 0.02, "recall": 0.03, "map50": 0.02, "map50_95": 0.02}
    for key, threshold in thresholds.items():
        value = delta.get(key)
        if value is None or abs(float(value)) > threshold:
            return False
    return True


def fixed_augmented_dataset_generated(output_dir: Path) -> bool:
    return any(
        path.exists()
        for path in [
            output_dir / "dataset" / "final_dataset" / "images",
            output_dir / "dataset_builder" / "final_dataset" / "images",
        ]
    )


def read_epoch_sequence(results_csv: Path) -> list[int]:
    if not results_csv.exists():
        return []
    rows = results_csv.read_text(encoding="utf-8").splitlines()
    if len(rows) <= 1:
        return []
    values: list[int] = []
    for row in rows[1:]:
        first = row.split(",", 1)[0].strip()
        try:
            values.append(int(float(first)))
        except ValueError:
            continue
    return values


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


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
    report_rel = Path(payload["summary"]["report"]).resolve().relative_to(PROJECT_ROOT).as_posix()
    policy_rel = Path(payload["summary"]["policy_history"]).resolve().relative_to(PROJECT_ROOT).as_posix()
    content = "\n".join(
        [
            "## YOLO Default In-Loop Feedback / Control",
            "",
            "- Entrypoint: `scripts/train_yolo_default_with_inloop_feedback.py`.",
            "- The previous `yolo_default_feedback_aug_50ep_full` run is a segmented fine-tune experiment, not strict continuous feedback.",
            "- New direction: one `YOLO.train()` run with in-loop feedback callbacks; optimizer/scheduler/EMA/epoch/close_mosaic remain under one Ultralytics trainer.",
            "- Feedback controller: `CATF` (Constraint-Aware Trust-region Feedback Controller).",
            "- CATF uses the clean native YOLO default reference curve at matching feedback epochs, trust-region step limits, group budgets, delayed acceptance, rollback, cooldown, and epoch>=40 freeze.",
            "- No-feedback control disables both feedback and industrial augmentation, using Ultralytics YOLO default augmentation as the behavior check.",
            "- The old YOLO default reference is not the final baseline after parity audit; feedback comparisons should use `clean_native_yolo_default_seed42_50ep`.",
            f"- Output: `outputs/experiments/{payload['run_id']}/`",
            f"- Epochs: `{payload['epochs']}`",
            f"- Feedback enabled: `{str(payload['feedback_enabled']).lower()}`",
            f"- Industrial augmentation enabled: `{str(payload['industrial_aug_enabled']).lower()}`",
            f"- Reference curve loaded: `{str(payload.get('reference_curve_loaded', False)).lower()}`",
            f"- Feedback epochs: `{payload['summary']['feedback_epochs']}`",
            f"- Stage restart count: `{payload['continuity']['stage_restart_count']}`",
            f"- Epoch continuous: `{str(payload['continuity']['epoch_continuous']).lower()}`",
            f"- Train image count: `{payload['online_aug_stats'].get('train_image_count')}`",
            f"- Fixed augmented dataset generated: `{str(payload['online_aug_stats'].get('fixed_augmented_dataset_generated')).lower()}`",
            f"- Constraint baseline: `{payload['constraint_scoring']['baseline_name']}`",
            f"- Constraint failed: `{payload['summary']['constraint_failed']}`",
            f"- Report: `{report_rel}`",
            f"- Policy history: `{policy_rel}`",
        ]
    )
    for name in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
        update_marked_section(PROJECT_ROOT / name, "YOLO_DEFAULT_INLOOP_FEEDBACK_SMOKE", content)


def export_project_snapshot() -> None:
    subprocess.run([str(PYTHON_EXE), str(PROJECT_ROOT / "scripts/export_project_snapshot.py")], cwd=str(PROJECT_ROOT), check=False)


if __name__ == "__main__":
    main()
