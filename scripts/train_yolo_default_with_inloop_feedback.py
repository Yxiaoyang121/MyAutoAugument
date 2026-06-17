from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from AutoAugment.catf_v2 import (  # noqa: E402
    AdaptiveBurninConfig,
    AdaptiveBurninController,
    CATFRollbackController,
    CATFGatedController,
    CATFSafeController,
    ClassAwareCATFController,
    ROIStats,
    SampleAwareAugmentationRouter,
    count_train_instances,
    force_noop_policy,
    initial_policy_matrix,
)
from AutoAugment.catf_v2.adaptive_burnin import annotate_policy_history_with_adaptive_burnin, probe_window  # noqa: E402
from AutoAugment.catf_v2.gated_controller import annotate_policy_history_with_gate  # noqa: E402
from AutoAugment.catf_v2.rollback_controller import annotate_policy_history_with_rollback  # noqa: E402
from AutoAugment.catf_v2.class_aware_controller import build_sample_weight_map  # noqa: E402
from AutoAugment.catf_v2.causal_probe import (  # noqa: E402
    ProbeConfig,
    build_probe_set,
    candidate_policy_catalog,
    evaluate_candidate_policy,
    select_best_candidate,
)
from AutoAugment.catf_v2.high_risk_class_ops import (  # noqa: E402
    apply_risk_guard_to_policy,
    build_sampler_only_fallback_map,
)
from AutoAugment.catf_v2.issue_attribution import attribute_class_issues  # noqa: E402
from AutoAugment.catf_v2.policy_matrix import active_class_ids, frozen_class_ids  # noqa: E402
from AutoAugment.catf_v2.per_class_diagnosis import build_per_class_diagnosis  # noqa: E402
from AutoAugment.catf_v2.sampler_only import build_sampler_only_artifacts  # noqa: E402
from AutoAugment.catf_v2.threshold_calibration import ThresholdCalibrationAnalyzer  # noqa: E402
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
    feedback_controller: Any | None = None
    adaptive_burnin_controller: AdaptiveBurninController | None = None
    rollback_controller: CATFRollbackController | None = None
    safe_controller: CATFSafeController | None = None
    gated_controller: CATFGatedController | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    adaptive_burnin_events: list[dict[str, Any]] = field(default_factory=list)
    rollback_events: list[dict[str, Any]] = field(default_factory=list)
    safe_events: list[dict[str, Any]] = field(default_factory=list)
    gated_events: list[dict[str, Any]] = field(default_factory=list)
    riskguard_events: list[dict[str, Any]] = field(default_factory=list)
    causal_probe_events: list[dict[str, Any]] = field(default_factory=list)
    offline_probe_decision: dict[str, Any] | None = None
    offline_probe_decision_schedule: dict[int, dict[str, Any]] | None = None
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
    paper_probe_audit = validate_paper_probe_configuration(args) if bool(getattr(args, "paper_probe_mode", False)) else {}
    if paper_probe_audit:
        write_json(output_dir / "reports" / "paper_probe_leakage_audit.json", paper_probe_audit)

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

    catf_v2_enabled = is_catf_v2(args)
    class_names = load_class_names_from_data_yaml(args.data) if catf_v2_enabled else {}
    train_instances = count_train_instances(args.data) if catf_v2_enabled else {}
    policy_state = initial_catf_v2_policy_state(class_names) if catf_v2_enabled else initial_policy_state()
    active_policy = training_policy(policy_state, enabled=bool(args.industrial_aug_enabled)) if not catf_v2_enabled else deepcopy(policy_state)
    write_json(output_dir / "configs" / "initial_policy_state.json", policy_state)
    write_json(output_dir / "configs" / "active_policy_epoch_000.json", active_policy)
    write_json(output_dir / "configs" / "train_config.json", build_train_config(args, output_dir))

    stats = OnlineAugmentationStats()
    if catf_v2_enabled:
        augmentor = SampleAwareAugmentationRouter(
            active_policy,
            seed=args.seed,
            num_classes=len(class_names) if class_names else None,
            stats=stats,
            roi_stats=ROIStats(),
            roi_aware=bool(args.roi_aware_aug),
            sample_aware=bool(args.sample_aware_routing),
            total_epochs=args.epochs,
            riskguard_enabled=bool(getattr(args, "catf_riskguard", False)),
            riskguard_sampler_only=True,
        )
    else:
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
        catf_noop=bool(getattr(args, "catf_noop", False)),
    )
    state = InLoopFeedbackState(
        output_dir=output_dir,
        args=args,
        context=context,
        policy_state=policy_state,
        reference_metrics=reference_metrics,
        reference_curve=reference_curve,
    )
    if catf_v2_enabled and bool(args.adaptive_burnin):
        burnin_config = adaptive_burnin_config_from_args(args)
        state.adaptive_burnin_controller = AdaptiveBurninController(burnin_config)
    if catf_v2_enabled and bool(args.catf_rollback_mode):
        state.rollback_controller = CATFRollbackController(probe_window=adaptive_probe_window(args))
    if catf_v2_enabled and bool(args.catf_safe_mode):
        state.safe_controller = CATFSafeController()
    if catf_v2_enabled and bool(args.catf_gated_mode):
        state.gated_controller = CATFGatedController()
    if catf_v2_enabled:
        state.feedback_controller = ClassAwareCATFController(
            policy_state,
            history_dir=output_dir / "reports",
            class_names=class_names,
            train_instances=train_instances,
            top_k_active_classes=int(args.top_k_active_classes),
            top_m_ops_per_class=int(args.top_m_ops_per_class),
            freeze_epoch=40,
            threshold_calibration_report=bool(args.threshold_calibration_report),
        )
    else:
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


def is_catf_v2(args: argparse.Namespace) -> bool:
    return str(getattr(args, "catf_version", "v1")).lower() == "v2"


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
    parser.add_argument("--diagnosis-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--feedback-interval", type=int, default=5)
    parser.add_argument("--feedback-start-epoch", type=int, default=5)
    parser.add_argument("--feedback-profile", default="industrial")
    parser.add_argument("--industrial-aug-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--catf-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--catf-noop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--catf-safe-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--catf-gated-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--catf-rollback-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--catf-riskguard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--catf-causal-probe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--causal-probe-mode", dest="catf_causal_probe", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--catf-causal-probe-mode", choices=["development", "paper"], default="development")
    parser.add_argument("--precision-aware-accept-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-only-mainline", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--preserve-original-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weak-image-aug-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weak-only-for-moderate-risk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attenuation-ratio", type=float, default=0.25)
    parser.add_argument("--disable-sampler-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sampler-only-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-offline-probe-decisions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offline-probe-decisions-dir", default=str(PROJECT_ROOT / "outputs/experiments/catf_v2_causal_probe"))
    parser.add_argument("--offline-probe-decisions-file", default=None)
    parser.add_argument("--paper-probe-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--probe-data", default=None)
    parser.add_argument("--train-core-data", default=None)
    parser.add_argument("--probe-source", default="train_probe_split")
    parser.add_argument("--forbid-final-val-policy-selection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adaptive-burnin", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-burnin-epoch", type=int, default=5)
    parser.add_argument("--max-burnin-epoch", type=int, default=15)
    parser.add_argument("--burnin-check-interval", type=int, default=5)
    parser.add_argument("--metric-stability-window", type=int, default=3)
    parser.add_argument("--map50-stability-threshold", type=float, default=0.02)
    parser.add_argument("--recall-stability-threshold", type=float, default=0.03)
    parser.add_argument("--min-diagnosis-evidence", type=int, default=5)
    parser.add_argument("--min-active-class-evidence", type=int, default=5)
    parser.add_argument("--allow-force-start-at-max-burnin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-aware-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--roi-aware-aug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-aware-routing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--threshold-calibration-report", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-k-active-classes", type=int, default=2)
    parser.add_argument("--top-m-ops-per-class", type=int, default=2)
    parser.add_argument("--reference-metrics", default=str(REFERENCE_METRICS))
    parser.add_argument("--reference-curve", default=str(DEFAULT_REFERENCE_CURVE))
    parser.add_argument("--control-metrics", default=str(DEFAULT_CONTROL_METRICS))
    parser.add_argument("--save-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--keep-diagnosis-predict-runs", action="store_true")
    parser.add_argument("--skip-doc-update", action="store_true")
    args = parser.parse_args(normalize_bool_cli_args(sys.argv[1:]))
    normalize_paper_probe_args(args)
    normalize_image_only_args(args)
    return args


def normalize_paper_probe_args(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "paper_probe_mode", False)):
        return
    args.catf_causal_probe = True
    args.catf_causal_probe_mode = "paper"
    if not getattr(args, "probe_data", None):
        raise ValueError("--paper-probe-mode requires --probe-data")
    if not bool(getattr(args, "forbid_final_val_policy_selection", False)):
        raise ValueError("--paper-probe-mode requires --forbid-final-val-policy-selection true")
    train_core = getattr(args, "train_core_data", None)
    if train_core and Path(train_core).resolve() != Path(args.data).resolve():
        raise ValueError("--train-core-data must match --data in paper probe mode")


def normalize_image_only_args(args: argparse.Namespace) -> None:
    if bool(getattr(args, "disable_sampler_only", False)):
        args.sampler_only_enabled = False
    if bool(getattr(args, "weak_image_aug_enabled", False)):
        ratio = float(getattr(args, "attenuation_ratio", 0.25) or 0.25)
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError("--attenuation-ratio must be in (0, 1]")
        if not bool(getattr(args, "image_only_mainline", False)):
            args.image_only_mainline = True
    if bool(getattr(args, "preserve_original_enabled", False)) or bool(getattr(args, "weak_only_for_moderate_risk", False)):
        args.image_only_mainline = True
        args.catf_causal_probe = True
        args.use_offline_probe_decisions = True


def normalize_bool_cli_args(argv: list[str]) -> list[str]:
    """Accept both `--flag` and `--flag true/false` for BooleanOptionalAction flags."""

    bool_flags = {
        "--feedback-enabled",
        "--diagnosis-only",
        "--industrial-aug-enabled",
        "--catf-noop",
        "--catf-safe-mode",
        "--catf-gated-mode",
        "--catf-rollback-mode",
        "--catf-riskguard",
        "--catf-causal-probe",
        "--causal-probe-mode",
        "--precision-aware-accept-gate",
        "--image-only-mainline",
        "--preserve-original-enabled",
        "--weak-image-aug-enabled",
        "--weak-only-for-moderate-risk",
        "--disable-sampler-only",
        "--sampler-only-enabled",
        "--use-offline-probe-decisions",
        "--paper-probe-mode",
        "--forbid-final-val-policy-selection",
        "--adaptive-burnin",
        "--allow-force-start-at-max-burnin",
        "--class-aware-feedback",
        "--roi-aware-aug",
        "--sample-aware-routing",
        "--threshold-calibration-report",
        "--save-preview",
    }
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in bool_flags and index + 1 < len(argv) and argv[index + 1].lower() in {"true", "false", "1", "0", "yes", "no"}:
            value = argv[index + 1].lower() in {"true", "1", "yes"}
            out.append(token if value else "--no-" + token[2:])
            index += 2
            continue
        out.append(token)
        index += 1
    return out


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
        "diagnosis_only": bool(args.diagnosis_only),
        "feedback_interval": int(args.feedback_interval),
        "feedback_start_epoch": int(args.feedback_start_epoch),
        "industrial_aug_enabled": bool(args.industrial_aug_enabled),
        "sampler_only_enabled": bool(getattr(args, "sampler_only_enabled", False)),
        "feedback_controller": "CATF-v2" if is_catf_v2(args) else "CATF",
        "catf_version": str(args.catf_version),
        "catf_noop": bool(args.catf_noop),
        "catf_safe_mode": bool(args.catf_safe_mode),
        "catf_gated_mode": bool(args.catf_gated_mode),
        "catf_rollback_mode": bool(args.catf_rollback_mode),
        "catf_riskguard": bool(getattr(args, "catf_riskguard", False)),
        "catf_causal_probe": bool(getattr(args, "catf_causal_probe", False)),
        "catf_causal_probe_mode": str(getattr(args, "catf_causal_probe_mode", "development")),
        "precision_aware_accept_gate": bool(getattr(args, "precision_aware_accept_gate", True)),
        "image_only_mainline": bool(getattr(args, "image_only_mainline", False)),
        "preserve_original_enabled": bool(getattr(args, "preserve_original_enabled", False)),
        "weak_image_aug_enabled": bool(getattr(args, "weak_image_aug_enabled", False)),
        "weak_only_for_moderate_risk": bool(getattr(args, "weak_only_for_moderate_risk", False)),
        "attenuation_ratio": float(getattr(args, "attenuation_ratio", 0.25) or 0.25),
        "disable_sampler_only": bool(getattr(args, "disable_sampler_only", False)),
        "sampler_only_enabled": bool(getattr(args, "sampler_only_enabled", False)),
        "use_offline_probe_decisions": bool(getattr(args, "use_offline_probe_decisions", False)),
        "offline_probe_decisions_dir": str(Path(getattr(args, "offline_probe_decisions_dir", "")).resolve()),
        "offline_probe_decisions_file": str(Path(getattr(args, "offline_probe_decisions_file")).resolve()) if getattr(args, "offline_probe_decisions_file", None) else None,
        "paper_probe_mode": bool(getattr(args, "paper_probe_mode", False)),
        "probe_data": str(Path(args.probe_data).resolve()) if getattr(args, "probe_data", None) else None,
        "train_core_data": str(Path(args.train_core_data).resolve()) if getattr(args, "train_core_data", None) else str(Path(args.data).resolve()),
        "probe_source": str(getattr(args, "probe_source", "train_probe_split")),
        "forbid_final_val_policy_selection": bool(getattr(args, "forbid_final_val_policy_selection", False)),
        "policy_selection_source": policy_selection_source(args),
        "policy_selection_data": str(policy_selection_data_yaml(args).resolve()),
        "adaptive_burnin": bool(args.adaptive_burnin),
        "min_burnin_epoch": int(args.min_burnin_epoch),
        "max_burnin_epoch": int(args.max_burnin_epoch),
        "burnin_check_interval": int(args.burnin_check_interval),
        "metric_stability_window": int(args.metric_stability_window),
        "map50_stability_threshold": float(args.map50_stability_threshold),
        "recall_stability_threshold": float(args.recall_stability_threshold),
        "min_diagnosis_evidence": int(args.min_diagnosis_evidence),
        "min_active_class_evidence": int(args.min_active_class_evidence),
        "allow_force_start_at_max_burnin": bool(args.allow_force_start_at_max_burnin),
        "adaptive_probe_window": adaptive_probe_window(args),
        "class_aware_feedback": bool(args.class_aware_feedback),
        "roi_aware_aug": bool(args.roi_aware_aug),
        "sample_aware_routing": bool(args.sample_aware_routing),
        "threshold_calibration_report": bool(args.threshold_calibration_report),
        "top_k_active_classes": int(args.top_k_active_classes),
        "top_m_ops_per_class": int(args.top_m_ops_per_class),
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
        f"diagnosis_only={bool(getattr(args, 'diagnosis_only', False))}",
        f"industrial_aug_enabled={bool(args.industrial_aug_enabled)}",
        f"feedback_controller={'CATF-v2' if is_catf_v2(args) else 'CATF'}",
        f"catf_version={getattr(args, 'catf_version', 'v1')}",
        f"catf_noop={bool(getattr(args, 'catf_noop', False))}",
        f"catf_safe_mode={bool(getattr(args, 'catf_safe_mode', False))}",
        f"catf_gated_mode={bool(getattr(args, 'catf_gated_mode', False))}",
        f"catf_rollback_mode={bool(getattr(args, 'catf_rollback_mode', False))}",
        f"catf_riskguard={bool(getattr(args, 'catf_riskguard', False))}",
        f"catf_causal_probe={bool(getattr(args, 'catf_causal_probe', False))}",
        f"catf_causal_probe_mode={getattr(args, 'catf_causal_probe_mode', 'development')}",
        f"precision_aware_accept_gate={bool(getattr(args, 'precision_aware_accept_gate', True))}",
        f"image_only_mainline={bool(getattr(args, 'image_only_mainline', False))}",
        f"preserve_original_enabled={bool(getattr(args, 'preserve_original_enabled', False))}",
        f"weak_image_aug_enabled={bool(getattr(args, 'weak_image_aug_enabled', False))}",
        f"weak_only_for_moderate_risk={bool(getattr(args, 'weak_only_for_moderate_risk', False))}",
        f"attenuation_ratio={float(getattr(args, 'attenuation_ratio', 0.25) or 0.25)}",
        f"disable_sampler_only={bool(getattr(args, 'disable_sampler_only', False))}",
        f"sampler_only_enabled={bool(getattr(args, 'sampler_only_enabled', False))}",
        f"use_offline_probe_decisions={bool(getattr(args, 'use_offline_probe_decisions', False))}",
        f"paper_probe_mode={bool(getattr(args, 'paper_probe_mode', False))}",
        f"probe_data={Path(getattr(args, 'probe_data')).resolve() if getattr(args, 'probe_data', None) else None}",
        f"probe_source={getattr(args, 'probe_source', 'train_probe_split')}",
        f"forbid_final_val_policy_selection={bool(getattr(args, 'forbid_final_val_policy_selection', False))}",
        f"adaptive_burnin={bool(getattr(args, 'adaptive_burnin', False))}",
        f"min_burnin_epoch={int(getattr(args, 'min_burnin_epoch', 5))}",
        f"max_burnin_epoch={int(getattr(args, 'max_burnin_epoch', 15))}",
        f"burnin_check_interval={int(getattr(args, 'burnin_check_interval', 5))}",
        f"class_aware_feedback={bool(getattr(args, 'class_aware_feedback', False))}",
        f"roi_aware_aug={bool(getattr(args, 'roi_aware_aug', False))}",
        f"sample_aware_routing={bool(getattr(args, 'sample_aware_routing', False))}",
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


def adaptive_burnin_config_from_args(args: argparse.Namespace) -> AdaptiveBurninConfig:
    return AdaptiveBurninConfig(
        min_burnin_epoch=int(getattr(args, "min_burnin_epoch", 5)),
        max_burnin_epoch=int(getattr(args, "max_burnin_epoch", 15)),
        burnin_check_interval=int(getattr(args, "burnin_check_interval", 5)),
        metric_stability_window=int(getattr(args, "metric_stability_window", 3)),
        map50_stability_threshold=float(getattr(args, "map50_stability_threshold", 0.02)),
        recall_stability_threshold=float(getattr(args, "recall_stability_threshold", 0.03)),
        min_diagnosis_evidence=int(getattr(args, "min_diagnosis_evidence", 5)),
        min_active_class_evidence=int(getattr(args, "min_active_class_evidence", 5)),
        allow_force_start_at_max_burnin=bool(getattr(args, "allow_force_start_at_max_burnin", True)),
        probe_window=adaptive_probe_window(args),
    )


def build_adaptive_burnin_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "adaptive_burnin": bool(getattr(args, "adaptive_burnin", False)),
        "min_burnin_epoch": int(getattr(args, "min_burnin_epoch", 5)),
        "max_burnin_epoch": int(getattr(args, "max_burnin_epoch", 15)),
        "burnin_check_interval": int(getattr(args, "burnin_check_interval", 5)),
        "metric_stability_window": int(getattr(args, "metric_stability_window", 3)),
        "map50_stability_threshold": float(getattr(args, "map50_stability_threshold", 0.02)),
        "recall_stability_threshold": float(getattr(args, "recall_stability_threshold", 0.03)),
        "min_diagnosis_evidence": int(getattr(args, "min_diagnosis_evidence", 5)),
        "min_active_class_evidence": int(getattr(args, "min_active_class_evidence", 5)),
        "allow_force_start_at_max_burnin": bool(getattr(args, "allow_force_start_at_max_burnin", True)),
        "probe_window": adaptive_probe_window(args),
    }


def adaptive_probe_window(args: argparse.Namespace) -> int:
    explicit = getattr(args, "probe_window", None)
    if explicit is not None:
        return max(1, int(explicit))
    return max(3, int(getattr(args, "feedback_interval", 5)))


def apply_catf_v2_riskguard_if_enabled(
    state: InLoopFeedbackState,
    controller: ClassAwareCATFController,
    policy: dict[str, Any],
    *,
    epoch_num: int,
) -> dict[str, Any]:
    if not bool(getattr(state.args, "catf_riskguard", False)):
        return policy
    latest_record = controller.history[-1] if controller.history else {}
    sample_weight_map_path = latest_record.get("sample_weight_map_path")
    guarded_policy, events = apply_risk_guard_to_policy(
        policy,
        epoch=epoch_num,
        sampler_only_fallback=True,
        sample_weight_map_path=sample_weight_map_path,
    )
    if not events:
        return guarded_policy

    blocked_class_ids = sorted({int(event["class_id"]) for event in events})
    riskguard_weight_map_path = state.output_dir / "reports" / f"riskguard_sample_weight_map_epoch_{epoch_num}.json"
    catf_riskguard_weight_map_path = state.output_dir / "reports" / "catf_v2" / f"riskguard_sample_weight_map_epoch_{epoch_num}.json"
    base_weight_map: dict[str, Any] = {}
    if sample_weight_map_path and Path(sample_weight_map_path).exists():
        base_weight_map = read_json(Path(sample_weight_map_path))
    fallback_map = build_sampler_only_fallback_map(base_weight_map, blocked_class_ids=blocked_class_ids)
    write_json(riskguard_weight_map_path, fallback_map)
    write_json(catf_riskguard_weight_map_path, fallback_map)
    for event in events:
        event["sample_weight_map_path"] = str(riskguard_weight_map_path)

    if hasattr(controller, "policy"):
        controller.policy.matrix = deepcopy(guarded_policy)
    if controller.history:
        record = controller.history[-1]
        record["riskguard_enabled"] = True
        record["riskguard_events"] = deepcopy(events)
        record["riskguard_blocked_op_count"] = len(events)
        record["riskguard_blocked_classes"] = blocked_class_ids
        record["riskguard_sampler_only_fallback"] = True
        record["riskguard_sample_weight_map_path"] = str(riskguard_weight_map_path)
        record["accepted_policy_before_riskguard"] = deepcopy(record.get("accepted_policy", policy))
        record["accepted_policy"] = deepcopy(guarded_policy)
        record["new_policy"] = deepcopy(guarded_policy)
        record["active_classes"] = active_class_ids(guarded_policy)
        record["frozen_classes"] = frozen_class_ids(guarded_policy)
        record["guard_triggered"] = list(
            dict.fromkeys(list(record.get("guard_triggered", []) or []) + ["riskguard_class_op_block"])
        )
        class_actions = list(record.get("class_actions", []) or [])
        before_policy = record.get("accepted_policy_before_riskguard") or policy
        for class_id in blocked_class_ids:
            class_actions.append(
                {
                    "class_id": int(class_id),
                    "action": "riskguard_block",
                    "adjustments": [],
                    "before": deepcopy((before_policy.get("classes") or {}).get(str(class_id), {})),
                    "after": deepcopy((guarded_policy.get("classes") or {}).get(str(class_id), {})),
                    "blocked_ops": [
                        {
                            "op_name": event["op_name"],
                            "canonical_op_name": event["canonical_op_name"],
                            "risk_reasons": event.get("risk_reasons", []),
                        }
                        for event in events
                        if int(event.get("class_id", -1)) == int(class_id)
                    ],
                }
            )
        record["class_actions"] = class_actions
    state.riskguard_events.extend(deepcopy(events))
    for base in (state.output_dir / "reports", state.output_dir / "reports" / "catf_v2"):
        write_json(base / f"policy_matrix_epoch_{epoch_num}_after.json", guarded_policy)
        write_json(base / f"policy_matrix_epoch_{epoch_num}_after_riskguard.json", guarded_policy)
        write_json(base / "riskguard_events.json", {"events": state.riskguard_events})
    if hasattr(controller, "_write_histories"):
        controller._write_histories(guarded_policy)
    return guarded_policy


def apply_catf_v2_causal_probe_if_enabled(
    state: InLoopFeedbackState,
    controller: ClassAwareCATFController,
    policy: dict[str, Any],
    *,
    epoch_num: int,
) -> dict[str, Any]:
    if not bool(getattr(state.args, "catf_causal_probe", False)):
        return policy
    decision_payload = load_offline_probe_decision_if_requested(state, epoch_num=epoch_num)
    if decision_payload:
        gated_policy, event = apply_offline_probe_decision_to_policy(
            policy,
            decision_payload,
            epoch_num=epoch_num,
            output_dir=state.output_dir,
            disable_sampler_only=bool(getattr(state.args, "disable_sampler_only", False)),
        )
    elif bool(getattr(state.args, "paper_probe_mode", False)):
        decision_payload = build_paper_probe_decision_payload(state, controller, policy, epoch_num=epoch_num)
        gated_policy, event = apply_offline_probe_decision_to_policy(
            policy,
            decision_payload,
            epoch_num=epoch_num,
            output_dir=state.output_dir,
            disable_sampler_only=bool(getattr(state.args, "disable_sampler_only", False)),
        )
        event["source"] = "paper_probe_split_online_decision"
    else:
        reason = "causal_probe_required_no_online_probe_decision"
        gated_policy = force_noop_policy(policy, reason=reason)
        event = {
            "epoch": int(epoch_num),
            "catf_causal_probe": True,
            "mode": str(getattr(state.args, "catf_causal_probe_mode", "development")),
            "source": "online_probe_unavailable",
            "action": "strict_no_op",
            "reason": reason,
            "policy_update_allowed": False,
            "image_modification_allowed": False,
            "sample_router_allowed": False,
            "development_probe_uses_existing_val_diagnostics": str(
                getattr(state.args, "catf_causal_probe_mode", "development")
            )
            == "development",
            "paper_mode_status": "probe_split_required"
            if str(getattr(state.args, "catf_causal_probe_mode", "development")) == "paper"
            else "not_requested",
        }
    event["use_offline_probe_decisions"] = bool(getattr(state.args, "use_offline_probe_decisions", False))
    if decision_payload:
        if bool(getattr(state.args, "use_offline_probe_decisions", False)):
            event["offline_probe_decision_path"] = str(resolve_offline_probe_decision_path(state.args))
        write_json(state.output_dir / "reports" / "causal_probe_decisions_used.json", decision_payload)
        write_json(state.output_dir / "reports" / "catf_v2" / "causal_probe_decisions_used.json", decision_payload)
    if bool(event.get("sample_weighting_allowed", False)):
        event = maybe_activate_sampler_only_dataloader(state, event, decision_payload or {}, epoch_num=epoch_num)
    if hasattr(controller, "policy"):
        controller.policy.matrix = deepcopy(gated_policy)
    if controller.history:
        record = controller.history[-1]
        record["catf_causal_probe"] = True
        record["causal_probe_event"] = deepcopy(event)
        record["accepted_policy_before_causal_probe"] = deepcopy(record.get("accepted_policy", policy))
        record["accepted_policy"] = deepcopy(gated_policy)
        record["new_policy"] = deepcopy(gated_policy)
        record["active_classes"] = active_class_ids(gated_policy)
        record["frozen_classes"] = frozen_class_ids(gated_policy)
        if not bool(event.get("sample_router_allowed", False)):
            record["guard_triggered"] = list(
                dict.fromkeys(list(record.get("guard_triggered", []) or []) + ["causal_probe_no_candidate_passed"])
            )
        elif "causal_probe_no_candidate_passed" in (record.get("guard_triggered", []) or []):
            record["guard_triggered"] = [
                item for item in (record.get("guard_triggered", []) or [])
                if item != "causal_probe_no_candidate_passed"
            ]
    state.causal_probe_events.append(deepcopy(event))
    for base in (state.output_dir / "reports", state.output_dir / "reports" / "catf_v2"):
        write_json(base / f"policy_matrix_epoch_{epoch_num}_after_causal_probe.json", gated_policy)
        write_json(base / "causal_probe_events.json", {"events": state.causal_probe_events})
    if hasattr(controller, "_write_histories"):
        controller._write_histories(gated_policy)
    return gated_policy


def maybe_activate_sampler_only_dataloader(
    state: InLoopFeedbackState,
    event: dict[str, Any],
    decision_payload: dict[str, Any],
    *,
    epoch_num: int,
) -> dict[str, Any]:
    event = deepcopy(event)
    enabled = bool(getattr(state.args, "sampler_only_enabled", False))
    event["sampler_only_enabled"] = enabled
    if not enabled:
        event["sample_weighting_effective"] = False
        event["sample_weighting_status"] = "pending_dataloader_support"
        event["sampler_only_effective"] = False
        event["weighted_sampler_enabled"] = False
        event["weighted_index_list_enabled"] = False
        return event
    debug_dir = PROJECT_ROOT / "outputs" / "debug" / "cp_catf_sampler_only_dataloader_impl"
    try:
        artifacts = build_sampler_only_artifacts(
            data_yaml=state.args.data,
            decision_payload=decision_payload,
            epoch_num=epoch_num,
            output_dir=state.output_dir,
            debug_dir=debug_dir,
        )
        activation = activate_sampler_only_weighted_index_list(state.context, artifacts)
    except Exception as exc:
        event.update(
            {
                "action": "sampler_only_pending",
                "sample_weight_map_generated": False,
                "sample_weighting_effective": False,
                "sample_weighting_status": "pending_sampler_only_error",
                "sampler_only_effective": False,
                "weighted_sampler_enabled": False,
                "weighted_index_list_enabled": False,
                "sampler_only_error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(state.output_dir / "reports" / f"sampler_only_dataloader_status_epoch_{epoch_num}.json", event)
        return event
    summary = artifacts.get("summary", {})
    paths = artifacts.get("paths", {})
    effective = bool(activation.get("sampler_only_effective", False))
    event.update(
        {
            "action": "sampler_only_effective" if effective else "sampler_only_pending",
            "sample_weight_map_generated": bool(summary.get("sample_weight_map_generated", False)),
            "sample_weighting_effective": effective,
            "sample_weighting_status": str(activation.get("sample_weighting_status", "effective_weighted_index_list" if effective else "pending_dataloader_support")),
            "sampler_only_effective": effective,
            "weighted_sampler_enabled": bool(activation.get("weighted_sampler_enabled", False)),
            "weighted_index_list_enabled": bool(activation.get("weighted_index_list_enabled", False)),
            "weighted_train_core_images_count": int(summary.get("weighted_train_core_images_count", 0) or 0),
            "sampled_distribution_changed": bool(summary.get("sampled_distribution_changed", False)),
            "weighted_train_indices_count": int((artifacts.get("weighted_train_indices") or {}).get("weighted_train_indices_count", 0) or 0),
            "sampler_only_extra_sample_count": int(summary.get("extra_sample_count", 0) or 0),
            "sampler_only_loader_reset": bool(activation.get("loader_reset", False)),
            "sampler_only_blocker": activation.get("blocker"),
            "sample_weight_map_path": str(paths.get("legacy_sample_weight_map_epoch_" + str(int(epoch_num))) or paths.get("legacy_sample_weight_map_latest") or paths.get("sample_weight_map_latest") or ""),
            "sample_weight_map_debug_path": str(paths.get("sample_weight_map_latest") or ""),
            "weighted_train_indices_path": str(paths.get("weighted_train_indices_epoch_" + str(int(epoch_num))) or paths.get("weighted_train_indices_latest") or ""),
            "weighted_train_indices_debug_path": str(paths.get("weighted_train_indices_latest") or ""),
            "sampled_distribution_before_after_path": str(
                paths.get("sampled_distribution_before_after_epoch_" + str(int(epoch_num)))
                or paths.get("sampled_distribution_before_after_latest")
                or ""
            ),
            "sampled_distribution_before_after_debug_path": str(paths.get("sampled_distribution_before_after_latest") or ""),
        }
    )
    write_json(state.output_dir / "reports" / f"sampler_only_dataloader_status_epoch_{epoch_num}.json", event)
    write_json(state.output_dir / "reports" / "catf_v2" / f"sampler_only_dataloader_status_epoch_{epoch_num}.json", event)
    return event


def activate_sampler_only_weighted_index_list(context: OnlineTrainingContext, artifacts: dict[str, Any]) -> dict[str, Any]:
    weighted_payload = artifacts.get("weighted_train_indices") or {}
    summary = artifacts.get("summary") or {}
    weighted_indices = [int(index) for index in (weighted_payload.get("weighted_train_indices") or [])]
    activation = {
        "weighted_sampler_enabled": False,
        "weighted_index_list_enabled": False,
        "sampler_only_effective": False,
        "sample_weighting_status": "pending_dataloader_support",
        "loader_reset": False,
        "blocker": None,
    }
    dataset = getattr(context, "train_dataset", None)
    if dataset is None:
        activation["blocker"] = "OnlineTrainingContext.train_dataset is None; trainer build_dataset did not expose the train dataset"
        context.sampler_only_status = str(activation["sample_weighting_status"])
        return activation
    if not hasattr(dataset, "set_weighted_indices"):
        activation["blocker"] = f"{type(dataset).__name__} does not expose set_weighted_indices"
        context.sampler_only_status = str(activation["sample_weighting_status"])
        return activation
    if not weighted_indices:
        activation["blocker"] = "weighted_train_indices is empty"
        context.sampler_only_status = str(activation["sample_weighting_status"])
        return activation
    if int(summary.get("weighted_train_core_images_count", 0) or 0) <= 0:
        activation["blocker"] = "sample_weight_map contains no train_core images with weight > 1"
        context.sampler_only_status = str(activation["sample_weighting_status"])
        return activation
    dataset.set_weighted_indices(
        weighted_indices,
        metadata={
            "sample_weight_map": artifacts.get("sample_weight_map", {}),
            "sampled_distribution_before_after": artifacts.get("sampled_distribution_before_after", {}),
            "paths": artifacts.get("paths", {}),
        },
    )
    loader = getattr(context, "train_loader", None)
    if loader is not None and hasattr(loader, "reset"):
        loader.reset()
        activation["loader_reset"] = True
    elif loader is not None:
        activation["blocker"] = f"{type(loader).__name__} does not expose reset(); weighted dataset mapping installed but current iterator was not reset"
        context.sampler_only_status = str(activation["sample_weighting_status"])
        return activation
    activation.update(
        {
            "weighted_index_list_enabled": True,
            "sampler_only_effective": bool(summary.get("sampler_only_effective", False)),
            "sample_weighting_status": "effective_weighted_index_list"
            if bool(summary.get("sampler_only_effective", False))
            else "pending_distribution_unchanged",
        }
    )
    context.weighted_sampler_enabled = False
    context.weighted_index_list_enabled = bool(activation["weighted_index_list_enabled"])
    context.sampler_only_effective = bool(activation["sampler_only_effective"])
    context.sampler_only_status = str(activation["sample_weighting_status"])
    context.sample_weight_map_generated = bool(summary.get("sample_weight_map_generated", False))
    context.weighted_train_core_images_count = int(summary.get("weighted_train_core_images_count", 0) or 0)
    context.sampled_distribution_changed = bool(summary.get("sampled_distribution_changed", False))
    context.sampler_only_artifact_paths = dict(artifacts.get("paths", {}))
    return activation


def load_offline_probe_decision_if_requested(state: InLoopFeedbackState, *, epoch_num: int) -> dict[str, Any] | None:
    if not bool(getattr(state.args, "use_offline_probe_decisions", False)):
        return None
    if state.offline_probe_decision_schedule is not None:
        payload = state.offline_probe_decision_schedule.get(int(epoch_num))
        if payload is None:
            return None
        return deepcopy(payload)
    if state.offline_probe_decision is not None:
        return deepcopy(state.offline_probe_decision)
    path = resolve_offline_probe_decision_path(state.args)
    if not path.exists():
        raise FileNotFoundError(f"Offline causal probe decision file not found: {path}")
    payload = read_json(path)
    schedule = offline_decision_schedule_from_payload(payload)
    if schedule:
        state.offline_probe_decision_schedule = schedule
        return deepcopy(schedule.get(int(epoch_num)))
    state.offline_probe_decision = deepcopy(payload)
    return payload


def offline_decision_schedule_from_payload(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw = payload.get("epoch_decisions") or payload.get("decisions_by_epoch") or {}
    out: dict[int, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                epoch = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                item = deepcopy(value)
                item.setdefault("epoch", epoch)
                out[epoch] = item
    elif isinstance(raw, list):
        for value in raw:
            if not isinstance(value, dict):
                continue
            try:
                epoch = int(value.get("epoch"))
            except (TypeError, ValueError):
                continue
            out[epoch] = deepcopy(value)
    return out


def build_paper_probe_decision_payload(
    state: InLoopFeedbackState,
    controller: ClassAwareCATFController,
    policy: dict[str, Any],
    *,
    epoch_num: int,
) -> dict[str, Any]:
    latest_record = controller.history[-1] if controller.history else {}
    per_class = {}
    per_class_path = latest_record.get("per_class_diagnosis_path")
    if per_class_path and Path(per_class_path).exists():
        per_class = read_json(Path(per_class_path))
    attribution = {}
    attribution_path = latest_record.get("issue_attribution_path")
    if attribution_path and Path(attribution_path).exists():
        attribution = read_json(Path(attribution_path))
    class_rows = merge_per_class_and_attribution(per_class, attribution)
    active_rows = [
        row for row in class_rows
        if int(row.get("class_id", -1)) in set(active_class_ids(policy))
    ]
    if not active_rows:
        active_rows = [
            row for row in class_rows
            if bool(row.get("strong_update_allowed", False)) and not bool(row.get("no_aug_class", False))
        ][: int(getattr(state.args, "top_k_active_classes", 2))]
    decision_payload = build_probe_decision_from_rows(
        active_rows=active_rows,
        context_rows=class_rows,
        policy=policy,
        epoch_num=epoch_num,
        mode="paper",
        policy_selection_data=str(policy_selection_data_yaml(state.args).resolve()),
        policy_selection_source=policy_selection_source(state.args),
    )
    decision_payload["paper_probe_mode"] = True
    decision_payload["development_probe_uses_existing_val_diagnostics"] = False
    decision_payload["final_val_used_for_policy_selection"] = False
    decision_payload["forbid_final_val_policy_selection"] = bool(getattr(state.args, "forbid_final_val_policy_selection", False))
    decision_payload["probe_source"] = str(getattr(state.args, "probe_source", "train_probe_split"))
    decision_payload["policy_selection_data_yaml"] = str(policy_selection_data_yaml(state.args).resolve())
    decision_payload["sampler_only_focus_rows"] = [sampler_only_row_payload(row) for row in active_rows]
    decision_payload["sampler_only_context_rows"] = [sampler_only_row_payload(row) for row in class_rows]
    decision_payload["riskguard_interpretation"] = {
        "riskguard_used_as_final_rule": False,
        "riskguard_role": "audit_debug_prior_only",
    }
    return decision_payload


def sampler_only_row_payload(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "class_id",
        "class_name",
        "dominant_issue",
        "secondary_issues",
        "strong_update_allowed",
        "oversampling_candidate",
        "copy_paste_candidate",
        "threshold_calibration_candidate",
        "no_aug_class",
        "domain_high_fp_prior",
        "high_fp_guarded",
        "high_fp_prior",
        "stable_class",
        "low_recall",
        "low_support",
        "low_support_only",
        "low_support_class",
        "low_contrast_fn",
        "texture_boundary_weak",
        "weak_localization",
        "high_fp",
        "support_level",
        "evidence_count",
        "diagnosis_confidence",
        "fn_count",
        "fp_count",
        "FN",
        "FP",
        "TP",
        "Precision",
        "Recall",
        "AP50",
        "AP50_95",
        "train_instances",
        "val_instances",
    ]
    return {key: deepcopy(row.get(key)) for key in keys if key in row}


def merge_per_class_and_attribution(per_class: dict[str, Any], attribution: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    per_class_rows = (per_class.get("classes") or {}) if isinstance(per_class, dict) else {}
    attr_rows = (attribution.get("classes") or {}) if isinstance(attribution, dict) else {}
    for class_key, row in per_class_rows.items():
        merged = deepcopy(row)
        attr = {}
        if isinstance(attr_rows, dict):
            attr = attr_rows.get(str(class_key), {})
            if not attr:
                try:
                    attr = attr_rows.get(int(class_key), {})
                except (TypeError, ValueError):
                    attr = {}
        if isinstance(attr, dict):
            merged.update({f"attribution_{key}": value for key, value in attr.items()})
            if attr.get("dominant_issue") is not None:
                merged["dominant_issue"] = attr.get("dominant_issue")
            if attr.get("primary_issue") is not None:
                merged.setdefault("dominant_issue", attr.get("primary_issue"))
        if "dominant_issue" not in merged:
            merged["dominant_issue"] = infer_dominant_issue_from_row(merged)
        rows.append(merged)
    return rows


def infer_dominant_issue_from_row(row: dict[str, Any]) -> str:
    ordered_flags = [
        ("texture_boundary_weak", "texture_boundary_weak"),
        ("low_contrast_fn", "low_contrast_fn"),
        ("weak_localization", "weak_localization"),
        ("small_object_fn", "small_object_fn"),
        ("edge_object_fn", "edge_object_fn"),
        ("dark_fn", "dark_fn"),
        ("low_recall", "low_recall"),
        ("high_fp", "high_fp"),
    ]
    for flag, issue in ordered_flags:
        if bool(row.get(flag, False)):
            return issue
    return "none"


def build_probe_decision_from_rows(
    *,
    active_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]] | None = None,
    policy: dict[str, Any],
    epoch_num: int,
    mode: str,
    policy_selection_data: str,
    policy_selection_source: str,
) -> dict[str, Any]:
    catalog = candidate_policy_catalog()
    risk_context_rows = list(context_rows or active_rows)
    candidate_ids = [
        "candidate_policy_1_roi_texture",
        "candidate_policy_2_roi_low_contrast",
        "candidate_policy_3_sampler_only",
    ]
    evaluations = []
    for candidate_id in candidate_ids:
        candidate = catalog[candidate_id]
        row = best_row_for_candidate(active_rows, candidate)
        if row is None:
            benefit = {"fn_recovery_rate": 0.0, "localization_iou_gain": 0.0, "low_conf_tp_conf_gain": 0.0}
            risk = {
                "fp_increase_rate": 0.0,
                "high_fp_spillover_rate": 0.0,
                "non_active_regression_rate": 0.0,
                "ok_class_false_activation": 0.0,
                "bbox_instability_rate": 0.0,
                "estimated_precision_drop": 0.0,
                "non_active_fp_delta": 0.0,
                "high_confidence_fp_delta": 0.0,
            }
            evidence_count = 0
            diagnosis_confidence = 0.0
            class_id = -1
        else:
            benefit = paper_probe_benefit_metrics(row, candidate)
            risk = paper_probe_risk_metrics(row, risk_context_rows, candidate)
            evidence_count = int(row.get("evidence_count", 0) or 0)
            diagnosis_confidence = float(row.get("diagnosis_confidence", 0.0) or 0.0)
            class_id = int(row.get("class_id", -1))
        probe_set = build_probe_set(
            candidate_class_id=class_id,
            candidate_policy_id=candidate_id,
            diagnosis_record=row or {},
            issue_record=row or {},
            config=ProbeConfig(mode=mode, development_probe_uses_existing_val_diagnostics=False),
        )
        evaluation = evaluate_candidate_policy(
            candidate_policy=candidate,
            probe_set=probe_set,
            benefit_metrics=benefit,
            risk_metrics=risk,
            evidence_count=evidence_count,
            diagnosis_confidence=diagnosis_confidence,
        )
        evaluation["policy_selection_source"] = policy_selection_source
        evaluation["policy_selection_data"] = policy_selection_data
        evaluations.append(evaluation)
    selected = select_best_candidate(evaluations)
    selected_action = str((selected.get("decision") or {}).get("decision") or (selected.get("candidate_policy") or {}).get("action") or "unknown")
    return {
        "epoch": int(epoch_num),
        "mode": mode,
        "decision_basis": "paper_probe_split_run_specific_benefit_risk_metrics",
        "policy_selection_source": policy_selection_source,
        "policy_selection_data": policy_selection_data,
        "selected_candidate_policy_id": selected.get("candidate_policy_id"),
        "selected_candidate_action": selected_action,
        "selected_candidate": selected,
        "candidate_evaluations": evaluations,
        "image_augmentation_rejected": not bool((selected.get("decision") or {}).get("image_modification_allowed", False)),
        "strict_image_noop": not bool((selected.get("decision") or {}).get("image_modification_allowed", False)),
        "sampler_only_selected": selected_action == "sampler_only",
        "seed_specific_rule": False,
        "fixed_class_id_specific_rule": False,
        "dataset_specific_rule": False,
    }


def best_row_for_candidate(active_rows: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any] | None:
    if not active_rows:
        return None
    ops = set(candidate.get("op_list") or [])
    if "gamma" in ops:
        preferred = [row for row in active_rows if str(row.get("dominant_issue", "")).startswith("low_contrast")]
    elif "sharpen_mild" in ops or "local_contrast" in ops:
        preferred = [
            row for row in active_rows
            if str(row.get("dominant_issue", "")) in {"texture_boundary_weak", "weak_localization", "low_contrast_fn"}
        ]
    else:
        preferred = active_rows
    rows = preferred or active_rows
    return max(
        rows,
        key=lambda row: (
            int(row.get("evidence_count", 0) or 0),
            float(row.get("diagnosis_confidence", 0.0) or 0.0),
            int(row.get("fn_count", 0) or 0),
        ),
    )


def paper_probe_benefit_metrics(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    evidence = max(1.0, float(row.get("evidence_count", 0) or row.get("val_instances", 0) or 1))
    fn_rate = float(row.get("fn_count", 0) or 0) / evidence
    issue = str(row.get("dominant_issue", ""))
    ap50 = row_float(row, "AP50", "ap50")
    ap95 = row_float(row, "AP50_95", "ap50_95", "ap95")
    texture_candidate = "sharpen_mild" in set(candidate.get("op_list") or [])
    low_contrast_candidate = "gamma" in set(candidate.get("op_list") or []) or "local_contrast" in set(candidate.get("op_list") or [])
    fn_recovery = min(0.08, max(0.0, fn_rate * 0.12))
    if issue in {"texture_boundary_weak", "low_contrast_fn"}:
        fn_recovery += 0.015
    if not texture_candidate and issue == "texture_boundary_weak":
        fn_recovery *= 0.5
    localization_gain = 0.0
    if texture_candidate and (issue in {"texture_boundary_weak", "weak_localization"} or ap50 - ap95 > 0.15):
        localization_gain = min(0.03, max(0.005, (ap50 - ap95) * 0.08))
    low_conf_gain = 0.0
    if low_contrast_candidate and issue in {"low_contrast_fn", "texture_boundary_weak"}:
        low_conf_gain = 0.015
    return {
        "fn_recovery_rate": float(min(0.12, fn_recovery)),
        "localization_iou_gain": float(localization_gain),
        "low_conf_tp_conf_gain": float(low_conf_gain),
    }


def paper_probe_risk_metrics(
    row: dict[str, Any],
    context_rows: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, float]:
    high_fp = bool(row.get("high_fp_guarded", False) or row.get("high_fp_prior", False))
    no_aug = bool(row.get("no_aug_class", False))
    stable = bool(row.get("stable_class", False))
    low_support = bool(row.get("low_support_only", False))
    issue = str(row.get("dominant_issue", ""))
    ap50 = row_float(row, "AP50", "ap50")
    ap95 = row_float(row, "AP50_95", "ap50_95", "ap95")
    precision = row_float(row, "Precision", "precision")
    texture_candidate = "sharpen_mild" in set(candidate.get("op_list") or [])
    target_class_id = int(row.get("class_id", -1))
    non_active_rows = [
        other for other in context_rows
        if int(other.get("class_id", -1)) != target_class_id
    ]
    non_active_risky = 0
    high_confidence_fp_risky = 0
    for other in non_active_rows:
        other_precision = row_float(other, "Precision", "precision")
        other_fp = row_float(other, "FP", "fp", "fp_count")
        other_high_fp = bool(other.get("high_fp_guarded", False) or other.get("high_fp_prior", False))
        other_no_aug = bool(other.get("no_aug_class", False))
        other_stable = bool(other.get("stable_class", False))
        if other_no_aug or other_high_fp or other_stable or other_precision < 0.65 or other_fp >= 5:
            non_active_risky += 1
        if other_no_aug or other_high_fp or other_fp >= 10:
            high_confidence_fp_risky += 1
    estimated_precision_drop = 0.0
    non_active_fp_delta = 0.0
    high_confidence_fp_delta = 0.0
    if texture_candidate:
        estimated_precision_drop = min(0.05, 0.003 * non_active_risky + max(0.0, 0.70 - precision) * 0.01)
        non_active_fp_delta = min(0.05, 0.006 * non_active_risky)
        high_confidence_fp_delta = min(0.05, 0.006 * high_confidence_fp_risky)
    return {
        "fp_increase_rate": 0.04 if high_fp or no_aug else max(0.0, 0.02 - precision * 0.02),
        "high_fp_spillover_rate": 0.03 if high_fp or no_aug else 0.005 * non_active_risky,
        "non_active_regression_rate": 0.025 if texture_candidate and stable else 0.006 * non_active_risky,
        "ok_class_false_activation": 1.0 if no_aug else 0.0,
        "bbox_instability_rate": min(0.03, max(0.0, (ap50 - ap95) * 0.05)) if texture_candidate and issue == "texture_boundary_weak" else 0.0,
        "estimated_precision_drop": estimated_precision_drop,
        "non_active_fp_delta": non_active_fp_delta,
        "high_confidence_fp_delta": high_confidence_fp_delta,
        "low_support_risk": 0.02 if low_support else 0.0,
    }


def row_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def resolve_offline_probe_decision_path(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "offline_probe_decisions_file", None)
    if explicit:
        return Path(explicit).resolve()
    base = Path(getattr(args, "offline_probe_decisions_dir", PROJECT_ROOT / "outputs/experiments/catf_v2_causal_probe")).resolve()
    return base / f"seed_{int(getattr(args, 'seed', 0))}" / "probe_decisions.json"


def apply_offline_probe_decision_to_policy(
    policy: dict[str, Any],
    decision_payload: dict[str, Any],
    *,
    epoch_num: int,
    output_dir: Path,
    disable_sampler_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = decision_payload.get("selected_candidate") or {}
    decision = selected.get("decision") or {}
    selected_policy = selected.get("candidate_policy") or {}
    selected_policy_id = str(decision_payload.get("selected_candidate_policy_id") or selected.get("candidate_policy_id") or "unknown")
    selected_action = str(decision_payload.get("selected_candidate_action") or decision.get("decision") or "unknown")
    op_whitelist = [str(item) for item in (selected_policy.get("op_list") or [])]
    image_allowed = bool(decision.get("image_modification_allowed", False))
    sampler_allowed = bool(decision.get("sample_weighting_allowed", False)) or str(decision.get("decision")) == "sampler_only"
    if bool(disable_sampler_only):
        sampler_allowed = False
    probe_set = selected.get("probe_set") or {}
    candidate_class_id = safe_int(probe_set.get("class_id"), default=-1)
    event = {
        "epoch": int(epoch_num),
        "catf_causal_probe": True,
        "source": "offline_probe_decisions",
        "selected_candidate_policy_id": selected_policy_id,
        "selected_candidate_action": selected_action,
        "selected_causal_score": selected.get("causal_score"),
        "candidate_class_id": candidate_class_id,
        "op_whitelist": op_whitelist,
        "image_modification_allowed": image_allowed,
        "sample_weighting_allowed": sampler_allowed,
        "sample_weighting_effective": bool(decision.get("sample_weighting_effective", False)),
        "sample_weighting_status": str(decision.get("sample_weighting_status", "not_requested" if not sampler_allowed else "pending_dataloader_support")),
        "probe_reject_image_aug": bool(decision_payload.get("image_augmentation_rejected", False) or not image_allowed),
        "development_probe_uses_existing_val_diagnostics": bool(decision_payload.get("development_probe_uses_existing_val_diagnostics", False)),
        "paper_probe_mode": bool(decision_payload.get("paper_probe_mode", False)),
        "policy_selection_source": str(decision_payload.get("policy_selection_source", "offline_probe_decisions")),
        "policy_selection_data": str(decision_payload.get("policy_selection_data") or decision_payload.get("policy_selection_data_yaml") or ""),
        "final_val_used_for_policy_selection": bool(decision_payload.get("final_val_used_for_policy_selection", False)),
        "forbid_final_val_policy_selection": bool(decision_payload.get("forbid_final_val_policy_selection", False)),
        "decision_basis": str(decision_payload.get("decision_basis", "run_specific_probe_benefit_risk_metrics")),
        "seed_specific_rule": False,
        "fixed_class_id_specific_rule": False,
        "riskguard_used_as_final_rule": bool((decision_payload.get("riskguard_interpretation") or {}).get("riskguard_used_as_final_rule", False)),
        "sampler_only_disabled": bool(disable_sampler_only),
        "original_candidate_policy_id": decision_payload.get("original_candidate_policy_id"),
        "downgraded_to_weak_roi_texture": bool(decision_payload.get("downgraded_to_weak_roi_texture", False)),
        "precision_aware_gate_passed": decision_payload.get("precision_aware_gate_passed"),
        "non_active_regression_gate_passed": decision_payload.get("non_active_regression_gate_passed"),
        "weak_rejection_reasons": decision_payload.get("weak_rejection_reasons"),
        "final_replay_action": decision_payload.get("final_replay_action"),
        "risk_level": decision_payload.get("risk_level"),
        "preserve_original": bool(decision_payload.get("preserve_original", False)),
        "weak_only_for_moderate_risk": bool(decision_payload.get("weak_only_for_moderate_risk", False)),
    }
    preserve_requested = bool(decision_payload.get("preserve_original", False)) or selected_policy_id == "candidate_policy_preserve_original" or selected_action == "preserve_original"
    if preserve_requested:
        gated_policy, preserve_meta = apply_preserve_original_policy(policy, decision_payload, epoch_num=epoch_num)
        event["action"] = "preserve_original"
        event["reason"] = str(decision_payload.get("decision_reason") or "preserve_fixed_original_policy")
        event["image_modification_allowed"] = True
        event["probe_reject_image_aug"] = False
        event["sample_weighting_allowed"] = False
        event["sample_weighting_effective"] = False
        event["sample_weighting_status"] = "not_requested"
        event["sample_router_allowed"] = bool(preserve_meta.get("installed_ops", []))
        event["preserved_active_classes"] = active_class_ids(gated_policy)
        event["preserve_expected_active_classes"] = preserve_meta.get("expected_active_classes", [])
        event["preserve_installed_active_classes"] = preserve_meta.get("installed_active_classes", [])
        event["preserve_installed_ops"] = preserve_meta.get("installed_ops", [])
        event["preserve_cleared_classes"] = preserve_meta.get("cleared_classes", [])
        event["preserve_missing_classes"] = preserve_meta.get("missing_classes", [])
        event["preserve_no_aug_blocked_classes"] = preserve_meta.get("no_aug_blocked_classes", [])
        event["preserve_source"] = preserve_meta.get("source", "offline_replay_expected_fixed_policy")
        event["preserve_epoch_exact"] = bool(preserve_meta.get("epoch_exact", False))
        event["preserve_policy_empty_for_epoch"] = bool(preserve_meta.get("policy_empty_for_epoch", False))
        event["preserved_original_fixed_op_list"] = decision_payload.get("original_fixed_op_list")
        event["preserved_original_fixed_prob_strength"] = decision_payload.get("original_fixed_prob_strength")
        event["preserved_epoch_exact_fixed_op_list"] = decision_payload.get("epoch_exact_fixed_op_list")
        event["preserved_epoch_exact_fixed_prob_strength"] = decision_payload.get("epoch_exact_fixed_prob_strength")
        event["candidate_policy_injected"] = bool(preserve_meta.get("installed_ops", []))
        event["candidate_ops_injected"] = preserve_meta.get("installed_ops", [])
        return gated_policy, event
    if bool(disable_sampler_only) and str(decision.get("decision")) == "sampler_only":
        event["sampler_only_blocked_by_image_only_mainline"] = True
    if not image_allowed:
        reason = "offline_causal_probe_rejected_image_augmentation"
        if sampler_allowed:
            reason = "offline_causal_probe_sampler_only_image_noop"
        gated_policy = force_noop_policy(policy, reason=reason)
        event["action"] = "sampler_only_pending" if sampler_allowed else "strict_no_op"
        event["reason"] = reason
        if sampler_allowed:
            sample_weight_map = {
                "cp_catf_sampler_only": {
                    "enabled": True,
                    "sample_weighting_effective": False,
                    "sample_weighting_status": "pending_dataloader_support",
                    "image_modification": False,
                    "selected_candidate_policy_id": selected_policy_id,
                }
            }
            path = output_dir / "reports" / f"cp_catf_sample_weight_map_epoch_{epoch_num}.json"
            catf_path = output_dir / "reports" / "catf_v2" / f"cp_catf_sample_weight_map_epoch_{epoch_num}.json"
            write_json(path, sample_weight_map)
            write_json(catf_path, sample_weight_map)
            event["sample_weight_map_path"] = str(path)
        return gated_policy, event

    gated_policy, activation = activate_causal_probe_candidate_policy(
        policy,
        selected_candidate=selected,
        op_whitelist=op_whitelist,
        epoch_num=epoch_num,
    )
    event["action"] = "accept_offline_probe_candidate"
    event["reason"] = "offline_causal_probe_candidate_accepted"
    event["candidate_class_id"] = activation.get("candidate_class_id", candidate_class_id)
    event["candidate_ops_injected"] = activation.get("candidate_ops_injected", [])
    event["candidate_policy_injected"] = bool(activation.get("candidate_policy_injected", False))
    event["sample_router_allowed"] = bool(activation.get("sample_router_allowed", False))
    event["activation_noop_reason"] = activation.get("noop_reason")
    event["weak_image_aug"] = bool(activation.get("weak_image_aug", False))
    event["attenuation_ratio"] = activation.get("attenuation_ratio")
    event["retained_op"] = activation.get("retained_op")
    event["original_prob"] = activation.get("original_prob")
    event["original_strength"] = activation.get("original_strength")
    event["weak_prob"] = activation.get("weak_prob")
    event["weak_strength"] = activation.get("weak_strength")
    event["max_aug_samples_per_interval"] = activation.get("max_aug_samples_per_interval")
    event["target_high_fp_guard_overridden_by_weak_gate"] = bool(
        activation.get("target_high_fp_guard_overridden_by_weak_gate", False)
    )
    if not event["sample_router_allowed"]:
        noop_reason = str(activation.get("noop_reason") or "causal_probe_accept_no_executable_policy")
        gated_policy = force_noop_policy(gated_policy, reason=noop_reason)
        event["action"] = "strict_no_op"
        event["reason"] = noop_reason
        event["image_modification_allowed"] = False
        event["probe_reject_image_aug"] = True
    return gated_policy, event


PRESERVE_CLASS_OP_RE = re.compile(r"^\s*c(?P<class_id>\d+):(?P<issue>[^:;]+):(?P<ops>.+?)\s*$")
PRESERVE_OP_RE = re.compile(r"^\s*(?P<op>[A-Za-z0-9_]+)@p=(?P<prob>[-+0-9.eE]+)\/s=(?P<strength>[-+0-9.eE]+)\s*$")
PRESERVE_PROB_STRENGTH_RE = re.compile(
    r"^\s*c(?P<class_id>\d+):(?P<op>[A-Za-z0-9_]+):(?P<prob>[-+0-9.eE]+)\/(?P<strength>[-+0-9.eE]+)\s*$"
)


def apply_preserve_original_policy(
    policy: dict[str, Any],
    decision_payload: dict[str, Any],
    *,
    epoch_num: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Install replayed fixed CATF-v2 image policy into the runtime matrix.

    Preserve-original is an execution action, not a candidate-selection label:
    it must not keep whatever the current online controller happened to propose
    for this epoch.  The replay payload carries the fixed CATF-v2 class/op
    policy that should be executable; this helper overlays that policy onto the
    matrix and clears stale active ops from non-preserved classes.
    """

    matrix = deepcopy(policy)
    epoch_exact = preserve_payload_has_epoch_exact_policy(decision_payload)
    expected_classes = parse_preserve_expected_classes(decision_payload)
    specs = parse_preserve_policy_specs(decision_payload)
    expected_classes.update(specs.keys())
    classes = matrix.setdefault("classes", {})
    installed_ops: list[dict[str, Any]] = []
    cleared_classes: list[int] = []
    missing_classes: list[int] = []
    no_aug_blocked_classes: list[int] = []

    for cid_text, class_policy in sorted(classes.items(), key=lambda item: int(item[0])):
        class_id = int(cid_text)
        if class_id in expected_classes:
            continue
        had_active_ops = bool(active_ops_from_policy(class_policy))
        if had_active_ops or class_policy.get("status") in {"active", "pending", "accepted"}:
            clear_class_policy_ops(class_policy)
            if class_policy.get("status") in {"active", "pending", "accepted"} and not bool(class_policy.get("no_aug_class", False)):
                class_policy["status"] = "observe"
                class_policy["state"] = "accepted"
                class_policy["preserve_original_cleared"] = True
                class_policy["preserve_original_clear_epoch"] = int(epoch_num)
            if had_active_ops:
                cleared_classes.append(class_id)

    for class_id in sorted(expected_classes):
        class_policy = classes.get(str(class_id))
        if not isinstance(class_policy, dict):
            missing_classes.append(int(class_id))
            continue
        if bool(class_policy.get("no_aug_class", False)):
            clear_class_policy_ops(class_policy)
            class_policy["status"] = "frozen"
            class_policy["state"] = "frozen"
            class_policy["frozen_reason"] = class_policy.get("frozen_reason") or "no_aug_class"
            no_aug_blocked_classes.append(int(class_id))
            continue

        class_spec = specs.get(class_id, {})
        clear_class_policy_ops(class_policy)
        class_policy["status"] = "active"
        class_policy["state"] = "accepted"
        class_policy["dominant_issue"] = class_spec.get("dominant_issue") or class_policy.get("dominant_issue") or "preserve_original"
        class_policy["secondary_issues"] = list(class_policy.get("secondary_issues") or [])
        class_policy["pending_since_epoch"] = None
        class_policy["frozen_reason"] = None
        class_policy["preserve_original"] = {
            "enabled": True,
            "epoch": int(epoch_num),
            "source": "offline_replay_expected_fixed_policy",
            "decision_reason": decision_payload.get("decision_reason"),
        }
        class_policy.pop("weak_image_aug", None)
        class_policy["causal_probe_selected"] = False
        class_policy["causal_probe_execution_source"] = "preserve_original_fixed_policy"
        guards = class_policy.setdefault("guards", {})
        # The replay already classified this candidate as low risk.  Preserve
        # should not be blocked by a newly generated weak/probe gate.
        guards["high_fp_guarded"] = False
        guards["precision_guard"] = False

        for op_name, values in sorted((class_spec.get("ops") or {}).items()):
            ops = class_policy.setdefault("ops", {})
            op = ops.get(str(op_name))
            if not isinstance(op, dict):
                continue
            prob = float(values.get("prob", 0.0) or 0.0)
            strength = float(values.get("strength", 0.0) or 0.0)
            op["prob"] = prob
            op["strength"] = strength
            op["preserve_original"] = True
            op["preserve_original_epoch"] = int(epoch_num)
            op.pop("weak_image_aug", None)
            op.pop("attenuation_ratio", None)
            installed_ops.append({"class_id": int(class_id), "op_name": str(op_name), "prob": prob, "strength": strength})

    metadata = {
        "source": "offline_replay_epoch_exact_fixed_policy" if epoch_exact else "offline_replay_expected_fixed_policy",
        "epoch_exact": bool(epoch_exact),
        "policy_empty_for_epoch": bool(epoch_exact and not expected_classes and not installed_ops),
        "expected_active_classes": sorted(int(cid) for cid in expected_classes),
        "installed_active_classes": active_class_ids(matrix),
        "installed_ops": installed_ops,
        "cleared_classes": sorted(cleared_classes),
        "missing_classes": sorted(missing_classes),
        "no_aug_blocked_classes": sorted(no_aug_blocked_classes),
    }
    return matrix, metadata


def parse_preserve_expected_classes(decision_payload: dict[str, Any]) -> set[int]:
    if preserve_payload_has_epoch_exact_policy(decision_payload):
        return parse_class_id_list(
            decision_payload.get("epoch_exact_fixed_active_class", decision_payload.get("fixed_epoch_active_class", ""))
        )
    selected = decision_payload.get("selected_candidate") or {}
    probe_set = selected.get("probe_set") or {}
    out: set[int] = set()
    active = probe_set.get("active_classes")
    if isinstance(active, list):
        for value in active:
            parsed = safe_int(value, default=-1)
            if parsed >= 0:
                out.add(parsed)
    elif active not in (None, ""):
        out.update(parse_class_id_list(active))
    out.update(parse_class_id_list(decision_payload.get("original_fixed_active_class")))
    class_id = safe_int(probe_set.get("class_id"), default=-1)
    if not out and class_id >= 0:
        out.add(class_id)
    return out


def preserve_payload_has_epoch_exact_policy(decision_payload: dict[str, Any]) -> bool:
    return any(
        key in decision_payload
        for key in (
            "epoch_exact_fixed_active_class",
            "epoch_exact_fixed_op_list",
            "epoch_exact_fixed_prob_strength",
            "fixed_epoch_active_class",
            "fixed_epoch_op_list",
            "fixed_epoch_prob_strength",
        )
    )


def parse_class_id_list(value: Any) -> set[int]:
    out: set[int] = set()
    if value in (None, ""):
        return out
    for part in re.split(r"[;,|\s]+", str(value)):
        part = part.strip()
        if not part:
            continue
        if part.startswith("c") and part[1:].isdigit():
            part = part[1:]
        if part.isdigit():
            out.add(int(part))
    return out


def parse_preserve_policy_specs(decision_payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    specs: dict[int, dict[str, Any]] = {}
    if preserve_payload_has_epoch_exact_policy(decision_payload):
        op_list = str(decision_payload.get("epoch_exact_fixed_op_list", decision_payload.get("fixed_epoch_op_list", "")) or "")
        prob_strength = str(
            decision_payload.get(
                "epoch_exact_fixed_prob_strength",
                decision_payload.get("fixed_epoch_prob_strength", ""),
            )
            or ""
        )
    else:
        op_list = str(decision_payload.get("original_fixed_op_list") or "")
        prob_strength = str(decision_payload.get("original_fixed_prob_strength") or "")
    for chunk in [item.strip() for item in op_list.split(";") if item.strip()]:
        match = PRESERVE_CLASS_OP_RE.match(chunk)
        if not match:
            continue
        class_id = int(match.group("class_id"))
        class_spec = specs.setdefault(class_id, {"dominant_issue": match.group("issue").strip(), "ops": {}})
        class_spec["dominant_issue"] = match.group("issue").strip()
        for op_chunk in [item.strip() for item in match.group("ops").split(",") if item.strip()]:
            op_match = PRESERVE_OP_RE.match(op_chunk)
            if not op_match:
                continue
            class_spec["ops"][op_match.group("op")] = {
                "prob": float(op_match.group("prob")),
                "strength": float(op_match.group("strength")),
            }

    for chunk in [item.strip() for item in prob_strength.split(";") if item.strip()]:
        match = PRESERVE_PROB_STRENGTH_RE.match(chunk)
        if not match:
            continue
        class_id = int(match.group("class_id"))
        class_spec = specs.setdefault(class_id, {"dominant_issue": None, "ops": {}})
        class_spec["ops"][match.group("op")] = {
            "prob": float(match.group("prob")),
            "strength": float(match.group("strength")),
        }
    return specs


def clear_class_policy_ops(class_policy: dict[str, Any]) -> None:
    for op in (class_policy.get("ops") or {}).values():
        op["prob"] = 0.0
        op["strength"] = 0.0


CAUSAL_PROBE_EXECUTION_FLOORS: dict[str, dict[str, float]] = {
    "sharpen_mild": {"prob": 0.20, "strength": 0.22},
    "local_contrast": {"prob": 0.18, "strength": 0.20},
    "gamma": {"prob": 0.15, "strength": 0.18},
    "clahe": {"prob": 0.15, "strength": 0.18},
}


def activate_causal_probe_candidate_policy(
    policy: dict[str, Any],
    *,
    selected_candidate: dict[str, Any],
    op_whitelist: list[str],
    epoch_num: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = restrict_policy_to_causal_probe_ops(policy, op_whitelist)
    probe_set = selected_candidate.get("probe_set") or {}
    class_id = safe_int(probe_set.get("class_id"), default=-1)
    metadata: dict[str, Any] = {
        "candidate_class_id": class_id,
        "candidate_ops_injected": [],
        "candidate_policy_injected": False,
        "sample_router_allowed": False,
        "noop_reason": None,
    }
    candidate_policy = selected_candidate.get("candidate_policy") or {}
    candidate_policy_id = str(selected_candidate.get("candidate_policy_id") or candidate_policy.get("policy_id") or "")
    weak_image_aug = bool(candidate_policy.get("weak_image_aug", False)) or candidate_policy_id == "candidate_policy_1b_weak_roi_texture"
    if class_id < 0:
        metadata["noop_reason"] = "causal_probe_accept_missing_target_class"
        return matrix, metadata
    classes = matrix.setdefault("classes", {})
    class_policy = classes.get(str(class_id))
    if not isinstance(class_policy, dict):
        metadata["noop_reason"] = "causal_probe_accept_target_class_not_in_policy"
        return matrix, metadata
    if bool(class_policy.get("no_aug_class", False)):
        metadata["noop_reason"] = "causal_probe_accept_target_no_aug_class"
        return matrix, metadata
    guards = class_policy.get("guards") or {}
    if bool(guards.get("high_fp_guarded", False)) and not weak_image_aug:
        metadata["noop_reason"] = "causal_probe_accept_target_high_fp_guarded"
        return matrix, metadata
    if bool(guards.get("high_fp_guarded", False)) and weak_image_aug:
        metadata["target_high_fp_guard_overridden_by_weak_gate"] = True

    class_policy["status"] = "active"
    class_policy["state"] = "accepted"
    class_policy["dominant_issue"] = class_policy.get("dominant_issue") or "causal_probe_accepted"
    class_policy["frozen_reason"] = None
    class_policy["causal_probe_selected"] = True
    class_policy["causal_probe_candidate_policy_id"] = selected_candidate.get("candidate_policy_id")
    class_policy["causal_probe_execution_source"] = "accepted_candidate_policy"
    if weak_image_aug:
        attenuation_ratio = float(candidate_policy.get("attenuation_ratio", 0.25) or 0.25)
        max_aug_samples = int(candidate_policy.get("max_aug_samples_per_interval", 16) or 16)
        retained_op = str((op_whitelist or candidate_policy.get("op_list") or [""])[0])
        op_whitelist = [retained_op] if retained_op else []
        class_policy["weak_image_aug"] = {
            "enabled": True,
            "candidate_policy_id": candidate_policy_id,
            "derived_from_policy_id": str(candidate_policy.get("derived_from_policy_id", "candidate_policy_1_roi_texture")),
            "attenuation_ratio": attenuation_ratio,
            "retained_op": retained_op,
            "max_aug_samples_per_interval": max_aug_samples,
            "interval_start_epoch": int(epoch_num),
            "precision_aware_gate_passed": bool((selected_candidate.get("decision") or {}).get("precision_aware_gate_passed", True)),
            "non_active_regression_gate_passed": bool(
                (selected_candidate.get("decision") or {}).get("non_active_regression_gate_passed", True)
            ),
        }
        metadata.update(
            {
                "weak_image_aug": True,
                "attenuation_ratio": attenuation_ratio,
                "retained_op": retained_op,
                "max_aug_samples_per_interval": max_aug_samples,
            }
        )
    ops = class_policy.setdefault("ops", {})
    for op_name in op_whitelist:
        op = ops.get(str(op_name))
        if not isinstance(op, dict):
            continue
        floor = CAUSAL_PROBE_EXECUTION_FLOORS.get(str(op_name), {"prob": 0.10, "strength": 0.10})
        max_prob = float(op.get("max_prob", floor["prob"]) or floor["prob"])
        max_strength = float(op.get("max_strength", max(floor["strength"], 0.10)) or max(floor["strength"], 0.10))
        if weak_image_aug:
            ratio = float(candidate_policy.get("attenuation_ratio", metadata.get("attenuation_ratio", 0.25)) or 0.25)
            prob = min(max_prob, max(0.0, float(floor["prob"]) * ratio))
            strength = min(max_strength, max(0.0, float(floor["strength"]) * ratio))
            metadata["original_prob"] = float(floor["prob"])
            metadata["original_strength"] = float(floor["strength"])
            metadata["weak_prob"] = prob
            metadata["weak_strength"] = strength
        else:
            prob = min(max_prob, max(float(op.get("prob", 0.0) or 0.0), float(floor["prob"])))
            strength = min(max_strength, max(float(op.get("strength", 0.0) or 0.0), float(floor["strength"])))
        op["prob"] = prob
        op["strength"] = strength
        op["causal_probe_selected"] = True
        if weak_image_aug:
            op["weak_image_aug"] = True
            op["attenuation_ratio"] = float(candidate_policy.get("attenuation_ratio", metadata.get("attenuation_ratio", 0.25)) or 0.25)
            op["original_prob"] = float(floor["prob"])
            op["original_strength"] = float(floor["strength"])
        metadata["candidate_ops_injected"].append({"op_name": str(op_name), "prob": prob, "strength": strength})

    metadata["candidate_policy_injected"] = bool(metadata["candidate_ops_injected"])
    metadata["sample_router_allowed"] = bool(active_ops_from_policy(class_policy))
    if not metadata["sample_router_allowed"]:
        metadata["noop_reason"] = "causal_probe_accept_no_active_ops_after_injection"
    return matrix, metadata


def safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def restrict_policy_to_causal_probe_ops(policy: dict[str, Any], op_whitelist: list[str]) -> dict[str, Any]:
    allowed = {str(item) for item in op_whitelist}
    matrix = deepcopy(policy)
    matrix.setdefault("causal_probe", {})["op_whitelist"] = sorted(allowed)
    matrix["causal_probe"]["policy_filter_active"] = True
    for class_policy in (matrix.get("classes") or {}).values():
        for op_name, op in (class_policy.get("ops") or {}).items():
            if str(op_name) in allowed:
                continue
            op["prob"] = 0.0
            op["strength"] = 0.0
        if class_policy.get("status") in {"active", "pending"} and not active_ops_from_policy(class_policy):
            class_policy["status"] = "observe"
            class_policy["state"] = "accepted"
            class_policy["causal_probe_filter_reason"] = "no_probe_allowed_ops_for_class"
    return matrix


def active_ops_from_policy(class_policy: dict[str, Any]) -> list[str]:
    out = []
    for op_name, op in (class_policy.get("ops") or {}).items():
        if float(op.get("prob", 0.0) or 0.0) > 0.0 and float(op.get("strength", 0.0) or 0.0) > 0.0:
            out.append(str(op_name))
    return out


def close_mosaic_start_epoch(args: argparse.Namespace) -> int | None:
    total_epochs = int(getattr(args, "epochs", 0) or 0)
    close_mosaic = 10
    start = total_epochs - close_mosaic
    # For short smoke runs, close_mosaic may cover the whole run. Do not let
    # that erase the burn-in audit path; the smoke still needs the epoch-5 check.
    if start <= int(getattr(args, "min_burnin_epoch", 5)):
        return None
    return start


def recent_metric_history(state: InLoopFeedbackState, window: int | None = None) -> list[dict[str, Any]]:
    records = [dict(record.get("metrics") or {}) for record in state.epoch_records if isinstance(record.get("metrics"), dict)]
    if window is None:
        return records
    return records[-max(1, int(window)) :]


def initial_policy_state() -> dict[str, Any]:
    return default_catf_policy()


def initial_catf_v2_policy_state(class_names: dict[int, str]) -> dict[str, Any]:
    return initial_policy_matrix(class_names)


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
        policy_metrics = policy_selection_metrics(state, diagnosis, metrics)
        reference_metrics = policy_reference_metrics_for_epoch(state, epoch_num, policy_metrics)
        if bool(getattr(state.args, "diagnosis_only", False)):
            record_diagnosis_only_feedback(state, trainer, epoch_num, policy_metrics, reference_metrics, diagnosis, old_policy)
            return
        if bool(getattr(state.args, "catf_noop", False)):
            record_diagnosis_only_feedback(
                state,
                trainer,
                epoch_num,
                policy_metrics,
                reference_metrics,
                diagnosis,
                old_policy,
                action="observe",
                catf_noop=True,
            )
            return
        if is_catf_v2(state.args):
            controller = state.feedback_controller
            if not isinstance(controller, ClassAwareCATFController):
                class_names = load_class_names_from_data_yaml(state.args.data)
                controller = ClassAwareCATFController(
                    state.policy_state,
                    history_dir=state.output_dir / "reports",
                    class_names=class_names,
                    train_instances=count_train_instances(state.args.data),
                    top_k_active_classes=int(state.args.top_k_active_classes),
                    top_m_ops_per_class=int(state.args.top_m_ops_per_class),
                    freeze_epoch=40,
                    threshold_calibration_report=bool(state.args.threshold_calibration_report),
                )
                state.feedback_controller = controller
            new_policy = controller.update(diagnosis, epoch=epoch_num, metrics=policy_metrics, reference_metrics=reference_metrics)
            latest_record = controller.history[-1] if controller.history else {}
            per_class = {}
            per_class_path = latest_record.get("per_class_diagnosis_path")
            if per_class_path and Path(per_class_path).exists():
                per_class = read_json(Path(per_class_path))
            new_policy = apply_catf_v2_causal_probe_if_enabled(state, controller, new_policy, epoch_num=epoch_num)
            new_policy = apply_catf_v2_riskguard_if_enabled(state, controller, new_policy, epoch_num=epoch_num)
            adaptive_terminal = False
            adaptive_event: dict[str, Any] | None = None
            if bool(getattr(state.args, "adaptive_burnin", False)):
                if state.adaptive_burnin_controller is None:
                    state.adaptive_burnin_controller = AdaptiveBurninController(adaptive_burnin_config_from_args(state.args))
                adaptive_event = state.adaptive_burnin_controller.evaluate(
                    epoch=epoch_num,
                    metrics=policy_metrics,
                    reference_metrics=reference_metrics,
                    clean_reference_metrics=state.reference_metrics,
                    metric_history=recent_metric_history(state, int(state.args.metric_stability_window)),
                    per_class_diagnosis=per_class,
                    old_policy=old_policy,
                    proposed_policy=new_policy,
                    close_mosaic_start_epoch=close_mosaic_start_epoch(state.args),
                )
                new_policy = deepcopy(adaptive_event.get("policy", new_policy))
                adaptive_terminal = adaptive_event.get("action") in {"burnin_observe", "no_op_fallback"}
                if hasattr(controller, "policy"):
                    controller.policy.matrix = deepcopy(new_policy)
                annotate_policy_history_with_adaptive_burnin(
                    controller.history,
                    adaptive_event,
                    new_policy,
                    old_policy=old_policy,
                )
                state.adaptive_burnin_events = deepcopy(state.adaptive_burnin_controller.events)
                if (
                    adaptive_event.get("candidate_branch_started")
                    and bool(getattr(state.args, "catf_rollback_mode", False))
                    and state.rollback_controller is not None
                ):
                    rb_event = state.rollback_controller.start_candidate(trainer=trainer, output_dir=state.output_dir, epoch=epoch_num)
                    annotate_policy_history_with_rollback(controller.history, rb_event, new_policy)
                    state.rollback_events = deepcopy(state.rollback_controller.events)
            safe_event: dict[str, Any] | None = None
            if bool(getattr(state.args, "catf_safe_mode", False)) and not adaptive_terminal:
                if state.safe_controller is None:
                    state.safe_controller = CATFSafeController()
                latest_record = controller.history[-1] if controller.history else {}
                safe_event = state.safe_controller.evaluate(
                    epoch=epoch_num,
                    policy=new_policy,
                    metrics=policy_metrics,
                    reference_metrics=reference_metrics,
                    per_class_diagnosis=per_class,
                    active_classes=latest_record.get("active_classes", []),
                    proposed_action=latest_record.get("action"),
                )
                new_policy = deepcopy(safe_event.get("policy", new_policy))
                if hasattr(controller, "policy"):
                    controller.policy.matrix = deepcopy(new_policy)
                if controller.history:
                    event_for_history = deepcopy(safe_event)
                    event_for_history.pop("policy", None)
                    controller.history[-1]["safe_controller_event"] = event_for_history
                    controller.history[-1]["catf_safe_mode"] = True
                    controller.history[-1]["safe_fallback_active"] = bool(event_for_history.get("fallback_active_after"))
                    controller.history[-1]["safe_accept_allowed"] = bool(event_for_history.get("safe_accept_allowed"))
                    if event_for_history.get("triggered"):
                        controller.history[-1]["action"] = event_for_history.get("action", controller.history[-1].get("action"))
                        controller.history[-1]["accepted_policy"] = deepcopy(new_policy)
                        controller.history[-1]["new_policy"] = deepcopy(new_policy)
                        controller.history[-1]["guard_triggered"] = list(
                            dict.fromkeys(
                                list(controller.history[-1].get("guard_triggered", []) or [])
                                + list(event_for_history.get("reasons", []) or [])
                            )
                        )
                        controller.history[-1]["frozen"] = event_for_history.get("action") == "no_op_freeze"
                state.safe_events = deepcopy(state.safe_controller.events)
            gated_event: dict[str, Any] | None = None
            if bool(getattr(state.args, "catf_gated_mode", False)) and not adaptive_terminal:
                if state.gated_controller is None:
                    state.gated_controller = CATFGatedController()
                latest_record = controller.history[-1] if controller.history else {}
                gated_event = state.gated_controller.evaluate(
                    epoch=epoch_num,
                    policy=new_policy,
                    metrics=policy_metrics,
                    reference_metrics=reference_metrics,
                    per_class_diagnosis=per_class,
                    active_classes=latest_record.get("active_classes", []),
                    proposed_action=latest_record.get("action"),
                )
                new_policy = deepcopy(gated_event.get("policy", new_policy))
                if hasattr(controller, "policy"):
                    controller.policy.matrix = deepcopy(new_policy)
                annotate_policy_history_with_gate(controller.history, gated_event, new_policy)
                state.gated_events = deepcopy(state.gated_controller.events)
            if (
                bool(getattr(state.args, "adaptive_burnin", False))
                and bool(getattr(state.args, "catf_rollback_mode", False))
                and state.rollback_controller is not None
                and not adaptive_terminal
            ):
                rb_probe_event = state.rollback_controller.maybe_evaluate_probe(
                    trainer=trainer,
                    epoch=epoch_num,
                    policy=new_policy,
                    metrics=policy_metrics,
                    reference_metrics=reference_metrics,
                    per_class_diagnosis=per_class,
                    active_classes=(controller.history[-1] if controller.history else {}).get("active_classes", []),
                    output_dir=state.output_dir,
                )
                if rb_probe_event is not None:
                    new_policy = deepcopy(rb_probe_event.get("policy", new_policy))
                    if hasattr(controller, "policy"):
                        controller.policy.matrix = deepcopy(new_policy)
                    annotate_policy_history_with_rollback(controller.history, rb_probe_event, new_policy)
                    state.rollback_events = deepcopy(state.rollback_controller.events)
            state.policy_state.clear()
            state.policy_state.update(deepcopy(new_policy))
            state.context.augmentor.set_policy(new_policy)
            state.feedback_epochs.append(epoch_num)
            state.history = deepcopy(controller.history)
            if state.history:
                state.history[-1]["diagnosis_global"] = diagnosis.get("global", {})
                state.history[-1]["diagnosis_vector"] = diagnosis.get("diagnosis_vector", {})
                state.history[-1]["policy_selection_source"] = policy_selection_source(state.args)
                state.history[-1]["policy_selection_data"] = str(policy_selection_data_yaml(state.args).resolve())
                state.history[-1]["final_val_used_for_policy_selection"] = False if bool(getattr(state.args, "paper_probe_mode", False)) else None
                state.history[-1]["trainer_final_val_metrics_observed"] = metrics
                state.history[-1]["old_policy_before_callback"] = old_policy
                state.history[-1]["new_policy"] = deepcopy(state.policy_state)
                state.history[-1]["copy_paste_status"] = "pending_object_bank_design"
                state.history[-1]["trainer_identity"] = {
                    "trainer_id": id(trainer),
                    "optimizer_id": id(getattr(trainer, "optimizer", None)) if getattr(trainer, "optimizer", None) is not None else None,
                    "scheduler_id": id(getattr(trainer, "scheduler", None)) if getattr(trainer, "scheduler", None) is not None else None,
                    "ema_id": id(getattr(trainer, "ema", None)) if getattr(trainer, "ema", None) is not None else None,
                }
            write_json(state.output_dir / "configs" / f"active_policy_epoch_{epoch_num:03d}.json", new_policy)
            return
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
    if not bool(getattr(args, "industrial_aug_enabled", True)) and not bool(getattr(args, "diagnosis_only", False)):
        return False
    if epoch_num >= int(args.epochs):
        return False
    if bool(getattr(args, "adaptive_burnin", False)) and is_catf_v2(args):
        if epoch_num < int(getattr(args, "min_burnin_epoch", 5)):
            return False
        interval = max(1, int(getattr(args, "burnin_check_interval", getattr(args, "feedback_interval", 5))))
        return epoch_num % interval == 0
    if epoch_num < int(args.feedback_start_epoch):
        return False
    interval = max(1, int(args.feedback_interval))
    return epoch_num % interval == 0


def record_diagnosis_only_feedback(
    state: InLoopFeedbackState,
    trainer: Any,
    epoch_num: int,
    metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    diagnosis: dict[str, Any],
    old_policy: dict[str, Any],
    *,
    action: str = "diagnosis_only",
    catf_noop: bool = False,
) -> None:
    """Record in-loop diagnosis without mutating policy_state or augmentation."""

    record: dict[str, Any] = {
        "epoch": epoch_num,
        "action": action,
        "catf_noop": bool(catf_noop),
        "metrics": metrics,
        "reference_metrics": reference_metrics,
        "delta_metrics": metric_delta(metrics, reference_metrics),
        "policy_selection_source": policy_selection_source(state.args),
        "policy_selection_data": str(policy_selection_data_yaml(state.args).resolve()),
        "final_val_used_for_policy_selection": False if bool(getattr(state.args, "paper_probe_mode", False)) else None,
        "diagnosis_global": diagnosis.get("global", {}),
        "diagnosis_vector": diagnosis.get("diagnosis_vector", {}),
        "old_policy_before_callback": deepcopy(old_policy),
        "new_policy": deepcopy(old_policy),
        "policy_update_applied": False,
        "industrial_aug_applied": False,
        "roi_aug_applied": False,
        "random_draws_allowed": False if catf_noop else None,
        "sample_routing_changes_sample_order": False if catf_noop else None,
        "guard_triggered": [],
        "rollback_reason": None,
        "frozen": False,
        "adjustments": [],
        "class_actions": [],
        "active_classes": [],
        "frozen_classes": [],
        "high_fp_guarded_classes": [],
        "copy_paste_status": "pending_object_bank_design",
        "trainer_identity": {
            "trainer_id": id(trainer),
            "optimizer_id": id(getattr(trainer, "optimizer", None)) if getattr(trainer, "optimizer", None) is not None else None,
            "scheduler_id": id(getattr(trainer, "scheduler", None)) if getattr(trainer, "scheduler", None) is not None else None,
            "ema_id": id(getattr(trainer, "ema", None)) if getattr(trainer, "ema", None) is not None else None,
        },
    }
    if is_catf_v2(state.args):
        class_names = load_class_names_from_data_yaml(state.args.data)
        per_class = build_per_class_diagnosis(
            diagnosis,
            class_names=class_names,
            train_instances=count_train_instances(state.args.data),
            epoch=epoch_num,
        )
        attribution = attribute_class_issues(per_class)
        sample_weight_map = build_sample_weight_map(per_class, attribution)
        catf_dir = state.output_dir / "reports" / "catf_v2"
        for base in (state.output_dir / "reports", catf_dir):
            write_json(base / f"per_class_diagnosis_epoch_{epoch_num}.json", per_class)
            write_json(base / f"issue_attribution_epoch_{epoch_num}.json", attribution)
            write_json(base / f"policy_matrix_epoch_{epoch_num}_before.json", old_policy)
            write_json(base / f"policy_matrix_epoch_{epoch_num}_after.json", old_policy)
            write_json(base / f"sample_weight_map_epoch_{epoch_num}.json", sample_weight_map)
        if bool(getattr(state.args, "threshold_calibration_report", False)):
            threshold_payload = ThresholdCalibrationAnalyzer().analyze(per_class, attribution)
            ThresholdCalibrationAnalyzer().write(state.output_dir / "reports" / "threshold_calibration.json", threshold_payload)
            ThresholdCalibrationAnalyzer().write(catf_dir / "threshold_calibration.json", threshold_payload)
        record["diagnosis_summary"] = {
            "per_class": per_class.get("summary", {}),
            "issue_attribution": attribution.get("summary", {}),
        }
        record["per_class_diagnosis_path"] = str(state.output_dir / "reports" / f"per_class_diagnosis_epoch_{epoch_num}.json")
        record["issue_attribution_path"] = str(state.output_dir / "reports" / f"issue_attribution_epoch_{epoch_num}.json")
        record["sample_weight_map_path"] = str(state.output_dir / "reports" / f"sample_weight_map_epoch_{epoch_num}.json")
    state.feedback_epochs.append(epoch_num)
    state.history.append(record)
    write_json(state.output_dir / "configs" / f"active_policy_epoch_{epoch_num:03d}.json", old_policy)
    if is_catf_v2(state.args):
        write_json(state.output_dir / "reports" / "policy_history.json", {"history": state.history, "latest_policy": old_policy})
        write_json(state.output_dir / "reports" / "class_policy_history.json", {"history": []})
    else:
        write_policy_history(state.output_dir / "reports", state.history, old_policy)


def policy_selection_source(args: argparse.Namespace) -> str:
    return "probe_split" if bool(getattr(args, "paper_probe_mode", False)) else "final_val_diagnostics"


def policy_selection_data_yaml(args: argparse.Namespace) -> Path:
    if bool(getattr(args, "paper_probe_mode", False)):
        probe_data = getattr(args, "probe_data", None)
        if not probe_data:
            raise ValueError("paper probe mode requires --probe-data")
        return Path(probe_data).resolve()
    return Path(args.data).resolve()


def policy_selection_metrics(
    state: InLoopFeedbackState,
    diagnosis: dict[str, Any],
    trainer_metrics: dict[str, Any],
) -> dict[str, Any]:
    if not bool(getattr(state.args, "paper_probe_mode", False)):
        out = deepcopy(trainer_metrics)
        out["source"] = "final_val_metrics"
        return out
    global_diag = diagnosis.get("global", {}) or {}
    tp = float(global_diag.get("tp", 0) or 0)
    fp = float(global_diag.get("fp", 0) or 0)
    fn = float(global_diag.get("fn", 0) or 0)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    # Probe mode intentionally avoids final validation AP. These proxies are
    # only guard signals for policy selection; final AP remains measured on val.
    map50_proxy = 0.5 * (precision + recall)
    weak = float(global_diag.get("localization_weak", 0) or 0)
    map95_proxy = max(0.0, map50_proxy - min(0.25, weak / max(1.0, tp + fn + fp) * 0.5))
    return {
        "precision": precision,
        "recall": recall,
        "map50": map50_proxy,
        "map50_95": map95_proxy,
        "images": global_diag.get("images"),
        "instances": global_diag.get("gt", global_diag.get("instances")),
        "source": "probe_split_diagnosis_proxy",
        "final_val_used_for_policy_selection": False,
    }


def policy_reference_metrics_for_epoch(
    state: InLoopFeedbackState,
    epoch_num: int,
    policy_metrics: dict[str, Any],
) -> dict[str, Any]:
    if bool(getattr(state.args, "paper_probe_mode", False)):
        reference = {key: policy_metrics.get(key) for key in METRIC_KEYS}
        reference["source"] = "probe_split_self_reference"
        reference["final_val_used_for_policy_selection"] = False
        return reference
    return reference_metrics_for_epoch(state, epoch_num)


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
    diagnosis_data = policy_selection_data_yaml(state.args)
    if bool(getattr(state.args, "paper_probe_mode", False)):
        validate_paper_probe_configuration(state.args)
    val_images, val_labels = resolve_val_image_label_dirs(diagnosis_data)
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
    diagnosis["policy_selection_source"] = policy_selection_source(state.args)
    diagnosis["policy_selection_data_yaml"] = str(diagnosis_data.resolve())
    diagnosis["policy_selection_images_dir"] = str(val_images.resolve())
    diagnosis["final_val_used_for_policy_selection"] = False if bool(getattr(state.args, "paper_probe_mode", False)) else None
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
    roi_aug_stats = (
        state.context.augmentor.roi_stats.to_dict()
        if hasattr(state.context.augmentor, "roi_stats") and state.context.augmentor.roi_stats is not None
        else {}
    )
    train_image_count = state.context.train_image_count
    if train_image_count is None:
        train_image_count = count_split_images(Path(args.data).resolve(), "train")
    expected_train_images = count_split_images(Path(args.data).resolve(), "train")
    stats.update(
        {
            "run_id": args.run_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "catf_noop": bool(getattr(args, "catf_noop", False)),
            "catf_safe_mode": bool(getattr(args, "catf_safe_mode", False)),
            "catf_gated_mode": bool(getattr(args, "catf_gated_mode", False)),
            "catf_rollback_mode": bool(getattr(args, "catf_rollback_mode", False)),
            "catf_riskguard": bool(getattr(args, "catf_riskguard", False)),
            "catf_causal_probe": bool(getattr(args, "catf_causal_probe", False)),
            "catf_causal_probe_mode": str(getattr(args, "catf_causal_probe_mode", "development")),
            "image_only_mainline": bool(getattr(args, "image_only_mainline", False)),
            "preserve_original_enabled": bool(getattr(args, "preserve_original_enabled", False)),
            "weak_image_aug_enabled": bool(getattr(args, "weak_image_aug_enabled", False)),
            "weak_only_for_moderate_risk": bool(getattr(args, "weak_only_for_moderate_risk", False)),
            "attenuation_ratio": float(getattr(args, "attenuation_ratio", 0.25) or 0.25),
            "disable_sampler_only": bool(getattr(args, "disable_sampler_only", False)),
            "paper_probe_mode": bool(getattr(args, "paper_probe_mode", False)),
            "probe_data": str(Path(args.probe_data).resolve()) if getattr(args, "probe_data", None) else None,
            "train_core_data": str(Path(args.train_core_data).resolve()) if getattr(args, "train_core_data", None) else str(Path(args.data).resolve()),
            "probe_source": str(getattr(args, "probe_source", "train_probe_split")),
            "forbid_final_val_policy_selection": bool(getattr(args, "forbid_final_val_policy_selection", False)),
            "policy_selection_source": policy_selection_source(args),
            "policy_selection_data": str(policy_selection_data_yaml(args).resolve()),
            "final_val_used_for_policy_selection": False if bool(getattr(args, "paper_probe_mode", False)) else None,
            "adaptive_burnin": bool(getattr(args, "adaptive_burnin", False)),
            "adaptive_start_epoch": state.adaptive_burnin_controller.start_epoch if state.adaptive_burnin_controller else None,
            "noop_transform_calls": int(getattr(state.context, "noop_transform_calls", 0) or 0),
            "router_random_draw_count": int(getattr(state.context.augmentor, "random_draw_count", 0) or 0),
            "weak_image_aug_interval_counts": dict(getattr(state.context.augmentor, "weak_image_aug_counts", {}) or {}),
            "sample_weight_map_generated": bool(getattr(state.context, "sample_weight_map_generated", False)),
            "weighted_train_core_images_count": int(getattr(state.context, "weighted_train_core_images_count", 0) or 0),
            "weighted_sampler_enabled": bool(getattr(state.context, "weighted_sampler_enabled", False)),
            "weighted_index_list_enabled": bool(getattr(state.context, "weighted_index_list_enabled", False)),
            "sampler_only_effective": bool(getattr(state.context, "sampler_only_effective", False)),
            "sampler_only_status": str(getattr(state.context, "sampler_only_status", "not_requested")),
            "sampled_distribution_changed": bool(getattr(state.context, "sampled_distribution_changed", False)),
            "sampler_only_artifact_paths": dict(getattr(state.context, "sampler_only_artifact_paths", {}) or {}),
            "sample_router_built": bool(is_catf_v2(args)),
            "online_augmentation": bool(args.industrial_aug_enabled),
            "industrial_online_augmentation": bool(args.industrial_aug_enabled),
            "inloop_feedback": bool(args.feedback_enabled),
            "diagnosis_only": bool(getattr(args, "diagnosis_only", False)),
            "train_image_count": train_image_count,
            "expected_original_train_images": expected_train_images,
            "train_image_count_matches_original": train_image_count == expected_train_images,
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
        "diagnosis_only": bool(getattr(args, "diagnosis_only", False)),
        "policy_update_applied_count": sum(1 for record in state.history if record.get("policy_update_applied", True)),
        "feedback_controller": "CATF-v2" if is_catf_v2(args) else "CATF",
        "catf_version": str(getattr(args, "catf_version", "v1")),
        "catf_noop": bool(getattr(args, "catf_noop", False)),
        "catf_safe_mode": bool(getattr(args, "catf_safe_mode", False)),
        "catf_gated_mode": bool(getattr(args, "catf_gated_mode", False)),
        "catf_rollback_mode": bool(getattr(args, "catf_rollback_mode", False)),
        "catf_riskguard": bool(getattr(args, "catf_riskguard", False)),
        "catf_causal_probe": bool(getattr(args, "catf_causal_probe", False)),
        "catf_causal_probe_mode": str(getattr(args, "catf_causal_probe_mode", "development")),
        "image_only_mainline": bool(getattr(args, "image_only_mainline", False)),
        "preserve_original_enabled": bool(getattr(args, "preserve_original_enabled", False)),
        "weak_image_aug_enabled": bool(getattr(args, "weak_image_aug_enabled", False)),
        "weak_only_for_moderate_risk": bool(getattr(args, "weak_only_for_moderate_risk", False)),
        "attenuation_ratio": float(getattr(args, "attenuation_ratio", 0.25) or 0.25),
        "disable_sampler_only": bool(getattr(args, "disable_sampler_only", False)),
        "paper_probe_mode": bool(getattr(args, "paper_probe_mode", False)),
        "probe_data": str(Path(args.probe_data).resolve()) if getattr(args, "probe_data", None) else None,
        "train_core_data": str(Path(args.train_core_data).resolve()) if getattr(args, "train_core_data", None) else str(Path(args.data).resolve()),
        "probe_source": str(getattr(args, "probe_source", "train_probe_split")),
        "forbid_final_val_policy_selection": bool(getattr(args, "forbid_final_val_policy_selection", False)),
        "policy_selection_source": policy_selection_source(args),
        "policy_selection_data": str(policy_selection_data_yaml(args).resolve()),
        "final_val_used_for_policy_selection": False if bool(getattr(args, "paper_probe_mode", False)) else None,
        "use_offline_probe_decisions": bool(getattr(args, "use_offline_probe_decisions", False)),
        "offline_probe_decisions_file": str(resolve_offline_probe_decision_path(args)) if bool(getattr(args, "use_offline_probe_decisions", False)) else None,
        "adaptive_burnin": bool(getattr(args, "adaptive_burnin", False)),
        "adaptive_start_epoch": state.adaptive_burnin_controller.start_epoch if state.adaptive_burnin_controller else None,
        "adaptive_candidate_started": bool(state.adaptive_burnin_controller and state.adaptive_burnin_controller.triggered),
        "adaptive_noop_fallback": bool(state.adaptive_burnin_controller and state.adaptive_burnin_controller.fallback_active),
        "adaptive_noop_reason": state.adaptive_burnin_controller.fallback_reason if state.adaptive_burnin_controller else None,
        "adaptive_burnin_events": str((output_dir / "reports" / "adaptive_burnin_events.json").resolve()),
        "rollback_triggered": any(event.get("action") == "rollback" for event in state.rollback_events),
        "rollback_controller_events": str((output_dir / "reports" / "rollback_controller_events.json").resolve()),
        "class_aware_feedback": bool(getattr(args, "class_aware_feedback", False)),
        "roi_aware_aug": bool(getattr(args, "roi_aware_aug", False)),
        "sample_aware_routing": bool(getattr(args, "sample_aware_routing", False)),
        "threshold_calibration_report": bool(getattr(args, "threshold_calibration_report", False)),
        "reference_curve_loaded": bool(state.reference_curve),
        "yolo_default_augmentation_enabled": True,
        "industrial_aug_enabled": bool(args.industrial_aug_enabled),
        "industrial_aug_dynamic": bool(args.industrial_aug_enabled and int(stats.get("samples_augmented", 0) or 0) > 0),
        "weak_image_aug_interval_counts": dict(getattr(state.context.augmentor, "weak_image_aug_counts", {}) or {}),
        "sampler_only_enabled": bool(getattr(args, "sampler_only_enabled", False)),
        "sample_weight_map_generated": bool(getattr(state.context, "sample_weight_map_generated", False)),
        "weighted_train_core_images_count": int(getattr(state.context, "weighted_train_core_images_count", 0) or 0),
        "weighted_sampler_enabled": bool(getattr(state.context, "weighted_sampler_enabled", False)),
        "weighted_index_list_enabled": bool(getattr(state.context, "weighted_index_list_enabled", False)),
        "sampler_only_effective": bool(getattr(state.context, "sampler_only_effective", False)),
        "sampler_only_status": str(getattr(state.context, "sampler_only_status", "not_requested")),
        "sampled_distribution_changed": bool(getattr(state.context, "sampled_distribution_changed", False)),
        "fixed_augmented_dataset_generated": stats["fixed_augmented_dataset_generated"],
        "train_image_count": train_image_count,
        "bbox_class_valid": stats["invalid_bbox_count"] == 0 and stats["class_id_oob_count"] == 0,
        "policy_history": str((output_dir / "reports" / "policy_history.json").resolve()),
        "online_aug_stats": str((output_dir / "reports" / "online_aug_stats.json").resolve()),
        "roi_aug_stats": str((output_dir / "reports" / "roi_aug_stats.json").resolve()) if roi_aug_stats else None,
        "report": str(primary_report_path(output_dir, args).resolve()),
        "constraint_failed": constraint_scoring["constraint_failed"] if args.feedback_enabled else None,
        "safe_fallback_triggered": any(event.get("action") == "no_op_freeze" for event in state.safe_events),
        "safe_controller_events": str((output_dir / "reports" / "safe_controller_events.json").resolve()),
        "gated_fallback_triggered": any(event.get("action") == "no_op_freeze" for event in state.gated_events),
        "gated_controller_events": str((output_dir / "reports" / "gated_controller_events.json").resolve()),
        "riskguard_enabled": bool(getattr(args, "catf_riskguard", False)),
        "riskguard_blocked_op_count": len(state.riskguard_events) + len(getattr(state.context.augmentor, "riskguard_events", []) or []),
        "riskguard_events": str((output_dir / "reports" / "riskguard_events.json").resolve()),
        "causal_probe_enabled": bool(getattr(args, "catf_causal_probe", False)),
        "causal_probe_event_count": len(state.causal_probe_events),
        "causal_probe_events": str((output_dir / "reports" / "causal_probe_events.json").resolve()),
        "causal_probe_decisions_used": str((output_dir / "reports" / "causal_probe_decisions_used.json").resolve())
        if bool(getattr(args, "use_offline_probe_decisions", False)) or bool(getattr(args, "paper_probe_mode", False))
        else None,
    }
    catf_v2_summary = state.feedback_controller.summary() if is_catf_v2(args) and hasattr(state.feedback_controller, "summary") else {}
    paper_probe_audit = validate_paper_probe_configuration(args) if bool(getattr(args, "paper_probe_mode", False)) else {}
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
        "diagnosis_only": bool(getattr(args, "diagnosis_only", False)),
        "industrial_aug_enabled": bool(args.industrial_aug_enabled),
        "feedback_controller": "CATF-v2" if is_catf_v2(args) else "CATF",
        "catf_version": str(getattr(args, "catf_version", "v1")),
        "catf_noop": bool(getattr(args, "catf_noop", False)),
        "catf_safe_mode": bool(getattr(args, "catf_safe_mode", False)),
        "catf_gated_mode": bool(getattr(args, "catf_gated_mode", False)),
        "catf_rollback_mode": bool(getattr(args, "catf_rollback_mode", False)),
        "catf_riskguard": bool(getattr(args, "catf_riskguard", False)),
        "catf_causal_probe": bool(getattr(args, "catf_causal_probe", False)),
        "catf_causal_probe_mode": str(getattr(args, "catf_causal_probe_mode", "development")),
        "use_offline_probe_decisions": bool(getattr(args, "use_offline_probe_decisions", False)),
        "offline_probe_decisions_file": str(resolve_offline_probe_decision_path(args)) if bool(getattr(args, "use_offline_probe_decisions", False)) else None,
        "causal_probe_events": state.causal_probe_events,
        "paper_probe_leakage_audit": paper_probe_audit,
        "adaptive_burnin": bool(getattr(args, "adaptive_burnin", False)),
        "adaptive_start_epoch": state.adaptive_burnin_controller.start_epoch if state.adaptive_burnin_controller else None,
        "adaptive_burnin_config": build_adaptive_burnin_config_payload(args),
        "class_aware_feedback": bool(getattr(args, "class_aware_feedback", False)),
        "roi_aware_aug": bool(getattr(args, "roi_aware_aug", False)),
        "sample_aware_routing": bool(getattr(args, "sample_aware_routing", False)),
        "threshold_calibration_report": bool(getattr(args, "threshold_calibration_report", False)),
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
        "adaptive_burnin_events": state.adaptive_burnin_events,
        "rollback_controller_events": state.rollback_events,
        "safe_controller_events": state.safe_events,
        "gated_controller_events": state.gated_events,
        "riskguard_events": state.riskguard_events + list(getattr(state.context.augmentor, "riskguard_events", []) or []),
        "causal_probe_events": state.causal_probe_events,
        "latest_policy_state": state.policy_state,
        "epoch_records": state.epoch_records,
        "online_aug_stats": stats,
        "roi_aug_stats": roi_aug_stats,
        "catf_v2": catf_v2_summary,
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
    if payload.get("roi_aug_stats"):
        write_json(output_dir / "reports" / "roi_aug_stats.json", payload["roi_aug_stats"])
    write_json(output_dir / "reports" / "epoch_records.json", payload["epoch_records"])
    write_json(output_dir / "reports" / "constraint_scoring.json", payload["constraint_scoring"])
    if payload.get("adaptive_burnin"):
        write_json(output_dir / "reports" / "adaptive_burnin_events.json", {"events": payload.get("adaptive_burnin_events", [])})
    if payload.get("catf_rollback_mode"):
        write_json(output_dir / "reports" / "rollback_controller_events.json", {"events": payload.get("rollback_controller_events", [])})
    if payload.get("catf_safe_mode"):
        write_json(output_dir / "reports" / "safe_controller_events.json", {"events": payload.get("safe_controller_events", [])})
    if payload.get("catf_gated_mode"):
        write_json(output_dir / "reports" / "gated_controller_events.json", {"events": payload.get("gated_controller_events", [])})
    if payload.get("catf_riskguard"):
        write_json(output_dir / "reports" / "riskguard_events.json", {"events": payload.get("riskguard_events", [])})
    if payload.get("catf_causal_probe"):
        write_json(output_dir / "reports" / "causal_probe_events.json", {"events": payload.get("causal_probe_events", [])})
    if payload.get("paper_probe_mode"):
        write_json(output_dir / "reports" / "paper_probe_leakage_audit.json", payload.get("paper_probe_leakage_audit", {}))
    if payload.get("catf_version") == "v2":
        write_catf_v2_history(output_dir / "reports", payload)
    else:
        write_policy_history(output_dir / "reports", payload["policy_history"], payload["latest_policy_state"])
    write_markdown(output_dir / "reports" / "final_report.md", build_final_report(payload))
    write_markdown(output_dir / "reports" / "inloop_feedback_smoke_report.md", build_smoke_report(payload))
    if payload["feedback_enabled"] and int(payload["epochs"]) <= 10:
        if payload.get("catf_version") == "v2":
            write_markdown(output_dir / "reports" / "catf_v2_smoke_report.md", build_catf_v2_smoke_report(payload))
            if payload.get("adaptive_burnin"):
                write_markdown(output_dir / "reports" / "adaptive_burnin_smoke_report.md", build_adaptive_burnin_smoke_report(payload))
        else:
            write_markdown(output_dir / "reports" / "catf_smoke_report.md", build_catf_smoke_report(payload))
    if payload.get("diagnosis_only") and int(payload["epochs"]) <= 10:
        write_markdown(output_dir / "reports" / "diagnosis_only_smoke_report.md", build_diagnosis_only_smoke_report(payload))
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


def write_catf_v2_history(history_dir: Path, payload: dict[str, Any]) -> None:
    history = payload.get("policy_history", [])
    latest_policy = payload.get("latest_policy_state", {})
    write_json(history_dir / "policy_history.json", {"history": history, "latest_policy": latest_policy})
    class_rows = []
    for record in history:
        for action in record.get("class_actions", []) or []:
            class_rows.append(
                {
                    "epoch": int(record.get("epoch", 0) or 0),
                    "class_id": int(action.get("class_id", -1)),
                    "action": action.get("action"),
                    "adjustment_count": len(action.get("adjustments", []) or []),
                    "before_status": (action.get("before") or {}).get("status"),
                    "after_status": (action.get("after") or {}).get("status"),
                    "dominant_issue": (action.get("after") or {}).get("dominant_issue"),
                }
            )
    write_json(history_dir / "class_policy_history.json", {"history": class_rows})
    lines = [
        "# CATF-v2 Class-Aware Policy History",
        "",
        "| epoch | action | active_classes | frozen_classes | high_fp_guarded | adjustments |",
        "|---:|---|---|---|---|---:|",
    ]
    for record in history:
        lines.append(
            f"| {record.get('epoch')} | {record.get('action')} | {record.get('active_classes', [])} | "
            f"{record.get('frozen_classes', [])} | {record.get('high_fp_guarded_classes', [])} | {len(record.get('adjustments', []) or [])} |"
        )
    write_markdown(history_dir / "policy_history.md", "\n".join(lines))
    with (history_dir / "policy_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "class_id",
                "action",
                "op",
                "field",
                "before",
                "after",
                "reason",
                "guard_triggered",
            ],
        )
        writer.writeheader()
        for record in history:
            for adjustment in record.get("adjustments", []) or [{}]:
                writer.writerow(
                    {
                        "epoch": record.get("epoch"),
                        "class_id": adjustment.get("class_id"),
                        "action": record.get("action"),
                        "op": adjustment.get("op"),
                        "field": adjustment.get("field"),
                        "before": adjustment.get("before"),
                        "after": adjustment.get("after"),
                        "reason": adjustment.get("reason"),
                        "guard_triggered": ",".join(record.get("guard_triggered", []) or []),
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


def build_catf_v2_smoke_report(payload: dict[str, Any]) -> str:
    history = payload.get("policy_history", [])
    stats = payload["online_aug_stats"]
    roi_stats = payload.get("roi_aug_stats") or {}
    catf = payload.get("catf_v2") or {}
    safe_events = payload.get("safe_controller_events") or []
    gated_events = payload.get("gated_controller_events") or []
    last = history[-1] if history else {}
    lines = [
        "# CATF-v2 Class-Aware Smoke Report",
        "",
        "## Answers",
        "",
        f"- CATF-v2 implemented: `{str(payload.get('catf_version') == 'v2').lower()}`",
        f"- Per-class diagnosis generated: `{str(bool(last.get('per_class_diagnosis_path'))).lower()}`",
        f"- Issue attribution generated: `{str(bool(last.get('issue_attribution_path'))).lower()}`",
        f"- Policy matrix active: `{str(bool(payload.get('latest_policy_state', {}).get('classes'))).lower()}`",
        f"- Sample-aware routing enabled: `{str(payload.get('sample_aware_routing')).lower()}`",
        f"- ROI-aware augmentation enabled: `{str(payload.get('roi_aware_aug')).lower()}`",
        f"- Threshold calibration report enabled: `{str(payload.get('threshold_calibration_report', True)).lower()}`",
        f"- CATF-v2 safe mode enabled: `{str(payload.get('catf_safe_mode', False)).lower()}`",
        f"- CATF-v2 gated mode enabled: `{str(payload.get('catf_gated_mode', False)).lower()}`",
        f"- CATF-v2 rollback mode enabled: `{str(payload.get('catf_rollback_mode', False)).lower()}`",
        f"- Adaptive burn-in enabled: `{str(payload.get('adaptive_burnin', False)).lower()}`",
        f"- Adaptive start epoch: `{payload.get('adaptive_start_epoch')}`",
        f"- Baseline protection triggered: `{str(any('baseline_protection' in ','.join(event.get('reasons', [])) for event in safe_events)).lower()}`",
        f"- Early abstention triggered: `{str(any('early_abstention' in ','.join(event.get('reasons', [])) for event in safe_events)).lower()}`",
        f"- No-op freeze entered: `{str(any(event.get('action') == 'no_op_freeze' for event in safe_events)).lower()}`",
        f"- Gated no-op freeze entered: `{str(any(event.get('action') == 'no_op_freeze' for event in gated_events)).lower()}`",
        f"- Reference curve loaded: `{str(bool(payload.get('reference_curve_epochs'))).lower()}`",
        f"- Epoch 5 update happened: `{str(5 in payload['summary']['feedback_epochs']).lower()}`",
        f"- BBox/class legal: `{str(payload['summary']['bbox_class_valid']).lower()}`",
        f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
        "",
        "## Class-Aware State",
        "",
        f"- Active classes: `{catf.get('active_classes', [])}`",
        f"- Frozen classes: `{catf.get('frozen_classes', [])}`",
        f"- High-FP guarded classes: `{catf.get('high_fp_guarded_classes', [])}`",
        f"- Low-contrast classes: `{last.get('low_contrast_classes', [])}`",
        f"- Texture classes: `{last.get('texture_classes', [])}`",
        f"- Low-support only classes: `{last.get('low_support_classes', [])}`",
        f"- Stable classes avoided: `{str(bool(catf.get('frozen_classes'))).lower()}`",
        "",
        "## ROI Augmentation",
        "",
        f"- ROI-aware applied count: `{roi_stats.get('roi_aug_applied', 0)}`",
        f"- ROI skipped small ROI: `{roi_stats.get('roi_aug_skipped_small_roi', 0)}`",
        f"- ROI skipped conflict: `{roi_stats.get('roi_aug_skipped_conflict', 0)}`",
        f"- Affected classes: `{roi_stats.get('affected_classes', {})}`",
        "",
        "## Required Artifacts",
        "",
        f"- Per-class diagnosis: `{last.get('per_class_diagnosis_path')}`",
        f"- Issue attribution: `{last.get('issue_attribution_path')}`",
        f"- Sample weight map: `{last.get('sample_weight_map_path')}`",
        f"- Policy history: `{payload['summary']['policy_history']}`",
        f"- Online aug stats: `{payload['summary']['online_aug_stats']}`",
        f"- ROI aug stats: `{payload['summary'].get('roi_aug_stats')}`",
        f"- Safe controller events: `{payload['summary'].get('safe_controller_events')}`",
        f"- Gated controller events: `{payload['summary'].get('gated_controller_events')}`",
        f"- Adaptive burn-in events: `{payload['summary'].get('adaptive_burnin_events')}`",
        f"- Rollback controller events: `{payload['summary'].get('rollback_controller_events')}`",
        "",
        "## Next Step",
        "",
        "- This smoke only validates the CATF-v2 control path. It is not a 50 epoch result.",
    ]
    return "\n".join(lines) + "\n"


def build_adaptive_burnin_smoke_report(payload: dict[str, Any]) -> str:
    events = payload.get("adaptive_burnin_events") or []
    stats = payload.get("online_aug_stats") or {}
    roi_stats = payload.get("roi_aug_stats") or {}
    epoch5_event = next((event for event in events if int(event.get("epoch", -1)) == 5), None)
    start_event = next((event for event in events if event.get("candidate_branch_started")), None)
    noop_event = next((event for event in events if event.get("action") == "no_op_fallback"), None)
    strict_noop = (
        not bool(start_event)
        and int(stats.get("samples_augmented", 0) or 0) == 0
        and int(roi_stats.get("roi_aug_applied", 0) or 0) == 0
        and int(stats.get("router_random_draw_count", 0) or 0) == 0
    )
    reasons = []
    if epoch5_event:
        reasons = list(epoch5_event.get("reasons") or [])
    lines = [
        "# Adaptive Burn-in CATF-v2 Smoke Report",
        "",
        "## Answers",
        "",
        f"- Adaptive burn-in enabled: `{str(payload.get('adaptive_burnin', False)).lower()}`",
        f"- Epoch 5 start_condition checked: `{str(bool(epoch5_event and epoch5_event.get('checked'))).lower()}`",
        f"- Candidate branch started: `{str(bool(start_event)).lower()}`",
        f"- Adaptive start epoch: `{payload.get('adaptive_start_epoch')}`",
        f"- No-start reason at epoch 5: `{reasons}`",
        f"- No-op fallback entered: `{str(bool(noop_event)).lower()}`",
        f"- Strict no-op so far: `{str(strict_noop).lower()}`",
        f"- BBox/class legal: `{str(payload['summary']['bbox_class_valid']).lower()}`",
        f"- Industrial samples augmented: `{stats.get('samples_augmented', 0)}`",
        f"- ROI applied: `{roi_stats.get('roi_aug_applied', 0)}`",
        f"- CATF router random draw count: `{stats.get('router_random_draw_count', 0)}`",
        f"- Safe checkpoint saved: `{str(any(event.get('action') == 'save_safe_checkpoint' for event in payload.get('rollback_controller_events', []))).lower()}`",
        "",
        "## Start Condition Trace",
        "",
        "| epoch | action | checked | start_condition | reasons | map50_range | recall_range | eligible_classes | strong_baseline |",
        "|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for event in events:
        cond = event.get("start_condition") or {}
        stability = cond.get("metric_stability") or {}
        lines.append(
            f"| {event.get('epoch')} | {event.get('action')} | {str(event.get('checked')).lower()} | "
            f"{str(event.get('start_condition_met')).lower()} | `{event.get('reasons', [])}` | "
            f"{fmt(stability.get('map50_range'))} | {fmt(stability.get('recall_range'))} | "
            f"`{cond.get('eligible_active_classes', [])}` | `{str(cond.get('strong_clean_baseline_protection', False)).lower()}` |"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- Enter the seed2 50ep adaptive-RB run only if the smoke remains strict no-op before candidate start, bbox/class validation is clean, and CATF router random draws remain zero while burn-in is observing.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_diagnosis_only_smoke_report(payload: dict[str, Any]) -> str:
    stats = payload.get("online_aug_stats") or {}
    roi_stats = payload.get("roi_aug_stats") or {}
    continuity = payload.get("continuity") or {}
    history = payload.get("policy_history") or []
    diagnosis_callbacks = len(history)
    policy_updates = sum(1 for record in history if record.get("policy_update_applied", True))
    lines = [
        "# Diagnosis-Only In-Loop Control Smoke Report",
        "",
        "## Run Integrity",
        "",
        f"- Training success: `{str(payload['train']['success']).lower()}`",
        f"- Diagnosis-only enabled: `{str(payload.get('diagnosis_only', False)).lower()}`",
        f"- YOLO default augmentation enabled: `{str(payload['summary']['yolo_default_augmentation_enabled']).lower()}`",
        f"- Industrial augmentation enabled: `{str(payload['industrial_aug_enabled']).lower()}`",
        f"- Single-run continuous training: `{str(payload['summary']['single_run_inloop_feedback']).lower()}`",
        f"- Stage restart count: `{continuity.get('stage_restart_count')}`",
        f"- Epoch sequence continuous: `{str(continuity.get('epoch_continuous')).lower()}`",
        f"- Epoch sequence: `{continuity.get('epoch_sequence')}`",
        f"- Train image count: `{stats.get('train_image_count')}`",
        f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
        f"- BBox/class legal: `{str(payload['summary']['bbox_class_valid']).lower()}`",
        "",
        "## Diagnosis Control Checks",
        "",
        f"- Diagnosis callback count: `{diagnosis_callbacks}`",
        f"- Feedback epochs: `{payload['summary'].get('feedback_epochs')}`",
        f"- Industrial samples augmented: `{stats.get('samples_augmented', 0)}`",
        f"- Industrial op stats: `{stats.get('ops', {})}`",
        f"- ROI applied: `{roi_stats.get('roi_aug_applied', 0)}`",
        f"- Policy update applied count: `{policy_updates}`",
        f"- Policy history path: `{payload['summary'].get('policy_history')}`",
        "",
        "## Conclusion",
        "",
        "- This smoke validates diagnosis callback execution without industrial augmentation, ROI augmentation, sample routing, threshold mutation, or policy-state mutation.",
    ]
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
    if bool(getattr(args, "diagnosis_only", False)):
        return output_dir / "reports" / "diagnosis_only_smoke_report.md"
    if not bool(args.feedback_enabled) and not bool(args.industrial_aug_enabled):
        return output_dir / "reports" / "inloop_no_feedback_control_report.md"
    if int(args.epochs) <= 10:
        if is_catf_v2(args):
            if bool(getattr(args, "adaptive_burnin", False)):
                return output_dir / "reports" / "adaptive_burnin_smoke_report.md"
            return output_dir / "reports" / "catf_v2_smoke_report.md"
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
    safe_events = payload.get("safe_controller_events") or []
    gated_events = payload.get("gated_controller_events") or []
    adaptive_events = payload.get("adaptive_burnin_events") or []
    rollback_events = payload.get("rollback_controller_events") or []
    riskguard_events = payload.get("riskguard_events") or []
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
        f"- CATF-v2 safe mode enabled: `{str(payload.get('catf_safe_mode', False)).lower()}`",
        f"- CATF-v2 gated mode enabled: `{str(payload.get('catf_gated_mode', False)).lower()}`",
        f"- CATF-v2 rollback mode enabled: `{str(payload.get('catf_rollback_mode', False)).lower()}`",
        f"- CATF-v2 RiskGuard enabled: `{str(payload.get('catf_riskguard', False)).lower()}`",
        f"- CATF-v2 causal probe enabled: `{str(payload.get('catf_causal_probe', False)).lower()}`",
        f"- Offline causal probe decisions used: `{str(payload.get('use_offline_probe_decisions', False)).lower()}`",
        f"- RiskGuard blocked ops: `{len(riskguard_events)}`",
        f"- RiskGuard sampler-only fallback: `{str(any(event.get('sampler_only_fallback') for event in riskguard_events)).lower()}`",
        f"- Adaptive burn-in enabled: `{str(payload.get('adaptive_burnin', False)).lower()}`",
        f"- Adaptive start epoch: `{payload.get('adaptive_start_epoch')}`",
        f"- Adaptive candidate started: `{str(any(event.get('candidate_branch_started') for event in adaptive_events)).lower()}`",
        f"- Adaptive no-op fallback: `{str(any(event.get('action') == 'no_op_fallback' for event in adaptive_events)).lower()}`",
        f"- RB rollback triggered: `{str(any(event.get('action') == 'rollback' for event in rollback_events)).lower()}`",
        f"- Safe no-op fallback triggered: `{str(any(event.get('action') == 'no_op_freeze' for event in safe_events)).lower()}`",
        f"- Safe controller reasons: `{safe_event_reasons(safe_events)}`",
        f"- Gated no-op fallback triggered: `{str(any(event.get('action') == 'no_op_freeze' for event in gated_events)).lower()}`",
        f"- Gated controller reasons: `{safe_event_reasons(gated_events)}`",
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


def safe_event_reasons(events: list[dict[str, Any]]) -> str:
    items = []
    for event in events:
        reasons = event.get("reasons") or []
        if reasons:
            items.append(f"{event.get('epoch')}:{'/'.join(str(reason) for reason in reasons)}")
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


def validate_paper_probe_configuration(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "paper_probe_mode", False)):
        return {}
    data_yaml = Path(args.data).resolve()
    probe_yaml = Path(getattr(args, "probe_data", "")).resolve()
    if data_yaml == probe_yaml:
        raise ValueError("paper probe mode requires probe_data to differ from final training data yaml")
    if not bool(getattr(args, "forbid_final_val_policy_selection", False)):
        raise ValueError("paper probe mode requires forbid_final_val_policy_selection=true")
    train_images, _ = resolve_split_image_label_dirs(data_yaml, "train")
    final_val_images, _ = resolve_split_image_label_dirs(data_yaml, "val")
    probe_images, _ = resolve_split_image_label_dirs(probe_yaml, "val")
    train_set = canonical_file_set(train_images)
    val_set = canonical_file_set(final_val_images)
    probe_set = canonical_file_set(probe_images)
    train_probe_overlap = sorted(str(path) for path in (train_set & probe_set))[:20]
    probe_val_overlap = sorted(str(path) for path in (probe_set & val_set))[:20]
    train_val_overlap = sorted(str(path) for path in (train_set & val_set))[:20]
    audit = {
        "paper_probe_mode": True,
        "policy_selection_source": "probe_split",
        "policy_selection_data_yaml": str(probe_yaml),
        "train_core_data_yaml": str(data_yaml),
        "final_val_used_for_policy_selection": False,
        "forbid_final_val_policy_selection": True,
        "train_core_images_dir": str(train_images.resolve()),
        "probe_images_dir": str(probe_images.resolve()),
        "final_val_images_dir": str(final_val_images.resolve()),
        "train_core_image_count": len(train_set),
        "probe_image_count": len(probe_set),
        "final_val_image_count": len(val_set),
        "train_core_probe_overlap_count": len(train_set & probe_set),
        "probe_final_val_overlap_count": len(probe_set & val_set),
        "train_core_final_val_overlap_count": len(train_set & val_set),
        "train_core_probe_overlap": train_probe_overlap,
        "probe_final_val_overlap": probe_val_overlap,
        "train_core_final_val_overlap": train_val_overlap,
    }
    if audit["probe_final_val_overlap_count"] or audit["train_core_probe_overlap_count"] or audit["train_core_final_val_overlap_count"]:
        raise ValueError(f"paper probe split leakage detected: {audit}")
    return audit


def canonical_file_set(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.resolve() for path in root.rglob("*") if path.is_file()}


def resolve_split_image_label_dirs(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig")) or {}
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    value = Path(str(data[split]))
    images = value if value.is_absolute() else root / value
    try:
        rel_parts = list(images.relative_to(root).parts)
        if rel_parts and rel_parts[0] == "images":
            labels = root.joinpath("labels", *rel_parts[1:])
        else:
            labels = images.parent.parent / "labels" / images.name
    except ValueError:
        parts = list(images.parts)
        if "images" in parts:
            index = parts.index("images")
            parts[index] = "labels"
            labels = Path(*parts)
        else:
            labels = images.parent.parent / "labels" / images.name
    return images, labels


def resolve_val_image_label_dirs(data_yaml: Path) -> tuple[Path, Path]:
    return resolve_split_image_label_dirs(data_yaml, "val")


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
            "- CATF-v1 is global feedback; CATF-v2 is class-aware, issue-aware, and sample-aware feedback with ROI-aware industrial augmentation.",
            "- CATF-v2 current goal is to reduce CATF-v1 Precision instability by activating only diagnosed classes and freezing stable classes.",
            "- Current CATF-v2 work is smoke-only; no formal 50 epoch CATF-v2 run should be inferred from it.",
            "- No-feedback control disables both feedback and industrial augmentation, using Ultralytics YOLO default augmentation as the behavior check.",
            "- The old YOLO default reference is not the final baseline after parity audit; feedback comparisons should use `clean_native_yolo_default_seed42_50ep`.",
            f"- Output: `outputs/experiments/{payload['run_id']}/`",
            f"- Epochs: `{payload['epochs']}`",
            f"- Feedback enabled: `{str(payload['feedback_enabled']).lower()}`",
            f"- Industrial augmentation enabled: `{str(payload['industrial_aug_enabled']).lower()}`",
            f"- CATF version: `{payload.get('catf_version', 'v1')}`",
            f"- Class-aware feedback: `{str(payload.get('class_aware_feedback', False)).lower()}`",
            f"- ROI-aware augmentation: `{str(payload.get('roi_aware_aug', False)).lower()}`",
            f"- Sample-aware routing: `{str(payload.get('sample_aware_routing', False)).lower()}`",
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
