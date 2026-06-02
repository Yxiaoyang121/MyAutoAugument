from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.bbox.convert import xyxy_to_yolo, yolo_to_xyxy  # noqa: E402
from AutoAugment.feedback_policy_controller import (  # noqa: E402
    FeedbackPolicyController,
    infer_feedback_diagnostics,
)
from AutoAugment.online_augmentation import (  # noqa: E402
    OnlineAugmentationStats,
    OnlinePolicyAugmentor,
)


EXPECTED_ULTRALYTICS_VERSION = "8.3.221"
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep"
    / "short_training"
    / "selected_policy.json"
)
DEFAULT_DATA = (
    PROJECT_ROOT
    / "outputs"
    / "datasets"
    / "tiled"
    / "tiled_1024_ov20_full_safe_no_ok_position"
    / "data.yaml"
)
DEFAULT_PROJECT = PROJECT_ROOT / "outputs" / "experiments"
DEFAULT_RUN_ID = "20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_smoke"
DISABLED_YOLO_AUG_ARGS: dict[str, Any] = {
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "erasing": 0.0,
    "cutmix": 0.0,
    "bgr": 0.0,
    "close_mosaic": 0,
    "auto_augment": None,
}


@dataclass
class OnlineTrainingContext:
    augmentor: OnlinePolicyAugmentor
    preview_dir: Path
    save_preview: bool
    preview_count: int
    num_classes: int | None = None
    train_image_count: int | None = None
    train_img_path: str | None = None
    global_epoch_offset: int = 0
    total_epochs: int | None = None
    catf_noop: bool = False
    noop_transform_calls: int = 0


def main() -> None:
    args = parse_args()
    output_dir = (Path(args.project) / args.run_id).resolve()
    configure_environment(output_dir)
    api = check_ultralytics_api()
    prepare_output_dirs(output_dir)

    policy = read_json(Path(args.policy))
    write_json(output_dir / "configs" / "policy.json", policy)
    train_config = build_train_config(args, output_dir)
    write_json(output_dir / "configs" / "train_config.json", train_config)

    if args.feedback_enabled:
        payload = run_feedback_training(args, output_dir, api, policy)
        if not args.skip_doc_update:
            update_state_docs(payload)
            export_project_snapshot()
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        if not payload["train"]["success"]:
            raise RuntimeError(f"feedback online augmentation training failed: {payload['train']['error']}")
        if not payload["val"]["success"]:
            raise RuntimeError(f"feedback online augmentation validation failed: {payload['val']['error']}")
        return

    stats = OnlineAugmentationStats()
    augmentor = OnlinePolicyAugmentor(
        policy,
        seed=args.seed,
        copy_paste_enabled=args.online_copy_paste,
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

    train_command = build_train_command(args, output_dir)
    write_text(output_dir / "configs" / "train_command.txt", train_command)
    write_text(output_dir / "logs" / "train.command.txt", train_command)

    train_start = time.time()
    train_success = False
    train_error: str | None = None
    train_result: Any = None
    try:
        trainer_cls = make_online_trainer(api, context)
        model = api["YOLO"](args.model)
        register_epoch_callback(model, context, total_epochs=args.epochs)
        train_kwargs = build_train_kwargs(args, output_dir)
        train_result = model.train(trainer=trainer_cls, **train_kwargs)
        train_success = True
    except Exception as exc:
        train_error = f"{type(exc).__name__}: {exc}"
    train_end = time.time()

    best_pt = output_dir / "train" / "weights" / "best.pt"
    last_pt = output_dir / "train" / "weights" / "last.pt"
    weights_for_val = best_pt if best_pt.exists() else last_pt

    val_success = False
    val_error: str | None = None
    val_metrics: dict[str, Any] = {}
    val_command = build_val_command(args, output_dir, weights_for_val)
    write_text(output_dir / "configs" / "val_command.txt", val_command)
    write_text(output_dir / "logs" / "val.command.txt", val_command)
    val_start = time.time()
    if train_success and weights_for_val.exists():
        try:
            val_result = api["YOLO"](str(weights_for_val)).val(
                data=str(Path(args.data)),
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
            val_success = True
        except Exception as exc:
            val_error = f"{type(exc).__name__}: {exc}"
    elif train_success:
        val_error = f"no weights found for validation under {output_dir / 'train' / 'weights'}"
    else:
        val_error = "validation skipped because training failed"
    val_end = time.time()

    payload = build_report_payload(
        args=args,
        output_dir=output_dir,
        policy=policy,
        context=context,
        train_success=train_success,
        train_error=train_error,
        train_result=train_result,
        train_wall_seconds=train_end - train_start,
        val_success=val_success,
        val_error=val_error,
        val_metrics=val_metrics,
        val_wall_seconds=val_end - val_start,
        weights_for_val=weights_for_val,
    )
    if not args.feedback_enabled:
        add_no_feedback_history(output_dir, payload)
    report_paths = report_paths_for(args, output_dir)
    payload["artifacts"].update({key: str(path.resolve()) for key, path in report_paths.items()})
    if is_formal_50ep_run(args):
        metrics_payload = build_metrics_payload(payload)
        comparison_payload = build_comparison_payload(metrics_payload)
        metrics_payload["comparison"] = comparison_payload
        payload["formal_metrics"] = metrics_payload
        write_json(report_paths["metrics_json"], metrics_payload)
        write_markdown(report_paths["report_md"], build_formal_report(metrics_payload))
        write_markdown(report_paths["comparison_md"], build_comparison_report(metrics_payload, comparison_payload))
    else:
        write_markdown(report_paths["report_md"], build_smoke_report(payload))
    write_json(output_dir / "reports" / "online_aug_stats.json", payload["online_aug_stats"])
    if not args.skip_doc_update:
        update_state_docs(payload)
        export_project_snapshot()

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if not train_success:
        raise RuntimeError(f"online augmentation training failed: {train_error}")
    if not val_success:
        raise RuntimeError(f"online augmentation validation failed: {val_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO with AutoAugment policy applied online in the train dataloader.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-yolo-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--preview-count", type=int, default=50)
    parser.add_argument("--online-copy-paste", action="store_true")
    parser.add_argument("--copy-paste-bank-size", type=int, default=512)
    parser.add_argument("--feedback-enabled", action="store_true")
    parser.add_argument("--feedback-interval", type=int, default=1)
    parser.add_argument("--feedback-start-epoch", type=int, default=1)
    parser.add_argument("--feedback-profile", default="industrial")
    parser.add_argument("--policy-state-path", default=None)
    parser.add_argument("--skip-doc-update", action="store_true")
    parser.add_argument(
        "--yolo-aug-overrides",
        default=None,
        help="Optional JSON object with Ultralytics augmentation kwargs to apply on top of its defaults.",
    )
    return parser.parse_args()


def configure_environment(output_dir: Path) -> None:
    yolo_config = output_dir / "configs" / "ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def check_ultralytics_api() -> dict[str, Any]:
    try:
        import inspect
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.data import build_dataloader
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.utils import colorstr
        from ultralytics.utils.instance import Instances
        from ultralytics.utils.torch_utils import unwrap_model
    except Exception as exc:
        raise RuntimeError(f"failed to import Ultralytics online augmentation hooks: {exc}") from exc

    version = getattr(ultralytics, "__version__", "<unknown>")
    if version != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Ultralytics API version mismatch: expected {EXPECTED_ULTRALYTICS_VERSION}, got {version}. "
            "Review DetectionTrainer.build_dataset and YOLODataset.build_transforms before running online augmentation."
        )
    build_dataset_sig = inspect.signature(DetectionTrainer.build_dataset)
    if not {"img_path", "mode", "batch"}.issubset(build_dataset_sig.parameters):
        raise RuntimeError(f"unsupported DetectionTrainer.build_dataset signature: {build_dataset_sig}")
    if not hasattr(YOLODataset, "build_transforms") or not hasattr(YOLODataset, "get_image_and_label"):
        raise RuntimeError("unsupported YOLODataset API: missing build_transforms or get_image_and_label")
    return {
        "YOLO": YOLO,
        "YOLODataset": YOLODataset,
        "DetectionTrainer": DetectionTrainer,
        "Instances": Instances,
        "build_dataloader": build_dataloader,
        "colorstr": colorstr,
        "unwrap_model": unwrap_model,
        "version": version,
    }


def prepare_output_dirs(output_dir: Path) -> None:
    for subdir in ["configs", "reports", "previews", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def run_feedback_training(args: argparse.Namespace, output_dir: Path, api: dict[str, Any], initial_policy: dict[str, Any]) -> dict[str, Any]:
    stats = OnlineAugmentationStats()
    augmentor = OnlinePolicyAugmentor(
        initial_policy,
        seed=args.seed,
        copy_paste_enabled=args.online_copy_paste,
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
    policy_state_path = Path(args.policy_state_path) if args.policy_state_path else output_dir / "configs" / "policy_state.json"
    controller = FeedbackPolicyController(
        initial_policy,
        history_dir=output_dir / "reports",
        policy_state_path=policy_state_path,
        profile=args.feedback_profile,
    )
    stage_lengths = build_feedback_stage_lengths(args.epochs, args.feedback_interval)
    stage_records: list[dict[str, Any]] = []
    current_model: str | Path = args.model
    global_epoch = 0
    train_success = True
    train_error: str | None = None
    val_success = True
    val_error: str | None = None
    final_val_metrics: dict[str, Any] = {}
    final_weights = Path()

    for stage_index, stage_epochs in enumerate(stage_lengths):
        stage_policy = deepcopy(controller.policy)
        augmentor.set_policy(stage_policy)
        context.global_epoch_offset = global_epoch
        context.total_epochs = args.epochs
        augmentor.set_epoch(global_epoch, args.epochs)
        write_json(output_dir / "configs" / f"stage_{stage_index}_policy.json", stage_policy)

        train_name = f"train_stage{stage_index}"
        val_name = f"val_stage{stage_index}"
        train_command = build_train_command(
            args,
            output_dir,
            epochs=stage_epochs,
            name=train_name,
            model_override=current_model,
        )
        write_text(output_dir / "configs" / f"stage_{stage_index}_train_command.txt", train_command)
        write_text(output_dir / "logs" / f"stage_{stage_index}.train.command.txt", train_command)

        stage_train_success = False
        stage_train_error: str | None = None
        train_start = time.time()
        try:
            trainer_cls = make_online_trainer(api, context)
            model = api["YOLO"](str(current_model))
            register_epoch_callback(model, context, total_epochs=args.epochs)
            train_kwargs = build_train_kwargs(args, output_dir, epochs=stage_epochs, name=train_name)
            train_result = model.train(trainer=trainer_cls, **train_kwargs)
            stage_train_success = True
            train_result_type = type(train_result).__name__
        except Exception as exc:
            stage_train_error = f"{type(exc).__name__}: {exc}"
            train_result_type = None
        train_end = time.time()

        stage_dir = output_dir / train_name
        best_pt = stage_dir / "weights" / "best.pt"
        last_pt = stage_dir / "weights" / "last.pt"
        weights_for_val = last_pt if last_pt.exists() else best_pt
        final_weights = weights_for_val

        stage_val_success = False
        stage_val_error: str | None = None
        stage_val_metrics: dict[str, Any] = {}
        val_command = build_val_command(args, output_dir, weights_for_val, name=val_name)
        write_text(output_dir / "configs" / f"stage_{stage_index}_val_command.txt", val_command)
        write_text(output_dir / "logs" / f"stage_{stage_index}.val.command.txt", val_command)
        val_start = time.time()
        if stage_train_success and weights_for_val.exists():
            try:
                val_result = api["YOLO"](str(weights_for_val)).val(
                    data=str(Path(args.data)),
                    imgsz=args.imgsz,
                    batch=args.batch,
                    workers=args.workers,
                    device=str(args.device),
                    project=str(output_dir),
                    name=val_name,
                    exist_ok=True,
                    plots=False,
                )
                stage_val_metrics = metrics_to_dict(val_result)
                stage_val_success = True
            except Exception as exc:
                stage_val_error = f"{type(exc).__name__}: {exc}"
        elif stage_train_success:
            stage_val_error = f"no weights found for validation under {stage_dir / 'weights'}"
        else:
            stage_val_error = "validation skipped because stage training failed"
        val_end = time.time()

        val_image_count = count_split_images(Path(args.data), "val")
        if stage_val_metrics and stage_val_metrics.get("images") is None:
            stage_val_metrics["images"] = val_image_count

        diagnosis: dict[str, Any] = {}
        policy_updated = False
        next_global_epoch = global_epoch + stage_epochs
        if (
            stage_val_success
            and stage_index < len(stage_lengths) - 1
            and next_global_epoch >= int(args.feedback_start_epoch)
        ):
            diagnosis = infer_feedback_diagnostics(stage_val_metrics, profile=args.feedback_profile)
            updated_policy = controller.update(diagnosis, stage_index=stage_index, metrics=stage_val_metrics)
            write_json(output_dir / "configs" / f"stage_{stage_index}_updated_policy.json", updated_policy)
            policy_updated = True

        stage_records.append(
            {
                "stage_index": stage_index,
                "epochs": stage_epochs,
                "global_epoch_start": global_epoch,
                "global_epoch_end": next_global_epoch,
                "train_name": train_name,
                "val_name": val_name,
                "train_success": stage_train_success,
                "train_error": stage_train_error,
                "train_wall_seconds": train_end - train_start,
                "train_result_type": train_result_type,
                "val_success": stage_val_success,
                "val_error": stage_val_error,
                "val_wall_seconds": val_end - val_start,
                "val_metrics": stage_val_metrics,
                "weights_for_val": str(weights_for_val) if weights_for_val.exists() else None,
                "policy_updated": policy_updated,
                "feedback_diagnostics": diagnosis,
            }
        )

        train_success = train_success and stage_train_success
        val_success = val_success and stage_val_success
        train_error = train_error or stage_train_error
        val_error = val_error or stage_val_error
        final_val_metrics = stage_val_metrics or final_val_metrics
        if not stage_train_success or not stage_val_success:
            break
        current_model = weights_for_val if weights_for_val.exists() else current_model
        global_epoch = next_global_epoch

    write_json(output_dir / "configs" / "policy_final.json", controller.policy)
    fixed_augmented_dataset_generated = any(
        path.exists()
        for path in [
            output_dir / "dataset" / "final_dataset" / "images",
            output_dir / "dataset_builder" / "final_dataset" / "images",
        ]
    )
    stats_payload = stats.to_dict()
    stats_payload.update(
        {
            "run_id": args.run_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "online_augmentation": True,
            "feedback_enabled": True,
            "feedback_interval": int(args.feedback_interval),
            "feedback_start_epoch": int(args.feedback_start_epoch),
            "feedback_profile": str(args.feedback_profile),
            "train_image_count": context.train_image_count,
            "expected_original_train_images": 2301,
            "train_image_count_matches_original": context.train_image_count == 2301,
            "fixed_augmented_dataset_generated": fixed_augmented_dataset_generated,
            "policy_id": initial_policy.get("policy_id", initial_policy.get("name")),
            "copy_paste_online_supported": False,
            "copy_paste_note": "online copy-paste is pending object-bank implementation; copy_paste ops are skipped safely.",
            "train_success": train_success,
            "train_error": train_error,
            "val_success": val_success,
            "val_error": val_error,
            "val_metrics": final_val_metrics,
            "policy_history_json": str((output_dir / "reports" / "policy_history.json").resolve()),
            "policy_history_md": str((output_dir / "reports" / "policy_history.md").resolve()),
            "policy_history_csv": str((output_dir / "reports" / "policy_history.csv").resolve()),
        }
    )
    payload = {
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "data": str(args.data),
        "model": str(args.model),
        "epochs": int(args.epochs),
        "disable_yolo_aug": bool(args.disable_yolo_aug),
        "policy": initial_policy,
        "final_policy": controller.policy,
        "mode": "only_custom_online_aug" if args.disable_yolo_aug else "yolo_default_plus_custom_online_aug",
        "online_aug_stats": stats_payload,
        "feedback": {
            "enabled": True,
            "stage_count": len(stage_records),
            "stages": stage_records,
            "policy_history_count": len(controller.history),
            "policy_history_json": str((output_dir / "reports" / "policy_history.json").resolve()),
            "policy_history_md": str((output_dir / "reports" / "policy_history.md").resolve()),
            "policy_history_csv": str((output_dir / "reports" / "policy_history.csv").resolve()),
        },
        "summary": {
            "online_augmentation_implemented": True,
            "feedback_enabled": True,
            "policy_history_updates": len(controller.history),
            "fixed_augmented_dataset_generated": fixed_augmented_dataset_generated,
            "train_image_count": context.train_image_count,
            "train_success": train_success,
            "val_success": val_success,
            "val_metrics": final_val_metrics,
            "preview_dir": str((output_dir / "previews").resolve()),
            "online_aug_stats": str((output_dir / "reports" / "online_aug_stats.json").resolve()),
            "policy_history": str((output_dir / "reports" / "policy_history.json").resolve()),
            "copy_paste_supported": False,
        },
        "train": {
            "success": train_success,
            "error": train_error,
            "weights_for_val": str(final_weights) if final_weights.exists() else None,
            "stages": stage_records,
        },
        "val": {
            "success": val_success,
            "error": val_error,
            "metrics": final_val_metrics,
        },
        "artifacts": {
            "policy_json": str((output_dir / "configs" / "policy.json").resolve()),
            "train_config_json": str((output_dir / "configs" / "train_config.json").resolve()),
            "report_md": str((output_dir / "reports" / "online_aug_smoke_report.md").resolve()),
            "stats_json": str((output_dir / "reports" / "online_aug_stats.json").resolve()),
            "preview_dir": str((output_dir / "previews").resolve()),
        },
    }
    write_json(output_dir / "reports" / "online_aug_stats.json", stats_payload)
    write_markdown(output_dir / "reports" / "online_aug_smoke_report.md", build_feedback_smoke_report(payload))
    return payload


def build_feedback_stage_lengths(total_epochs: int, interval: int) -> list[int]:
    total = max(1, int(total_epochs))
    step = max(1, int(interval))
    lengths = []
    remaining = total
    while remaining > 0:
        current = min(step, remaining)
        lengths.append(current)
        remaining -= current
    return lengths


def add_no_feedback_history(output_dir: Path, payload: dict[str, Any]) -> None:
    history_json = output_dir / "reports" / "policy_history.json"
    history_md = output_dir / "reports" / "policy_history.md"
    history_csv = output_dir / "reports" / "policy_history.csv"
    history_payload = {
        "feedback_applied": False,
        "note": "no feedback applied for this base policy run",
        "history": [],
        "latest_policy": payload["policy"],
    }
    write_json(history_json, history_payload)
    write_markdown(
        history_md,
        "\n".join(
            [
                "# Policy History",
                "",
                "- Feedback applied: `false`",
                "- Note: no feedback applied for this base policy run.",
            ]
        )
        + "\n",
    )
    write_text(history_csv, "stage_index,op,field,before,after,reason\n")
    paths = {
        "policy_history_json": str(history_json.resolve()),
        "policy_history_md": str(history_md.resolve()),
        "policy_history_csv": str(history_csv.resolve()),
    }
    payload["online_aug_stats"].update({"feedback_enabled": False, "policy_history_count": 0, **paths})
    payload["summary"].update({"feedback_enabled": False, "policy_history_updates": 0, "policy_history": paths["policy_history_json"]})


def make_online_trainer(api: dict[str, Any], context: OnlineTrainingContext):
    YOLODataset = api["YOLODataset"]
    DetectionTrainer = api["DetectionTrainer"]
    Instances = api["Instances"]
    colorstr = api["colorstr"]
    unwrap_model = api["unwrap_model"]

    class OnlineYOLODataset(YOLODataset):
        def __init__(self, *dataset_args, online_context: OnlineTrainingContext | None = None, **dataset_kwargs):
            self.online_context = online_context
            super().__init__(*dataset_args, **dataset_kwargs)
            if self.online_context is not None:
                self.online_context.augmentor.set_sample_provider(self.sample_online_source)

        def build_transforms(self, hyp: dict | None = None):
            transforms = super().build_transforms(hyp)
            if self.augment and self.online_context is not None:
                transforms.insert(0, UltralyticsOnlinePolicyTransform(self.online_context, Instances))
            return transforms

        def sample_online_source(self, rng: np.random.Generator):
            if len(self) <= 0:
                return None
            index = int(rng.integers(0, len(self)))
            label = self.get_image_and_label(index)
            image = label.get("img")
            instances = label.get("instances")
            if image is None or instances is None:
                return None
            height, width = image.shape[:2]
            class_values = np.asarray(label.get("cls", np.zeros((0, 1))), dtype=np.float32).reshape(-1).astype(np.int64)
            bboxes = instances_to_xyxy(instances, width=width, height=height)
            return image, class_values, bboxes

    class OnlineAugDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            if mode != "train":
                return super().build_dataset(img_path, mode=mode, batch=batch)
            data = self.data
            context.num_classes = int(data.get("nc", len(data.get("names", [])))) if isinstance(data, dict) else None
            context.augmentor.num_classes = context.num_classes
            dataset = OnlineYOLODataset(
                img_path=img_path,
                imgsz=self.args.imgsz,
                batch_size=batch,
                augment=True,
                hyp=self.args,
                rect=self.args.rect,
                cache=self.args.cache or None,
                single_cls=self.args.single_cls or False,
                stride=gs,
                pad=0.0,
                prefix=colorstr("train: "),
                task=self.args.task,
                classes=self.args.classes,
                data=data,
                fraction=self.args.fraction,
                online_context=context,
            )
            context.train_image_count = len(dataset)
            context.train_img_path = str(img_path)
            return dataset

    return OnlineAugDetectionTrainer


def register_epoch_callback(model: Any, context: OnlineTrainingContext, *, total_epochs: int) -> None:
    def _on_train_epoch_start(trainer: Any) -> None:
        local_epoch = int(getattr(trainer, "epoch", 0) or 0)
        context.augmentor.set_epoch(context.global_epoch_offset + local_epoch, total_epochs)

    if hasattr(model, "add_callback"):
        model.add_callback("on_train_epoch_start", _on_train_epoch_start)


class UltralyticsOnlinePolicyTransform:
    def __init__(self, context: OnlineTrainingContext, instances_cls: Any) -> None:
        self.context = context
        self.instances_cls = instances_cls

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if getattr(self.context, "catf_noop", False):
            self.context.noop_transform_calls += 1
            return labels

        image = labels.get("img")
        instances = labels.get("instances")
        if image is None or instances is None:
            raise RuntimeError("online augmentation expected Ultralytics labels with 'img' and 'instances'")
        height, width = image.shape[:2]
        class_values = np.asarray(labels.get("cls", np.zeros((0, 1))), dtype=np.float32).reshape(-1)
        bboxes = instances_to_xyxy(instances, width=width, height=height)
        if len(class_values) != len(bboxes):
            raise RuntimeError(f"class/bbox count mismatch before online augmentation: {len(class_values)} vs {len(bboxes)}")

        before_image = image.copy() if self.context.save_preview and self.context.augmentor.stats.preview_saved < self.context.preview_count else None
        before_boxes = bboxes.copy() if before_image is not None else None
        before_labels = class_values.astype(np.int64, copy=True) if before_image is not None else None

        result = self.context.augmentor.apply(image, class_values.astype(np.int64), bboxes)
        applied_any_aug = bool(result.audit.get("applied_any_aug", bool(result.audit.get("applied_ops"))))
        if not applied_any_aug:
            return labels
        labels["img"] = result.image
        labels["cls"] = result.labels.reshape(-1, 1).astype(np.float32)
        labels["instances"] = make_instances(
            self.instances_cls,
            result.bboxes,
            width=result.image.shape[1],
            height=result.image.shape[0],
        )

        if before_image is not None and before_boxes is not None and before_labels is not None:
            save_preview(
                self.context.preview_dir,
                self.context.augmentor.stats.preview_saved,
                before_image,
                before_labels,
                before_boxes,
                result.image,
                result.labels,
                result.bboxes,
                result.audit,
            )
            self.context.augmentor.stats.preview_saved += 1
        return labels


def instances_to_xyxy(instances: Any, *, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(instances.bboxes, dtype=np.float32).reshape(-1, 4).copy()
    bbox_format = getattr(getattr(instances, "_bboxes", None), "format", "xywh")
    normalized = bool(getattr(instances, "normalized", True))
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if normalized:
        if bbox_format == "xywh":
            return yolo_to_xyxy(boxes, width, height)
        if bbox_format == "xyxy":
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            return boxes
        if bbox_format == "ltwh":
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
            return np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], axis=1)
    if bbox_format == "xyxy":
        return boxes
    if bbox_format == "xywh":
        out = boxes.copy()
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out
    if bbox_format == "ltwh":
        return np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], axis=1)
    raise RuntimeError(f"unsupported Ultralytics bbox format for online augmentation: {bbox_format}")


def make_instances(instances_cls: Any, bboxes_xyxy: np.ndarray, *, width: int, height: int) -> Any:
    yolo_boxes = xyxy_to_yolo(np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4), width, height)
    segments = np.zeros((0, 1000, 2), dtype=np.float32)
    return instances_cls(yolo_boxes, segments=segments, bbox_format="xywh", normalized=True)


def build_train_kwargs(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    epochs: int | None = None,
    name: str = "train",
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(Path(args.data)),
        "epochs": int(epochs if epochs is not None else args.epochs),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": str(args.device),
        "seed": args.seed,
        "project": str(output_dir),
        "name": name,
        "exist_ok": True,
        "plots": False,
    }
    if args.disable_yolo_aug:
        kwargs.update(DISABLED_YOLO_AUG_ARGS)
    else:
        kwargs.update(parse_yolo_aug_overrides(args.yolo_aug_overrides))
    return kwargs


def build_train_command(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    epochs: int | None = None,
    name: str = "train",
    model_override: str | Path | None = None,
) -> str:
    parts = [
        "YOLO.train",
        f"model={model_override if model_override is not None else args.model}",
        f"data={Path(args.data)}",
        f"policy={Path(args.policy)}",
        f"epochs={int(epochs if epochs is not None else args.epochs)}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"project={output_dir}",
        f"name={name}",
        "trainer=OnlineAugDetectionTrainer",
    ]
    if args.disable_yolo_aug:
        parts.extend(f"{key}={value}" for key, value in DISABLED_YOLO_AUG_ARGS.items())
    elif args.yolo_aug_overrides:
        parts.append(f"yolo_aug_overrides={args.yolo_aug_overrides}")
    return subprocess.list2cmdline([str(part) for part in parts])


def build_val_command(args: argparse.Namespace, output_dir: Path, weights: Path, *, name: str = "val") -> str:
    parts = [
        "YOLO.val",
        f"model={weights}",
        f"data={Path(args.data)}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"device={args.device}",
        f"project={output_dir}",
        f"name={name}",
        "exist_ok=True",
    ]
    return subprocess.list2cmdline([str(part) for part in parts])


def build_train_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "model": args.model,
        "data": str(Path(args.data)),
        "policy": str(Path(args.policy)),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(args.device),
        "workers": args.workers,
        "seed": args.seed,
        "disable_yolo_aug": bool(args.disable_yolo_aug),
        "disabled_yolo_aug_args": DISABLED_YOLO_AUG_ARGS if args.disable_yolo_aug else {},
        "save_preview": bool(args.save_preview),
        "preview_count": args.preview_count,
        "online_copy_paste": bool(args.online_copy_paste),
        "copy_paste_bank_size": args.copy_paste_bank_size,
        "feedback_enabled": bool(args.feedback_enabled),
        "feedback_interval": int(args.feedback_interval),
        "feedback_start_epoch": int(args.feedback_start_epoch),
        "feedback_profile": str(args.feedback_profile),
        "policy_state_path": args.policy_state_path,
        "yolo_aug_overrides": parse_yolo_aug_overrides(args.yolo_aug_overrides),
        "mode": "only_custom_online_aug" if args.disable_yolo_aug else "yolo_default_plus_custom_online_aug",
    }


def parse_yolo_aug_overrides(raw: str | None) -> dict[str, Any]:
    if raw is None or str(raw).strip() == "":
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--yolo-aug-overrides must be a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--yolo-aug-overrides must decode to a JSON object")
    allowed = {
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "fliplr",
        "flipud",
        "mosaic",
        "mixup",
        "copy_paste",
        "erasing",
        "cutmix",
        "bgr",
        "close_mosaic",
        "auto_augment",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unsupported YOLO augmentation override(s): {unknown}")
    return payload


def save_preview(
    preview_dir: Path,
    index: int,
    before_image: np.ndarray,
    before_labels: np.ndarray,
    before_bboxes: np.ndarray,
    after_image: np.ndarray,
    after_labels: np.ndarray,
    after_bboxes: np.ndarray,
    audit: dict[str, Any],
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    before = draw_boxes(before_image, before_labels, before_bboxes, title="before")
    after = draw_boxes(after_image, after_labels, after_bboxes, title="after " + ",".join(audit.get("applied_ops", [{}])[i].get("name", "") for i in range(min(3, len(audit.get("applied_ops", []))))))
    height = max(before.shape[0], after.shape[0])
    canvas = np.full((height, before.shape[1] + after.shape[1], 3), 255, dtype=np.uint8)
    canvas[: before.shape[0], : before.shape[1]] = before
    canvas[: after.shape[0], before.shape[1] :] = after
    cv2.imwrite(str(preview_dir / f"before_after_{index:03d}.jpg"), canvas)


def draw_boxes(image: np.ndarray, labels: np.ndarray, bboxes: np.ndarray, *, title: str) -> np.ndarray:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for label, box in zip(labels, bboxes):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(canvas, str(int(label)), (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, title[:90], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 230), 2, cv2.LINE_AA)
    return canvas


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {}
    per_class = []
    names = getattr(metrics, "names", {}) or {}
    nt_per_class = np.asarray(getattr(metrics, "nt_per_class", []))
    nt_per_image = np.asarray(getattr(metrics, "nt_per_image", []))
    ap_class_index = np.asarray(getattr(box, "ap_class_index", []), dtype=int)
    for result_index, class_id in enumerate(ap_class_index.tolist()):
        precision, recall, ap50, ap50_95 = box.class_result(result_index)
        per_class.append(
            {
                "class_id": int(class_id),
                "name": str(names.get(int(class_id), class_id)) if isinstance(names, dict) else str(class_id),
                "images": int(nt_per_image[class_id]) if class_id < len(nt_per_image) else None,
                "instances": int(nt_per_class[class_id]) if class_id < len(nt_per_class) else None,
                "precision": float(precision),
                "recall": float(recall),
                "ap50": float(ap50),
                "ap50_95": float(ap50_95),
            }
        )
    return {
        "images": None,
        "instances": int(np.sum(nt_per_class)) if len(nt_per_class) else None,
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "per_class": per_class,
    }


def count_split_images(data_yaml: Path, split: str) -> int | None:
    try:
        import yaml
    except Exception:
        return None
    try:
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig")) or {}
    except Exception:
        return None
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    value = data.get(split)
    if value is None:
        return None
    split_paths = value if isinstance(value, list) else [value]
    count = 0
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for item in split_paths:
        path = Path(str(item))
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            count += sum(1 for child in path.rglob("*") if child.suffix.lower() in suffixes)
        elif path.is_file():
            lines = [line.strip() for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()]
            count += sum(1 for line in lines if line and Path(line).suffix.lower() in suffixes)
    return count


def formal_run_kind(args: argparse.Namespace) -> str | None:
    if int(args.epochs) < 50:
        return None
    run_id = str(args.run_id)
    if "online_random_like" in run_id:
        return "online_random_like"
    if "custom_yolo_like_base" in run_id:
        return "custom_yolo_like_base"
    if "online_diag_policy_001" in run_id:
        return "online_diag_policy_001"
    return None


def is_formal_50ep_run(args: argparse.Namespace) -> bool:
    return formal_run_kind(args) is not None


def report_paths_for(args: argparse.Namespace, output_dir: Path) -> dict[str, Path]:
    kind = formal_run_kind(args)
    if kind == "online_random_like":
        return {
            "report_md": output_dir / "reports" / "online_random_like_50ep_report.md",
            "metrics_json": output_dir / "reports" / "online_random_like_metrics.json",
            "comparison_md": output_dir / "reports" / "compare_online_random_like_with_all.md",
        }
    if kind == "online_diag_policy_001":
        return {
            "report_md": output_dir / "reports" / "online_diag_policy_001_50ep_report.md",
            "metrics_json": output_dir / "reports" / "online_diag_policy_001_50ep_metrics.json",
            "comparison_md": output_dir / "reports" / "compare_online_offline_yolo_default_random.md",
        }
    if kind == "custom_yolo_like_base":
        return {
            "report_md": output_dir / "reports" / "custom_yolo_like_base_50ep_report.md",
            "metrics_json": output_dir / "reports" / "custom_yolo_like_base_50ep_metrics.json",
            "comparison_md": output_dir / "reports" / "compare_custom_yolo_like_with_yolo_default.md",
        }
    return {"report_md": output_dir / "reports" / "online_aug_smoke_report.md"}


def compact_overall(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in ["images", "instances", "precision", "recall", "map50", "map50_95"]}


def build_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(payload["output_dir"])
    val_metrics = payload["val"]["metrics"]
    stats = payload["online_aug_stats"]
    best_pt = output_dir / "train" / "weights" / "best.pt"
    last_pt = output_dir / "train" / "weights" / "last.pt"
    references = load_reference_metrics()
    final_metrics = compact_overall(val_metrics)
    result = {
        "run_id": payload["run_id"],
        "run_kind": formal_run_kind(argparse.Namespace(epochs=payload.get("epochs", 0), run_id=payload["run_id"])),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": payload.get("data"),
        "model": payload.get("model"),
        "mode": payload["mode"],
        "policy": payload["policy"],
        "online_mechanism": {
            "online_augmentation": True,
            "train_images": stats.get("train_image_count"),
            "train_image_count_matches_original": stats.get("train_image_count_matches_original"),
            "fixed_augmented_dataset_generated": stats.get("fixed_augmented_dataset_generated"),
            "validation_custom_augmentation": False,
            "yolo_builtin_augmentations_disabled": payload.get("disable_yolo_aug", True),
            "copy_paste_online_supported": stats.get("copy_paste_online_supported", False),
            "copy_paste_status": stats.get("copy_paste_status"),
        },
        "online_aug_stats": stats,
        "final_metrics": final_metrics,
        "final_per_class": val_metrics.get("per_class", []),
        "references": references,
        "artifacts": {
            "best_pt": str(best_pt.resolve()) if best_pt.exists() else None,
            "last_pt": str(last_pt.resolve()) if last_pt.exists() else None,
            "train_args_yaml": str((output_dir / "train" / "args.yaml").resolve()),
            "train_results_csv": str((output_dir / "train" / "results.csv").resolve()),
            "preview_dir": payload["artifacts"]["preview_dir"],
            "metrics_json": payload["artifacts"].get("metrics_json"),
            "online_aug_stats": payload["artifacts"]["stats_json"],
            "policy_history_json": stats.get("policy_history_json"),
            "policy_history_md": stats.get("policy_history_md"),
            "policy_history_csv": stats.get("policy_history_csv"),
            "train_command": str((output_dir / "configs" / "train_command.txt").resolve()),
            "val_command": str((output_dir / "configs" / "val_command.txt").resolve()),
        },
        "train": payload["train"],
        "val": payload["val"],
    }
    return result


def load_reference_metrics() -> dict[str, dict[str, Any]]:
    paths = {
        "baseline_no_aug": PROJECT_ROOT
        / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep/reports/baseline_50ep_metrics.json",
        "offline_diag_policy_001": PROJECT_ROOT
        / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep/reports/diagaug_50ep_metrics.json",
        "yolo_default": PROJECT_ROOT
        / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep/reports/yolo_default_aug_50ep_metrics.json",
        "random_external": PROJECT_ROOT
        / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep/reports/random_external_aug_50ep_metrics.json",
        "online_diag_policy_001": PROJECT_ROOT
        / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_online_diag_policy_001_yolo11n_50ep/reports/online_diag_policy_001_50ep_metrics.json",
    }
    references: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.exists():
            references[name] = {"path": str(path), "missing": True}
            continue
        raw = read_json(path)
        references[name] = {
            "path": str(path),
            "missing": False,
            "overall": extract_overall_metrics(raw),
            "per_class": extract_per_class_metrics(raw),
        }
    return references


def extract_overall_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    for key_path in [
        ("final_metrics",),
        ("validation", "overall"),
        ("val_metrics",),
        ("overall",),
    ]:
        current: Any = payload
        for key in key_path:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, dict):
            return compact_overall(current)
    return {}


def extract_per_class_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key_path in [
        ("final_per_class",),
        ("validation", "per_class"),
        ("per_class",),
    ]:
        current: Any = payload
        for key in key_path:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, list):
            return current
    return []


def build_comparison_payload(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    online = metrics_payload["final_metrics"]
    references = metrics_payload.get("references", {})
    run_kind = metrics_payload.get("run_kind") or "online_diag_policy_001"
    current_label = {
        "online_random_like": "Online random-like",
        "custom_yolo_like_base": "Custom YOLO-like base",
        "online_diag_policy_001": "Online DiagAug",
    }.get(run_kind, run_kind)
    order = [
        ("baseline_no_aug", "Baseline no aug"),
        ("offline_diag_policy_001", "Offline DiagAug"),
        ("yolo_default", "YOLO default"),
        ("random_external", "Random external"),
        ("online_diag_policy_001", "Online DiagAug"),
        ("online_random_like", "Online random-like"),
        ("custom_yolo_like_base", "Custom YOLO-like base"),
    ]
    rows = []
    for key, label in order:
        if key == run_kind:
            overall = online
        else:
            overall = references.get(key, {}).get("overall", {})
        if not overall:
            continue
        rows.append({"key": key, "name": label, **compact_overall(overall)})
    deltas = {
        "vs_baseline": metric_delta(online, references.get("baseline_no_aug", {}).get("overall", {})),
        "vs_offline_diag_policy_001": metric_delta(online, references.get("offline_diag_policy_001", {}).get("overall", {})),
        "vs_yolo_default": metric_delta(online, references.get("yolo_default", {}).get("overall", {})),
        "vs_random_external": metric_delta(online, references.get("random_external", {}).get("overall", {})),
        "vs_online_diag_policy_001": metric_delta(online, references.get("online_diag_policy_001", {}).get("overall", {})),
    }
    comparisons = {
        "current_key": run_kind,
        "current_label": current_label,
        "summary_rows": rows,
        **deltas,
        "answers": {
            "online_better_than_offline_diagaug": compare_primary(online, references.get("offline_diag_policy_001", {}).get("overall", {})),
            "online_close_to_yolo_default": close_to_reference(online, references.get("yolo_default", {}).get("overall", {})),
            "online_improves_precision_map": online_improves_precision_map(online, references.get("offline_diag_policy_001", {}).get("overall", {})),
            "online_random_like_better_than_offline_random": compare_primary(online, references.get("random_external", {}).get("overall", {})),
            "online_random_like_better_than_online_diagaug": compare_primary(online, references.get("online_diag_policy_001", {}).get("overall", {})),
            "online_random_like_close_to_offline_random": close_to_reference(online, references.get("random_external", {}).get("overall", {})),
            "custom_yolo_like_close_to_yolo_default": close_to_reference(online, references.get("yolo_default", {}).get("overall", {})),
            "custom_yolo_like_better_than_yolo_default": compare_primary(online, references.get("yolo_default", {}).get("overall", {})),
            "custom_yolo_like_better_than_baseline": compare_primary(online, references.get("baseline_no_aug", {}).get("overall", {})),
            "custom_yolo_like_better_than_online_diagaug": compare_primary(online, references.get("online_diag_policy_001", {}).get("overall", {})),
            "custom_yolo_like_better_than_random_external": compare_primary(online, references.get("random_external", {}).get("overall", {})),
            "is_current_best": is_current_best(online, references, exclude_key=run_kind),
            "random_external_advantage_source": infer_random_external_advantage_source(
                online,
                references.get("random_external", {}).get("overall", {}),
            )
            if run_kind == "online_random_like"
            else None,
            "online_mechanism_more_reasonable": True,
        },
    }
    return comparisons


def metric_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for key in ["precision", "recall", "map50", "map50_95"]:
        if current.get(key) is None or reference.get(key) is None:
            delta[key] = None
        else:
            delta[key] = float(current[key]) - float(reference[key])
    return delta


def compare_primary(current: dict[str, Any], reference: dict[str, Any]) -> bool | None:
    if not current or not reference:
        return None
    return float(current.get("map50_95", 0.0)) > float(reference.get("map50_95", 0.0))


def close_to_reference(current: dict[str, Any], reference: dict[str, Any], tolerance: float = 0.03) -> bool | None:
    if not current or not reference:
        return None
    return abs(float(current.get("map50_95", 0.0)) - float(reference.get("map50_95", 0.0))) <= tolerance


def online_improves_precision_map(current: dict[str, Any], reference: dict[str, Any]) -> bool | None:
    if not current or not reference:
        return None
    return (
        float(current.get("precision", 0.0)) > float(reference.get("precision", 0.0))
        and float(current.get("map50_95", 0.0)) > float(reference.get("map50_95", 0.0))
    )


def is_current_best(current: dict[str, Any], references: dict[str, dict[str, Any]], *, exclude_key: str) -> bool | None:
    if current.get("map50_95") is None:
        return None
    current_score = float(current["map50_95"])
    reference_scores = []
    for key, payload in references.items():
        if key == exclude_key:
            continue
        score = payload.get("overall", {}).get("map50_95")
        if score is not None:
            reference_scores.append(float(score))
    if not reference_scores:
        return None
    return current_score >= max(reference_scores)


def infer_random_external_advantage_source(current: dict[str, Any], offline_random: dict[str, Any]) -> str:
    if current.get("map50_95") is None or offline_random.get("map50_95") is None:
        return "unknown"
    delta = float(current["map50_95"]) - float(offline_random["map50_95"])
    if delta >= -0.01:
        return "mostly_operator_combo"
    if delta >= -0.04:
        return "operator_combo_helps_but_offline_doubling_or_training_variance_still_contributes"
    return "offline_doubling_or_fixed_dataset_effect_likely_contributes_substantially"


def build_report_payload(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    policy: dict[str, Any],
    context: OnlineTrainingContext,
    train_success: bool,
    train_error: str | None,
    train_result: Any,
    train_wall_seconds: float,
    val_success: bool,
    val_error: str | None,
    val_metrics: dict[str, Any],
    val_wall_seconds: float,
    weights_for_val: Path,
) -> dict[str, Any]:
    fixed_augmented_dataset_generated = any(
        path.exists()
        for path in [
            output_dir / "dataset" / "final_dataset" / "images",
            output_dir / "dataset_builder" / "final_dataset" / "images",
        ]
    )
    stats = context.augmentor.stats.to_dict()
    stats.update(
        {
            "run_id": args.run_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "online_augmentation": True,
            "train_image_count": context.train_image_count,
            "expected_original_train_images": 2301,
            "train_image_count_matches_original": context.train_image_count == 2301,
            "fixed_augmented_dataset_generated": fixed_augmented_dataset_generated,
            "policy_id": policy.get("policy_id", policy.get("name")),
            "copy_paste_online_supported": False,
            "copy_paste_note": "online copy-paste is pending object-bank implementation; copy_paste ops are skipped safely.",
            "train_success": train_success,
            "train_error": train_error,
            "val_success": val_success,
            "val_error": val_error,
            "val_metrics": val_metrics,
        }
    )
    train_results_csv = output_dir / "train" / "results.csv"
    val_image_count = count_split_images(Path(args.data), "val")
    if val_metrics and val_metrics.get("images") is None:
        val_metrics["images"] = val_image_count
    summary = {
        "online_augmentation_implemented": True,
        "fixed_augmented_dataset_generated": fixed_augmented_dataset_generated,
        "train_image_count": context.train_image_count,
        "train_success": train_success,
        "val_success": val_success,
        "val_metrics": val_metrics,
        "preview_dir": str((output_dir / "previews").resolve()),
        "online_aug_stats": str((output_dir / "reports" / "online_aug_stats.json").resolve()),
        "copy_paste_supported": False,
    }
    return {
        "run_id": args.run_id,
        "output_dir": str(output_dir),
        "data": str(args.data),
        "model": str(args.model),
        "epochs": int(args.epochs),
        "disable_yolo_aug": bool(args.disable_yolo_aug),
        "policy": policy,
        "mode": "only_custom_online_aug" if args.disable_yolo_aug else "yolo_default_plus_custom_online_aug",
        "online_aug_stats": stats,
        "summary": summary,
        "train": {
            "success": train_success,
            "error": train_error,
            "wall_seconds": train_wall_seconds,
            "result_type": type(train_result).__name__ if train_result is not None else None,
            "results_csv": str(train_results_csv) if train_results_csv.exists() else None,
            "weights_for_val": str(weights_for_val) if weights_for_val.exists() else None,
        },
        "val": {
            "success": val_success,
            "error": val_error,
            "wall_seconds": val_wall_seconds,
            "metrics": val_metrics,
        },
        "artifacts": {
            "policy_json": str((output_dir / "configs" / "policy.json").resolve()),
            "train_config_json": str((output_dir / "configs" / "train_config.json").resolve()),
            "report_md": str((output_dir / "reports" / "online_aug_smoke_report.md").resolve()),
            "stats_json": str((output_dir / "reports" / "online_aug_stats.json").resolve()),
            "preview_dir": str((output_dir / "previews").resolve()),
        },
    }


def build_smoke_report(payload: dict[str, Any]) -> str:
    stats = payload["online_aug_stats"]
    val_metrics = payload["val"]["metrics"]
    ops = stats.get("ops", {})
    lines = [
        "# Online Augmentation Smoke Report",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Online augmentation implemented: `{str(stats['online_augmentation']).lower()}`",
        f"- Train image count: `{stats.get('train_image_count')}`",
        f"- Train image count remains original 2301: `{str(stats.get('train_image_count_matches_original')).lower()}`",
        f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
        f"- Policy: `{stats.get('policy_id')}`",
        f"- Training success: `{str(payload['train']['success']).lower()}`",
        f"- Validation success: `{str(payload['val']['success']).lower()}`",
        f"- Preview dir: `{payload['artifacts']['preview_dir']}`",
        f"- Stats JSON: `{payload['artifacts']['stats_json']}`",
        "",
        "## Validation Metrics",
        "",
        f"- Precision: `{val_metrics.get('precision', 0.0):.4f}`",
        f"- Recall: `{val_metrics.get('recall', 0.0):.4f}`",
        f"- mAP50: `{val_metrics.get('map50', 0.0):.4f}`",
        f"- mAP50-95: `{val_metrics.get('map50_95', 0.0):.4f}`",
        "",
        "## Operation Counts",
        "",
        "| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, counts in sorted(ops.items()):
        lines.append(
            f"| {name} | {counts.get('seen', 0)} | {counts.get('applied', 0)} | "
            f"{counts.get('skipped_probability', 0)} | {counts.get('skipped_safety', 0)} | "
            f"{counts.get('skipped_copy_paste_pending', 0)} | {counts.get('skipped_unsupported', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- Bbox transform valid: `{str(stats.get('invalid_bbox_count', 0) == 0).lower()}`",
            f"- Invalid bbox count: `{stats.get('invalid_bbox_count', 0)}`",
            f"- Bbox out-of-bounds count before clipping: `{stats.get('bbox_oob_count', 0)}`",
            f"- Class id out-of-range count: `{stats.get('class_id_oob_count', 0)}`",
            f"- Cutout holes applied: `{stats.get('cutout_safe', {}).get('holes_applied', 0)}`",
            f"- Cutout skipped by center safety: `{stats.get('cutout_safe', {}).get('holes_skipped_center', 0)}`",
            f"- Cutout skipped by overlap safety: `{stats.get('cutout_safe', {}).get('holes_skipped_overlap', 0)}`",
            f"- Cutout skipped by no safe region: `{stats.get('cutout_safe', {}).get('holes_skipped_no_safe_region', 0)}`",
            "",
            "## Copy-Paste",
            "",
            "- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.",
            "- Current online policy verifies YOLO-like and industrial online operators while leaving copy-paste execution pending.",
        ]
    )
    if payload["train"]["error"]:
        lines.extend(["", "## Train Error", "", f"`{payload['train']['error']}`"])
    if payload["val"]["error"]:
        lines.extend(["", "## Validation Error", "", f"`{payload['val']['error']}`"])
    return "\n".join(lines) + "\n"


def build_feedback_smoke_report(payload: dict[str, Any]) -> str:
    stats = payload["online_aug_stats"]
    val_metrics = payload["val"]["metrics"]
    feedback = payload.get("feedback", {})
    lines = [
        "# Feedback Online Augmentation Smoke Report",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Online augmentation implemented: `{str(stats['online_augmentation']).lower()}`",
        f"- Feedback enabled: `{str(feedback.get('enabled', False)).lower()}`",
        f"- Stage count: `{feedback.get('stage_count', 0)}`",
        f"- Policy history updates: `{feedback.get('policy_history_count', 0)}`",
        f"- Train image count: `{stats.get('train_image_count')}`",
        f"- Train image count remains original 2301: `{str(stats.get('train_image_count_matches_original')).lower()}`",
        f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
        f"- YOLO built-in augmentation disabled: `{str(payload.get('disable_yolo_aug', True)).lower()}`",
        "- Validation custom augmentation: `false`",
        f"- Training success: `{str(payload['train']['success']).lower()}`",
        f"- Validation success: `{str(payload['val']['success']).lower()}`",
        f"- Preview dir: `{payload['artifacts']['preview_dir']}`",
        f"- Stats JSON: `{payload['artifacts']['stats_json']}`",
        f"- Policy history JSON: `{feedback.get('policy_history_json')}`",
        "",
        "## Validation Metrics",
        "",
        f"- Precision: `{val_metrics.get('precision', 0.0):.4f}`",
        f"- Recall: `{val_metrics.get('recall', 0.0):.4f}`",
        f"- mAP50: `{val_metrics.get('map50', 0.0):.4f}`",
        f"- mAP50-95: `{val_metrics.get('map50_95', 0.0):.4f}`",
        "",
        "## Stage Summary",
        "",
        "| stage | epochs | train | val | policy_updated |",
        "|---:|---:|---|---|---|",
    ]
    for stage in feedback.get("stages", []):
        lines.append(
            f"| {stage.get('stage_index')} | {stage.get('epochs')} | "
            f"{str(stage.get('train_success')).lower()} | {str(stage.get('val_success')).lower()} | "
            f"{str(stage.get('policy_updated')).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Operation Counts",
            "",
            "| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported | skipped_close_mosaic |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, counts in sorted(stats.get("ops", {}).items()):
        lines.append(
            f"| {name} | {counts.get('seen', 0)} | {counts.get('applied', 0)} | "
            f"{counts.get('skipped_probability', 0)} | {counts.get('skipped_safety', 0)} | "
            f"{counts.get('skipped_copy_paste_pending', 0)} | {counts.get('skipped_unsupported', 0)} | "
            f"{counts.get('skipped_close_mosaic', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- Bbox transform valid: `{str(stats.get('invalid_bbox_count', 0) == 0 and stats.get('class_id_oob_count', 0) == 0).lower()}`",
            f"- Invalid bbox count: `{stats.get('invalid_bbox_count', 0)}`",
            f"- Bbox out-of-bounds count before clipping: `{stats.get('bbox_oob_count', 0)}`",
            f"- Class id out-of-range count: `{stats.get('class_id_oob_count', 0)}`",
            f"- Cutout holes applied: `{stats.get('cutout_safe', {}).get('holes_applied', 0)}`",
            f"- Cutout skipped by center safety: `{stats.get('cutout_safe', {}).get('holes_skipped_center', 0)}`",
            f"- Cutout skipped by overlap safety: `{stats.get('cutout_safe', {}).get('holes_skipped_overlap', 0)}`",
            f"- Mosaic applied: `{stats.get('mosaic4', {}).get('applied', 0)}`",
            "",
            "## Copy-Paste",
            "",
            "- Online copy-paste pending: object-bank paste is not enabled; copy_paste and class_balanced_copy_paste updates are recorded as pending.",
        ]
    )
    if payload["train"]["error"]:
        lines.extend(["", "## Train Error", "", f"`{payload['train']['error']}`"])
    if payload["val"]["error"]:
        lines.extend(["", "## Validation Error", "", f"`{payload['val']['error']}`"])
    return "\n".join(lines) + "\n"


def build_formal_report(metrics_payload: dict[str, Any]) -> str:
    stats = metrics_payload["online_aug_stats"]
    final_metrics = metrics_payload["final_metrics"]
    comparison = metrics_payload.get("comparison", {})
    run_kind = metrics_payload.get("run_kind")
    is_random_like = run_kind == "online_random_like"
    is_custom_yolo_like = run_kind == "custom_yolo_like_base"
    if is_custom_yolo_like:
        title = "Custom YOLO-Like Base 50 Epoch Report"
    elif is_random_like:
        title = "Online Random-Like 50 Epoch Report"
    else:
        title = "Online Diag Policy 001 50 Epoch Report"
    lines = [
        f"# {title}",
        "",
        f"- Run ID: `{metrics_payload['run_id']}`",
        f"- Mode: `{metrics_payload['mode']}`",
        f"- Train images: `{stats.get('train_image_count')}`",
        f"- Train images remain 2301: `{str(stats.get('train_image_count_matches_original')).lower()}`",
        f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
        "- Validation custom augmentation: `false`",
            f"- YOLO built-in augmentation disabled: `{str(metrics_payload['online_mechanism'].get('yolo_builtin_augmentations_disabled')).lower()}`",
            f"- Copy-paste online supported: `{str(metrics_payload['online_mechanism'].get('copy_paste_online_supported')).lower()}`",
            f"- Feedback applied: `{str(stats.get('feedback_enabled', False)).lower()}`",
            f"- Policy history: `{metrics_payload['artifacts'].get('policy_history_json')}`",
            "",
            "## Final Metrics",
        "",
        "| Precision | Recall | mAP50 | mAP50-95 |",
        "|---:|---:|---:|---:|",
        (
            f"| {fmt(final_metrics.get('precision'))} | {fmt(final_metrics.get('recall'))} | "
            f"{fmt(final_metrics.get('map50'))} | {fmt(final_metrics.get('map50_95'))} |"
        ),
        "",
        "## Online Operation Counts",
        "",
        "| op | seen | applied | skipped_probability | skipped_safety | skipped_copy_paste_pending | skipped_unsupported |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, counts in sorted(stats.get("ops", {}).items()):
        lines.append(
            f"| {name} | {counts.get('seen', 0)} | {counts.get('applied', 0)} | "
            f"{counts.get('skipped_probability', 0)} | {counts.get('skipped_safety', 0)} | "
            f"{counts.get('skipped_copy_paste_pending', 0)} | {counts.get('skipped_unsupported', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- Invalid bbox count: `{stats.get('invalid_bbox_count', 0)}`",
            f"- Bbox out-of-bounds count before clipping: `{stats.get('bbox_oob_count', 0)}`",
            f"- Class id out-of-range count: `{stats.get('class_id_oob_count', 0)}`",
            f"- Cutout holes applied: `{stats.get('cutout_safe', {}).get('holes_applied', 0)}`",
            f"- Cutout skipped by center safety: `{stats.get('cutout_safe', {}).get('holes_skipped_center', 0)}`",
            f"- Cutout skipped by overlap safety: `{stats.get('cutout_safe', {}).get('holes_skipped_overlap', 0)}`",
            f"- Cutout skipped by no safe region: `{stats.get('cutout_safe', {}).get('holes_skipped_no_safe_region', 0)}`",
            f"- Mosaic applied: `{stats.get('mosaic4', {}).get('applied', 0)}`",
            f"- Mosaic skipped by close_mosaic: `{stats.get('mosaic4', {}).get('skipped_close_mosaic', stats.get('ops', {}).get('mosaic4', {}).get('skipped_close_mosaic', 0))}`",
            f"- close_mosaic active: `{str(stats.get('ops', {}).get('mosaic4', {}).get('skipped_close_mosaic', 0) > 0).lower()}`",
            "",
            "## Per-Class Recall/AP50",
            "",
            "| class_id | name | instances | Precision | Recall | AP50 | AP50-95 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics_payload.get("final_per_class", []):
        lines.append(
            f"| {row.get('class_id')} | {row.get('name')} | {row.get('instances')} | "
            f"{fmt(row.get('precision'))} | {fmt(row.get('recall'))} | {fmt(row.get('ap50'))} | {fmt(row.get('ap50_95'))} |"
        )
    lines.extend(
        [
            "",
            "## Comparison Summary",
            "",
            comparison_table(comparison.get("summary_rows", [])),
            "",
            "## Key Answers",
            "",
        ]
    )
    if is_random_like:
        lines.extend(
            [
                answer_line("Online random-like better than offline random external", comparison.get("answers", {}).get("online_random_like_better_than_offline_random")),
                answer_line("Online random-like better than online DiagAug", comparison.get("answers", {}).get("online_random_like_better_than_online_diagaug")),
                answer_line("Online random-like close to YOLO default by mAP50-95 within 0.03", comparison.get("answers", {}).get("online_close_to_yolo_default")),
                answer_line("Online random-like is current best by mAP50-95", comparison.get("answers", {}).get("is_current_best")),
                f"- Random external advantage source: `{comparison.get('answers', {}).get('random_external_advantage_source', 'unknown')}`",
                "- Online random-like removes the fixed doubled dataset confound; if it approaches offline random, the operator mix is the likely driver.",
            ]
        )
    elif is_custom_yolo_like:
        lines.extend(
            [
                answer_line("Custom YOLO-like close to Ultralytics YOLO default by mAP50-95 within 0.03", comparison.get("answers", {}).get("custom_yolo_like_close_to_yolo_default")),
                answer_line("Custom YOLO-like better than YOLO default", comparison.get("answers", {}).get("custom_yolo_like_better_than_yolo_default")),
                answer_line("Custom YOLO-like better than baseline", comparison.get("answers", {}).get("custom_yolo_like_better_than_baseline")),
                answer_line("Custom YOLO-like better than Online DiagAug", comparison.get("answers", {}).get("custom_yolo_like_better_than_online_diagaug")),
                answer_line("Custom YOLO-like better than offline random external", comparison.get("answers", {}).get("custom_yolo_like_better_than_random_external")),
                "- This run isolates the custom implementation of YOLO-like online operators from Ultralytics built-in augmentation.",
            ]
        )
    else:
        lines.extend(
            [
                answer_line("Online DiagAug better than offline DiagAug", comparison.get("answers", {}).get("online_better_than_offline_diagaug")),
                answer_line("Online DiagAug close to YOLO default by mAP50-95 within 0.03", comparison.get("answers", {}).get("online_close_to_yolo_default")),
                answer_line("Online improves Precision and mAP50-95 vs offline DiagAug", comparison.get("answers", {}).get("online_improves_precision_map")),
                "- Online mechanism is more methodologically reasonable than fixed offline doubling because the train image count stays unchanged and policy randomness is sampled per epoch/sample in the dataloader.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- best.pt: `{metrics_payload['artifacts'].get('best_pt')}`",
            f"- last.pt: `{metrics_payload['artifacts'].get('last_pt')}`",
            f"- Stats JSON: `{metrics_payload['artifacts'].get('online_aug_stats')}`",
            f"- Preview dir: `{metrics_payload['artifacts'].get('preview_dir')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_comparison_report(metrics_payload: dict[str, Any], comparison: dict[str, Any]) -> str:
    is_random_like = metrics_payload.get("run_kind") == "online_random_like"
    is_custom_yolo_like = metrics_payload.get("run_kind") == "custom_yolo_like_base"
    if is_custom_yolo_like:
        title = "Custom YOLO-Like Base vs YOLO Default Comparison"
        delta_title = "Deltas For Custom YOLO-Like Base"
    elif is_random_like:
        title = "Online Random-Like vs All Comparison"
        delta_title = "Deltas For Online Random-Like"
    else:
        title = "Online vs Offline/Yolo Default/Random Comparison"
        delta_title = "Deltas For Online DiagAug"
    lines = [
        f"# {title}",
        "",
        comparison_table(comparison.get("summary_rows", [])),
        "",
        f"## {delta_title}",
        "",
        "| reference | dP | dR | d_mAP50 | d_mAP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "vs_baseline": "Baseline no aug",
        "vs_offline_diag_policy_001": "Offline DiagAug",
        "vs_yolo_default": "YOLO default",
        "vs_random_external": "Random external",
        "vs_online_diag_policy_001": "Online DiagAug",
    }
    for key, label in labels.items():
        delta = comparison.get(key, {})
        if all(delta.get(metric) is None for metric in ["precision", "recall", "map50", "map50_95"]):
            continue
        lines.append(
            f"| {label} | {fmt(delta.get('precision'), signed=True)} | {fmt(delta.get('recall'), signed=True)} | "
            f"{fmt(delta.get('map50'), signed=True)} | {fmt(delta.get('map50_95'), signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if is_random_like:
        lines.extend(
            [
                answer_line("Online random-like better than offline random external", comparison.get("answers", {}).get("online_random_like_better_than_offline_random")),
                answer_line("Online random-like better than online DiagAug", comparison.get("answers", {}).get("online_random_like_better_than_online_diagaug")),
                answer_line("Online random-like is current best by mAP50-95", comparison.get("answers", {}).get("is_current_best")),
                f"- Random external advantage source: `{comparison.get('answers', {}).get('random_external_advantage_source', 'unknown')}`",
                "- This run isolates the random-like operator mix from offline train-set doubling.",
            ]
        )
    elif is_custom_yolo_like:
        lines.extend(
            [
                answer_line("Custom YOLO-like close to Ultralytics YOLO default by mAP50-95 within 0.03", comparison.get("answers", {}).get("custom_yolo_like_close_to_yolo_default")),
                answer_line("Custom YOLO-like better than YOLO default", comparison.get("answers", {}).get("custom_yolo_like_better_than_yolo_default")),
                answer_line("Custom YOLO-like better than baseline", comparison.get("answers", {}).get("custom_yolo_like_better_than_baseline")),
                answer_line("Custom YOLO-like better than Online DiagAug", comparison.get("answers", {}).get("custom_yolo_like_better_than_online_diagaug")),
                answer_line("Custom YOLO-like better than offline random external", comparison.get("answers", {}).get("custom_yolo_like_better_than_random_external")),
                "- This run evaluates whether the custom YOLO-like operator pool can stand in for Ultralytics default augmentation while keeping augmentation auditable.",
            ]
        )
    else:
        lines.extend(
            [
                answer_line("Online DiagAug better than offline DiagAug", comparison.get("answers", {}).get("online_better_than_offline_diagaug")),
                answer_line("Online DiagAug close to YOLO default by mAP50-95 within 0.03", comparison.get("answers", {}).get("online_close_to_yolo_default")),
                answer_line("Online improves Precision and mAP50-95 vs offline DiagAug", comparison.get("answers", {}).get("online_improves_precision_map")),
                "- Online DiagAug removes the fixed doubled dataset confound and is the fairer mechanism to compare against YOLO default online augmentation.",
            ]
        )
    lines.extend(
        [
            "",
            "## Source",
            "",
            f"- Online metrics JSON: `{metrics_payload['artifacts'].get('metrics_json')}`",
            f"- Online augmentation stats JSON: `{metrics_payload['artifacts'].get('online_aug_stats')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def comparison_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('name')} | {fmt(row.get('precision'))} | {fmt(row.get('recall'))} | "
            f"{fmt(row.get('map50'))} | {fmt(row.get('map50_95'))} |"
        )
    return "\n".join(lines)


def answer_line(label: str, value: bool | None) -> str:
    if value is None:
        text = "unknown"
    else:
        text = "yes" if value else "no"
    return f"- {label}: `{text}`"


def fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if signed:
        return f"{number:+.4f}"
    return f"{number:.4f}"


def update_state_docs(payload: dict[str, Any]) -> None:
    stats = payload["online_aug_stats"]
    val = payload["val"]["metrics"]
    if payload.get("feedback", {}).get("enabled"):
        feedback = payload["feedback"]
        section = "\n".join(
            [
                "## Feedback Online Augmentation Smoke",
                "",
                f"- Run ID: `{payload['run_id']}`",
                "- Entrypoint: `scripts/train_yolo_online_aug.py` with `--feedback-enabled`.",
                "- Mechanism: custom YOLO-like/industrial online augmentation remains inside the training dataloader; no fixed augmented dataset is built.",
                f"- Stage count: `{feedback.get('stage_count')}`",
                f"- Policy history updates: `{feedback.get('policy_history_count')}`",
                f"- Train image count: `{stats.get('train_image_count')}`; no train image doubling.",
                f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
                "- Validation custom augmentation: `false`; val uses original val tiles.",
                "- YOLO built-in augmentation: disabled for `only_custom_online_aug`.",
                "- Online copy-paste: pending object-bank implementation; feedback may raise pending copy-paste probabilities but execution is skipped safely.",
                f"- Train success: `{str(payload['train']['success']).lower()}`",
                f"- Val success: `{str(payload['val']['success']).lower()}`",
                f"- Val P/R/mAP50/mAP50-95: `{val.get('precision', 0.0):.4f}/{val.get('recall', 0.0):.4f}/{val.get('map50', 0.0):.4f}/{val.get('map50_95', 0.0):.4f}`",
                f"- Report: `outputs/experiments/{payload['run_id']}/reports/online_aug_smoke_report.md`",
                f"- Stats JSON: `outputs/experiments/{payload['run_id']}/reports/online_aug_stats.json`",
                f"- Policy history JSON: `outputs/experiments/{payload['run_id']}/reports/policy_history.json`",
            ]
        )
        for rel in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
            upsert_section(PROJECT_ROOT / rel, "FEEDBACK_ONLINE_AUG_SMOKE", section)
        return
    if "formal_metrics" in payload:
        metrics_payload = payload["formal_metrics"]
        comparison = metrics_payload.get("comparison", {})
        if metrics_payload.get("run_kind") == "custom_yolo_like_base":
            section = "\n".join(
                [
                    "## Custom YOLO-Like Base 50 Epoch",
                    "",
                    f"- Run ID: `{payload['run_id']}`",
                    "- Entrypoint: `scripts/train_yolo_online_aug.py`.",
                    "- Policy: `configs/online_policies/yolo_like_base_policy.json`.",
                    "- Mechanism: custom YOLO-like operators are sampled online in the YOLO training dataloader; no fixed augmented dataset is built.",
                    f"- Train image count: `{stats.get('train_image_count')}`; no train image doubling.",
                    f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
                    "- Validation custom augmentation: `false`; val uses original val tiles.",
                    "- YOLO built-in augmentation: disabled for `only_custom_online_aug`.",
                    f"- Feedback applied: `{str(stats.get('feedback_enabled', False)).lower()}`; policy history records no feedback applied.",
                    f"- close_mosaic active: `{str(stats.get('ops', {}).get('mosaic4', {}).get('skipped_close_mosaic', 0) > 0).lower()}`",
                    f"- Train success: `{str(payload['train']['success']).lower()}`",
                    f"- Val success: `{str(payload['val']['success']).lower()}`",
                    f"- Val P/R/mAP50/mAP50-95: `{val.get('precision', 0.0):.4f}/{val.get('recall', 0.0):.4f}/{val.get('map50', 0.0):.4f}/{val.get('map50_95', 0.0):.4f}`",
                    f"- Close to YOLO default by mAP50-95 within 0.03: `{str(comparison.get('answers', {}).get('custom_yolo_like_close_to_yolo_default')).lower()}`",
                    f"- Report: `outputs/experiments/{payload['run_id']}/reports/custom_yolo_like_base_50ep_report.md`",
                    f"- Metrics JSON: `outputs/experiments/{payload['run_id']}/reports/custom_yolo_like_base_50ep_metrics.json`",
                    f"- Comparison: `outputs/experiments/{payload['run_id']}/reports/compare_custom_yolo_like_with_yolo_default.md`",
                    f"- Stats JSON: `outputs/experiments/{payload['run_id']}/reports/online_aug_stats.json`",
                    f"- Policy history JSON: `outputs/experiments/{payload['run_id']}/reports/policy_history.json`",
                ]
            )
            for rel in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
                upsert_section(PROJECT_ROOT / rel, "CUSTOM_YOLO_LIKE_BASE_50EP", section)
            return
        if metrics_payload.get("run_kind") == "online_random_like":
            section = "\n".join(
                [
                    "## Online Random-Like 50 Epoch",
                    "",
                    f"- Run ID: `{payload['run_id']}`",
                    "- Entrypoint: `scripts/train_yolo_online_aug.py`.",
                    "- Policy: online random-like mix of `sharpen_mild`, `brightness`, `cutout_safe`, and `horizontal_flip`.",
                    "- Mechanism: policy is sampled online in the YOLO training dataloader; no fixed augmented dataset is built.",
                    f"- Train image count: `{stats.get('train_image_count')}`; no train image doubling.",
                    f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
                    "- Validation custom augmentation: `false`; val uses original val tiles.",
                    "- YOLO built-in augmentation: disabled for `only_custom_online_aug`.",
                    "- Online copy-paste: disabled.",
                    f"- Train success: `{str(payload['train']['success']).lower()}`",
                    f"- Val success: `{str(payload['val']['success']).lower()}`",
                    f"- Val P/R/mAP50/mAP50-95: `{val.get('precision', 0.0):.4f}/{val.get('recall', 0.0):.4f}/{val.get('map50', 0.0):.4f}/{val.get('map50_95', 0.0):.4f}`",
                    f"- Online random-like better than offline random by mAP50-95: `{str(comparison.get('answers', {}).get('online_random_like_better_than_offline_random')).lower()}`",
                    f"- Online random-like better than online DiagAug by mAP50-95: `{str(comparison.get('answers', {}).get('online_random_like_better_than_online_diagaug')).lower()}`",
                    f"- Random external advantage source: `{comparison.get('answers', {}).get('random_external_advantage_source', 'unknown')}`",
                    f"- Report: `outputs/experiments/{payload['run_id']}/reports/online_random_like_50ep_report.md`",
                    f"- Metrics JSON: `outputs/experiments/{payload['run_id']}/reports/online_random_like_metrics.json`",
                    f"- Comparison: `outputs/experiments/{payload['run_id']}/reports/compare_online_random_like_with_all.md`",
                    f"- Stats JSON: `outputs/experiments/{payload['run_id']}/reports/online_aug_stats.json`",
                ]
            )
            for rel in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
                upsert_section(PROJECT_ROOT / rel, "ONLINE_RANDOM_LIKE_50EP", section)
            return

        section = "\n".join(
            [
                "## Online Diag Policy 001 50 Epoch",
                "",
                f"- Run ID: `{payload['run_id']}`",
                "- Entrypoint: `scripts/train_yolo_online_aug.py`.",
                "- Mechanism: custom policy is sampled online in the YOLO training dataloader; no fixed augmented dataset is built.",
                f"- Train image count: `{stats.get('train_image_count')}`; no train image doubling.",
                f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
                "- Validation custom augmentation: `false`; val uses original val tiles.",
                "- YOLO built-in augmentation: disabled for `only_custom_online_aug`.",
                "- Online copy-paste: pending object-bank implementation; copy_paste ops are skipped safely.",
                f"- Train success: `{str(payload['train']['success']).lower()}`",
                f"- Val success: `{str(payload['val']['success']).lower()}`",
                f"- Val P/R/mAP50/mAP50-95: `{val.get('precision', 0.0):.4f}/{val.get('recall', 0.0):.4f}/{val.get('map50', 0.0):.4f}/{val.get('map50_95', 0.0):.4f}`",
                f"- Online better than offline DiagAug by mAP50-95: `{str(comparison.get('answers', {}).get('online_better_than_offline_diagaug')).lower()}`",
                f"- Online close to YOLO default by mAP50-95 within 0.03: `{str(comparison.get('answers', {}).get('online_close_to_yolo_default')).lower()}`",
                f"- Report: `outputs/experiments/{payload['run_id']}/reports/online_diag_policy_001_50ep_report.md`",
                f"- Metrics JSON: `outputs/experiments/{payload['run_id']}/reports/online_diag_policy_001_50ep_metrics.json`",
                f"- Comparison: `outputs/experiments/{payload['run_id']}/reports/compare_online_offline_yolo_default_random.md`",
                f"- Stats JSON: `outputs/experiments/{payload['run_id']}/reports/online_aug_stats.json`",
            ]
        )
        for rel in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
            upsert_section(PROJECT_ROOT / rel, "ONLINE_DIAG_POLICY_001_50EP", section)
        return

    section = "\n".join(
        [
            "## Online Policy Augmentation Smoke",
            "",
            f"- Run ID: `{payload['run_id']}`",
            "- New entrypoint: `scripts/train_yolo_online_aug.py`.",
            "- Method change: custom policy is applied dynamically inside the YOLO training dataloader instead of building a fixed offline augmented dataset.",
            f"- Train image count: `{stats.get('train_image_count')}`; no train image doubling.",
            f"- Fixed augmented dataset generated: `{str(stats.get('fixed_augmented_dataset_generated')).lower()}`",
            "- YOLO built-in augmentation mode for this smoke: disabled, so this is `only_custom_online_aug`.",
            "- Online copy-paste: pending object-bank implementation; copy_paste ops are skipped safely for now.",
            f"- 1 epoch smoke train success: `{str(payload['train']['success']).lower()}`",
            f"- 1 epoch smoke val success: `{str(payload['val']['success']).lower()}`",
            f"- Val P/R/mAP50/mAP50-95: `{val.get('precision', 0.0):.4f}/{val.get('recall', 0.0):.4f}/{val.get('map50', 0.0):.4f}/{val.get('map50_95', 0.0):.4f}`",
            f"- Preview dir: `{payload['artifacts']['preview_dir']}`",
            f"- Report: `outputs/experiments/{payload['run_id']}/reports/online_aug_smoke_report.md`",
            f"- Stats JSON: `outputs/experiments/{payload['run_id']}/reports/online_aug_stats.json`",
            "- Next step: inspect smoke safety/history and tune the feedback controller before any formal 50 epoch experiment.",
        ]
    )
    for rel in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
        upsert_section(PROJECT_ROOT / rel, "ONLINE_AUG_SMOKE", section)


def export_project_snapshot() -> None:
    script = PROJECT_ROOT / "scripts" / "export_project_snapshot.py"
    if script.exists():
        subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), check=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    write_text(path, text)


def upsert_section(path: Path, key: str, section: str) -> None:
    start = f"<!-- {key}_START -->"
    end = f"<!-- {key}_END -->"
    text = path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""
    block = f"{start}\n{section.rstrip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _match: block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
