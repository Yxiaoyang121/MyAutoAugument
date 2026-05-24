from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
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

    stats = OnlineAugmentationStats()
    augmentor = OnlinePolicyAugmentor(
        policy,
        seed=args.seed,
        copy_paste_enabled=args.online_copy_paste,
        stats=stats,
    )
    context = OnlineTrainingContext(
        augmentor=augmentor,
        preview_dir=output_dir / "previews",
        save_preview=args.save_preview,
        preview_count=args.preview_count,
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
    write_json(output_dir / "reports" / "online_aug_stats.json", payload["online_aug_stats"])
    write_markdown(output_dir / "reports" / "online_aug_smoke_report.md", build_smoke_report(payload))
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

        def build_transforms(self, hyp: dict | None = None):
            transforms = super().build_transforms(hyp)
            if self.augment and self.online_context is not None:
                transforms.insert(0, UltralyticsOnlinePolicyTransform(self.online_context, Instances))
            return transforms

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


class UltralyticsOnlinePolicyTransform:
    def __init__(self, context: OnlineTrainingContext, instances_cls: Any) -> None:
        self.context = context
        self.instances_cls = instances_cls

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
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


def build_train_kwargs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(Path(args.data)),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": str(args.device),
        "seed": args.seed,
        "project": str(output_dir),
        "name": "train",
        "exist_ok": True,
        "plots": False,
    }
    if args.disable_yolo_aug:
        kwargs.update(DISABLED_YOLO_AUG_ARGS)
    return kwargs


def build_train_command(args: argparse.Namespace, output_dir: Path) -> str:
    parts = [
        "YOLO.train",
        f"model={args.model}",
        f"data={Path(args.data)}",
        f"policy={Path(args.policy)}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"project={output_dir}",
        "name=train",
        "trainer=OnlineAugDetectionTrainer",
    ]
    if args.disable_yolo_aug:
        parts.extend(f"{key}={value}" for key, value in DISABLED_YOLO_AUG_ARGS.items())
    return subprocess.list2cmdline([str(part) for part in parts])


def build_val_command(args: argparse.Namespace, output_dir: Path, weights: Path) -> str:
    parts = [
        "YOLO.val",
        f"model={weights}",
        f"data={Path(args.data)}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"device={args.device}",
        f"project={output_dir}",
        "name=val",
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
        "mode": "only_custom_online_aug" if args.disable_yolo_aug else "custom_online_plus_yolo_default_reserved",
    }


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
    return {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
    }


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
        "policy": policy,
        "mode": "only_custom_online_aug" if args.disable_yolo_aug else "custom_online_plus_yolo_default_reserved",
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
            "",
            "## Copy-Paste",
            "",
            "- Online copy-paste pending: object-bank paste is not enabled in this first smoke implementation.",
            "- Current online policy verifies photometric / texture / cutout / flip / mild geometry operations.",
        ]
    )
    if payload["train"]["error"]:
        lines.extend(["", "## Train Error", "", f"`{payload['train']['error']}`"])
    if payload["val"]["error"]:
        lines.extend(["", "## Validation Error", "", f"`{payload['val']['error']}`"])
    return "\n".join(lines) + "\n"


def update_state_docs(payload: dict[str, Any]) -> None:
    stats = payload["online_aug_stats"]
    val = payload["val"]["metrics"]
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
            "- Next step after smoke success: run online `diag_policy_001` for 50 epochs and compare fairly against YOLO default augmentation.",
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
