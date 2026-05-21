from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostic_pipeline import run_error_diagnosis, run_validation_prediction  # noqa: E402


RUN_ID = "20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain"
POLICY_PATH = PROJECT_ROOT / "outputs" / "experiments" / "20260518_tiled1024_safe_no_ok_position_per_class_diagnosis" / "policies" / "class_aware_mixed_policy.json"
DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
DATA_YAML = DATASET_ROOT / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID
DIAG_POLICY_001_RESULTS = PROJECT_ROOT / "outputs" / "experiments" / "20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain" / "reports" / "top3_policy_shorttrain_results.json"

YOLO = Path("D:/Anaconda/envs/pytorch/Scripts/yolo.exe")
MODEL = "yolo11n.pt"
EPOCHS = 5
IMGSZ = 1024
DEFAULT_BATCH = 2
WORKERS = 0
DEVICE = "0"
SEED = 42
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DISABLED_YOLO_AUGS = {
    "mosaic": 0,
    "mixup": 0,
    "copy_paste": 0,
    "hsv_h": 0,
    "hsv_s": 0,
    "hsv_v": 0,
    "degrees": 0,
    "translate": 0,
    "scale": 0,
    "shear": 0,
    "perspective": 0,
    "fliplr": 0,
    "flipud": 0,
}
CLASS_NAMES = {
    0: "OK2",
    1: "OK3",
    2: "加强筋打伤",
    3: "开裂",
    4: "油污",
    5: "浅划伤",
    6: "漏背锡",
    7: "碰伤",
    8: "脏污",
    9: "轮廓划伤",
    10: "锡丝残留",
    11: "锡尖",
    12: "锡膏",
}


@dataclass(frozen=True)
class YoloRecord:
    image_path: Path
    label_path: Path
    relative_path: Path


def main() -> None:
    configure_environment()
    prepare_output_dir(OUTPUT_DIR)
    policy = read_json(POLICY_PATH)
    diag_policy = load_diag_policy_001()
    train_records = find_records(DATASET_ROOT / "images" / "train", DATASET_ROOT / "labels" / "train")
    val_records = find_records(DATASET_ROOT / "images" / "val", DATASET_ROOT / "labels" / "val")
    write_json(
        OUTPUT_DIR / "run_config.json",
        {
            "run_id": RUN_ID,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "scope": "top1 class-aware policy short-training only; no final 50 epoch training",
            "policy_path": rel(POLICY_PATH),
            "data_yaml": rel(DATA_YAML),
            "model": MODEL,
            "train_settings": {
                "epochs": EPOCHS,
                "imgsz": IMGSZ,
                "batch": DEFAULT_BATCH,
                "workers": WORKERS,
                "device": DEVICE,
                "seed": SEED,
                "disabled_yolo_augmentations": DISABLED_YOLO_AUGS,
            },
            "diag_policy_001_shorttrain_reference": {
                "precision": diag_policy["metrics"]["precision"],
                "recall": diag_policy["metrics"]["recall"],
                "map50": diag_policy["metrics"]["map50"],
                "map50_95": diag_policy["metrics"]["map50_95"],
                "short_train_score": diag_policy["metrics"]["short_train_score"],
            },
        },
    )

    print("[class-aware] building augmented dataset")
    dataset_summary = build_class_aware_dataset(policy=policy, train_records=train_records, val_records=val_records)
    write_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)

    batch = DEFAULT_BATCH
    oom = False
    print("[class-aware] training 5 epochs")
    train_result = run_train(dataset_summary["data_yaml"], batch=batch)
    if train_result["returncode"] != 0 and has_oom(OUTPUT_DIR / "train_stdout.log", OUTPUT_DIR / "train_stderr.log"):
        oom = True
        batch = 1
        print("[class-aware] OOM at batch=2; retrying batch=1")
        train_result = run_train(dataset_summary["data_yaml"], batch=batch)
        if train_result["returncode"] != 0 and has_oom(OUTPUT_DIR / "train_stdout.log", OUTPUT_DIR / "train_stderr.log"):
            raise RuntimeError("class-aware policy still OOM at batch=1")
    if train_result["returncode"] != 0:
        raise RuntimeError(f"class-aware short training failed; see {OUTPUT_DIR / 'train_stderr.log'}")

    copy_weights()
    print("[class-aware] validating best.pt")
    val_result = run_val(dataset_summary["data_yaml"], batch=batch)
    if val_result["returncode"] != 0 and has_oom(OUTPUT_DIR / "val_stdout.log", OUTPUT_DIR / "val_stderr.log"):
        oom = True
        batch = 1
        val_result = run_val(dataset_summary["data_yaml"], batch=batch)
    if val_result["returncode"] != 0:
        raise RuntimeError(f"class-aware validation failed; see {OUTPUT_DIR / 'val_stderr.log'}")

    val_metrics = parse_yolo_val_log(OUTPUT_DIR / "val_stdout.log")
    small_object = compute_small_object_recall(dataset_summary, batch=batch)
    score = compute_short_train_score(val_metrics["overall"], small_object)
    train_loop = read_training_summary(OUTPUT_DIR / "train" / "results.csv")
    comparison = compare_to_diag_policy_001(val_metrics, score, diag_policy)
    payload = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy_id": policy["policy_id"],
        "policy": policy,
        "dataset": dataset_summary,
        "train_settings": {
            "model": MODEL,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": batch,
            "workers": WORKERS,
            "device": DEVICE,
            "seed": SEED,
            "disabled_yolo_augmentations": DISABLED_YOLO_AUGS,
        },
        "commands": {
            "train_command": train_result["command"],
            "val_command": val_result["command"],
            "train_returncode": train_result["returncode"],
            "val_returncode": val_result["returncode"],
        },
        "artifacts": {
            "output_dir": str(OUTPUT_DIR.resolve()),
            "data_yaml": dataset_summary["data_yaml"],
            "best_pt": str((OUTPUT_DIR / "weights" / "best.pt").resolve()),
            "last_pt": str((OUTPUT_DIR / "weights" / "last.pt").resolve()),
            "train_stdout": str((OUTPUT_DIR / "train_stdout.log").resolve()),
            "train_stderr": str((OUTPUT_DIR / "train_stderr.log").resolve()),
            "val_stdout": str((OUTPUT_DIR / "val_stdout.log").resolve()),
            "val_stderr": str((OUTPUT_DIR / "val_stderr.log").resolve()),
        },
        "metrics": {
            **val_metrics["overall"],
            "per_class": val_metrics["per_class"],
            "small_object_recall": small_object.get("small_object_recall"),
            "small_object_recall_basis": small_object,
            "short_train_score": score["score"],
            "short_train_score_formula": score["formula"],
        },
        "diag_policy_001_reference": diag_policy,
        "comparison_to_diag_policy_001": comparison,
        "last_epoch_metrics_from_training_loop": train_loop.get("last_epoch", {}),
        "best_epoch_by_map50_95_from_training_loop": train_loop.get("best_epoch_by_map50_95", {}),
        "oom": oom,
        "training_wall_seconds": train_result["wall_seconds"],
        "validation_wall_seconds": val_result["wall_seconds"],
        "recommend_formal_50epoch": comparison["class_aware_beats_diag_policy_001"],
    }
    write_json(OUTPUT_DIR / "metrics" / "class_aware_shorttrain_metrics.json", payload)
    write_class_aware_report(OUTPUT_DIR / "reports" / "class_aware_shorttrain_report.md", payload)
    write_comparison_report(OUTPUT_DIR / "reports" / "class_aware_vs_diag_policy_001_shorttrain.md", payload)
    update_state_docs(payload)
    print(json.dumps(payload["comparison_to_diag_policy_001"]["summary"], ensure_ascii=False, indent=2))


def configure_environment() -> None:
    if YOLO.parent.exists():
        os.environ["PATH"] = str(YOLO.parent) + os.pathsep + os.environ.get("PATH", "")
    yolo_config = PROJECT_ROOT / "outputs" / "Ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def prepare_output_dir(path: Path) -> None:
    resolved = path.resolve()
    experiments_root = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    if resolved.exists():
        if not str(resolved).startswith(str(experiments_root)):
            raise RuntimeError(f"refusing to remove output outside experiments: {resolved}")
        shutil.rmtree(resolved)
    (resolved / "reports").mkdir(parents=True, exist_ok=True)
    (resolved / "metrics").mkdir(parents=True, exist_ok=True)


def build_class_aware_dataset(*, policy: dict[str, Any], train_records: list[YoloRecord], val_records: list[YoloRecord]) -> dict[str, Any]:
    dataset_dir = OUTPUT_DIR / "dataset" / "final_dataset"
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_val = dataset_dir / "labels" / "val"
    for directory in [images_train, labels_train, images_val, labels_val]:
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    branch_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    augmented_modified = 0
    augmented_unmodified = 0
    original_bbox_count = 0
    augmented_bbox_count = 0
    invalid_bbox_count = 0
    copy_paste_new_boxes = 0

    for record in train_records:
        image = read_image(record.image_path)
        labels, boxes = load_labels(record.label_path, image.shape[1], image.shape[0])
        original_bbox_count += len(labels)
        stem = flatten_stem(record.relative_path)
        ext = record.image_path.suffix.lower() or ".jpg"
        save_sample(images_train / f"orig_{stem}{ext}", labels_train / f"orig_{stem}.txt", image, labels, boxes)

        aug_image, aug_labels, aug_boxes, audit = apply_class_aware_policy(image, labels, boxes, policy, rng)
        clipped_boxes, clipped_labels, invalid = clip_filter_boxes(aug_boxes, aug_labels, aug_image.shape[1], aug_image.shape[0])
        invalid_bbox_count += invalid
        augmented_bbox_count += len(clipped_labels)
        branch_counts.update(audit["branches_applied"])
        op_counts.update(audit["ops_applied"])
        copy_paste_new_boxes += int(audit["copy_paste_new_boxes"])
        if audit["modified"]:
            augmented_modified += 1
        else:
            augmented_unmodified += 1
        save_sample(images_train / f"aug_{stem}{ext}", labels_train / f"aug_{stem}.txt", aug_image, clipped_labels, clipped_boxes)

    for record in val_records:
        image = read_image(record.image_path)
        labels, boxes = load_labels(record.label_path, image.shape[1], image.shape[0])
        save_sample(images_val / record.relative_path, labels_val / record.relative_path.with_suffix(".txt"), image, labels, boxes)

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(CLASS_NAMES)}",
                "names:",
                *[f"- {CLASS_NAMES[index]}" for index in range(len(CLASS_NAMES))],
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = {
        "stage": "class_aware_augmented_dataset_builder",
        "status": "completed",
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "source_data_yaml": rel(DATA_YAML),
        "policy_id": policy["policy_id"],
        "original_train_count": len(train_records),
        "augmented_train_count": len(train_records),
        "total_train_images": count_images(images_train),
        "val_count": len(val_records),
        "train_images": count_images(images_train),
        "train_bboxes": count_label_rows(labels_train),
        "val_images": count_images(images_val),
        "val_bboxes": count_label_rows(labels_val),
        "original_train_bboxes": original_bbox_count,
        "augmented_train_bboxes": augmented_bbox_count,
        "copy_paste_new_boxes": copy_paste_new_boxes,
        "invalid_bbox_count": invalid_bbox_count,
        "augmented_modified_images": augmented_modified,
        "augmented_unmodified_images": augmented_unmodified,
        "branch_application_counts": dict(branch_counts),
        "operation_application_counts": dict(op_counts),
        "images_train": str(images_train.resolve()),
        "labels_train": str(labels_train.resolve()),
        "images_val": str(images_val.resolve()),
        "labels_val": str(labels_val.resolve()),
    }
    write_json(OUTPUT_DIR / "dataset_build_report.json", report)
    return report


def apply_class_aware_policy(
    image: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    policy: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    out = image.copy()
    out_labels = labels.copy()
    out_boxes = boxes.astype(np.float32, copy=True)
    image_classes = {int(value) for value in labels.tolist()}
    branches_applied: list[str] = []
    ops_applied: list[str] = []
    modified = False
    copy_paste_new_boxes = 0
    for branch in policy.get("branches", []):
        target_ids = {int(value) for value in branch.get("target_class_ids", [])}
        if not image_classes.intersection(target_ids):
            continue
        branch_touched = False
        for op in branch.get("ops", []):
            prob = float(op.get("prob", 0.0) or 0.0)
            if rng.random() > prob:
                continue
            before_count = len(out_labels)
            out, out_labels, out_boxes, changed = apply_operation(out, out_labels, out_boxes, op, target_ids, rng)
            if changed:
                modified = True
                branch_touched = True
                ops_applied.append(str(op.get("name")))
                if str(op.get("name")) == "class_balanced_copy_paste":
                    copy_paste_new_boxes += max(0, len(out_labels) - before_count)
        if branch_touched:
            branches_applied.append(str(branch.get("branch_name")))
    return out, out_labels, out_boxes, {
        "modified": modified,
        "branches_applied": branches_applied,
        "ops_applied": ops_applied,
        "copy_paste_new_boxes": copy_paste_new_boxes,
    }


def apply_operation(
    image: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    op: dict[str, Any],
    target_ids: set[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    name = str(op.get("name", "")).lower()
    params = op.get("params", {}) or {}
    strength = float(op.get("strength", 0.0) or 0.0)
    if name == "clahe":
        clip_low, clip_high = params.get("clip_limit_range", [2.0, 3.0])
        clip_limit = lerp(float(clip_low), float(clip_high), strength)
        return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=int(params.get("tile_grid_size", 8))), labels, boxes, True
    if name == "contrast":
        low, high = params.get("alpha_range", [1.1, 1.3])
        alpha = lerp(float(low), float(high), strength)
        return adjust_contrast(image, alpha=alpha), labels, boxes, True
    if name == "gamma":
        low, high = params.get("gamma_range", [0.75, 0.9])
        gamma = lerp(float(high), float(low), strength)
        return adjust_gamma(image, gamma=gamma), labels, boxes, True
    if name == "brightness":
        low, high = params.get("beta_range", [10, 30])
        beta = lerp(float(low), float(high), strength)
        return adjust_brightness(image, beta=beta), labels, boxes, True
    if name in {"sharpen", "sharpen_mild"}:
        low, high = params.get("amount_range", [0.5, 0.9])
        amount = lerp(float(low), float(high), strength)
        return sharpen_mild(image, amount=amount, sigma=float(params.get("sigma", 1.0))), labels, boxes, True
    if name == "local_contrast":
        low, high = params.get("clip_limit_range", [1.5, 2.5])
        clip_limit = lerp(float(low), float(high), strength)
        return apply_clahe(image, clip_limit=clip_limit, tile_grid_size=8), labels, boxes, True
    if name == "mild_noise":
        low, high = params.get("std_range", [0.01, 0.03])
        std = lerp(float(low), float(high), strength) * 255.0
        noise = rng.normal(0.0, std, size=image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8), labels, boxes, True
    if name == "mild_translate":
        max_translate = float(params.get("max_translate", 0.04)) * max(0.25, strength)
        dx = float(rng.uniform(-max_translate, max_translate))
        dy = float(rng.uniform(-max_translate, max_translate))
        return translate_image_boxes(image, labels, boxes, dx=dx, dy=dy)
    if name == "mild_scale":
        low, high = params.get("scale_range", [0.92, 1.08])
        max_delta = max(abs(1.0 - float(low)), abs(float(high) - 1.0)) * max(0.25, strength)
        scale_value = float(rng.uniform(1.0 - max_delta, 1.0 + max_delta))
        return scale_image_boxes(image, labels, boxes, scale_value=scale_value)
    if name == "class_balanced_copy_paste":
        return class_balanced_copy_paste(image, labels, boxes, target_ids=target_ids, params=params, rng=rng)
    return image, labels, boxes, False


def apply_clahe(image: np.ndarray, *, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_contrast(image: np.ndarray, *, alpha: float) -> np.ndarray:
    mean = image.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    out = (image.astype(np.float32) - mean) * alpha + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_gamma(image: np.ndarray, *, gamma: float) -> np.ndarray:
    table = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def adjust_brightness(image: np.ndarray, *, beta: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def sharpen_mild(image: np.ndarray, *, amount: float, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(image.astype(np.float32), 1.0 + amount, blurred.astype(np.float32), -amount, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def translate_image_boxes(
    image: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    h, w = image.shape[:2]
    tx = dx * w
    ty = dy * h
    matrix = np.asarray([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    fill = tuple(float(x) for x in image.reshape(-1, image.shape[-1]).mean(axis=0)) if image.ndim == 3 else float(image.mean())
    out = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    new_boxes = boxes.astype(np.float32, copy=True)
    if len(new_boxes):
        new_boxes[:, [0, 2]] += tx
        new_boxes[:, [1, 3]] += ty
    clipped, clipped_labels, _ = clip_filter_boxes(new_boxes, labels, w, h)
    return out, clipped_labels, clipped, True


def scale_image_boxes(
    image: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    *,
    scale_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    matrix = cv2.getRotationMatrix2D((cx, cy), 0.0, scale_value).astype(np.float32)
    fill = tuple(float(x) for x in image.reshape(-1, image.shape[-1]).mean(axis=0)) if image.ndim == 3 else float(image.mean())
    out = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    new_boxes = transform_boxes_affine(boxes, matrix)
    clipped, clipped_labels, _ = clip_filter_boxes(new_boxes, labels, w, h)
    return out, clipped_labels, clipped, True


def class_balanced_copy_paste(
    image: np.ndarray,
    labels: np.ndarray,
    boxes: np.ndarray,
    *,
    target_ids: set[int],
    params: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    candidates = [index for index, label in enumerate(labels) if int(label) in target_ids]
    if not candidates:
        return image, labels, boxes, False
    h, w = image.shape[:2]
    out = image.copy()
    labels_out = labels.copy()
    boxes_out = boxes.astype(np.float32, copy=True)
    max_paste = max(1, int(params.get("max_paste_count", 2)))
    paste_count = int(rng.integers(1, max_paste + 1))
    max_overlap = float(params.get("max_overlap", 0.20))
    changed = False
    for _ in range(paste_count):
        source_index = int(rng.choice(candidates))
        source_box = boxes[source_index]
        x1, y1, x2, y2 = [int(round(float(v))) for v in source_box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        patch = image[y1:y2, x1:x2].copy()
        ph, pw = patch.shape[:2]
        if ph < 2 or pw < 2 or ph >= h or pw >= w:
            continue
        destination = find_paste_location(pw, ph, w, h, boxes_out, max_overlap=max_overlap, rng=rng)
        if destination is None:
            continue
        dx1, dy1, dx2, dy2 = [int(round(float(v))) for v in destination]
        out[dy1:dy2, dx1:dx2] = patch
        labels_out = np.concatenate([labels_out, np.asarray([labels[source_index]], dtype=np.int64)])
        boxes_out = np.vstack([boxes_out, destination.astype(np.float32)])
        changed = True
    return out, labels_out, boxes_out, changed


def find_paste_location(
    patch_w: int,
    patch_h: int,
    width: int,
    height: int,
    existing_boxes: np.ndarray,
    *,
    max_overlap: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    max_x = width - patch_w
    max_y = height - patch_h
    if max_x < 0 or max_y < 0:
        return None
    best: np.ndarray | None = None
    best_iou = float("inf")
    for _ in range(50):
        x1 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        y1 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        candidate = np.asarray([x1, y1, x1 + patch_w, y1 + patch_h], dtype=np.float32)
        overlap = float(np.max(bbox_iou(candidate.reshape(1, 4), existing_boxes))) if len(existing_boxes) else 0.0
        if overlap <= max_overlap:
            return candidate
        if overlap < best_iou:
            best_iou = overlap
            best = candidate
    if best is not None and best_iou <= 0.50:
        return best
    return None


def transform_boxes_affine(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return boxes.astype(np.float32).reshape(0, 4)
    corners = np.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        axis=1,
    )
    ones = np.ones((corners.shape[0], corners.shape[1], 1), dtype=np.float32)
    homo = np.concatenate([corners.astype(np.float32), ones], axis=2)
    transformed = homo @ matrix.T
    x_min = transformed[:, :, 0].min(axis=1)
    y_min = transformed[:, :, 1].min(axis=1)
    x_max = transformed[:, :, 0].max(axis=1)
    y_max = transformed[:, :, 1].max(axis=1)
    return np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.float32)


def clip_filter_boxes(boxes: np.ndarray, labels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, int]:
    if len(boxes) == 0:
        return boxes.astype(np.float32).reshape(0, 4), labels.astype(np.int64), 0
    clipped = boxes.astype(np.float32, copy=True)
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    keep = (clipped[:, 2] - clipped[:, 0] >= 2.0) & (clipped[:, 3] - clipped[:, 1] >= 2.0)
    invalid = int((~keep).sum())
    return clipped[keep], labels[keep].astype(np.int64), invalid


def run_train(data_yaml: str, *, batch: int) -> dict[str, Any]:
    command = [
        str(YOLO),
        "detect",
        "train",
        f"model={MODEL}",
        f"data={data_yaml}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"batch={batch}",
        f"workers={WORKERS}",
        f"device={DEVICE}",
        f"seed={SEED}",
        f"project={OUTPUT_DIR}",
        "name=train",
        "exist_ok=True",
        *[f"{key}={value}" for key, value in DISABLED_YOLO_AUGS.items()],
    ]
    return run_command(command, OUTPUT_DIR, "train")


def run_val(data_yaml: str, *, batch: int) -> dict[str, Any]:
    command = [
        str(YOLO),
        "detect",
        "val",
        f"model={OUTPUT_DIR / 'weights' / 'best.pt'}",
        f"data={data_yaml}",
        f"imgsz={IMGSZ}",
        f"batch={batch}",
        f"workers={WORKERS}",
        f"device={DEVICE}",
        f"project={OUTPUT_DIR}",
        "name=val",
        "exist_ok=True",
    ]
    return run_command(command, OUTPUT_DIR, "val")


def run_command(command: list[str], output_dir: Path, prefix: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_text = subprocess.list2cmdline(command)
    (output_dir / f"{prefix}_command.txt").write_text(command_text + "\n", encoding="utf-8")
    stdout_path = output_dir / f"{prefix}_stdout.log"
    stderr_path = output_dir / f"{prefix}_stderr.log"
    start = time.time()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, env=os.environ.copy())
        returncode = process.wait()
    return {"command": command_text, "returncode": returncode, "wall_seconds": time.time() - start}


def copy_weights() -> None:
    src = OUTPUT_DIR / "train" / "weights"
    dst = OUTPUT_DIR / "weights"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["best.pt", "last.pt"]:
        source = src / name
        if not source.exists():
            raise FileNotFoundError(f"missing trained weight: {source}")
        shutil.copy2(source, dst / name)


def compute_small_object_recall(dataset_summary: dict[str, Any], *, batch: int) -> dict[str, Any]:
    try:
        pred_record = run_validation_prediction(
            weights=OUTPUT_DIR / "weights" / "best.pt",
            val_images_dir=Path(dataset_summary["images_val"]),
            val_labels_dir=Path(dataset_summary["labels_val"]),
            output_dir=OUTPUT_DIR / "diagnosis_prediction",
            imgsz=IMGSZ,
            workers=WORKERS,
            device=DEVICE,
            conf=0.25,
            iou=0.5,
            dry_run=False,
        )
        diagnosis = run_error_diagnosis(
            val_images_dir=Path(dataset_summary["images_val"]),
            val_labels_dir=Path(dataset_summary["labels_val"]),
            predictions_dir=pred_record["predictions_dir"],
            output_dir=OUTPUT_DIR / "diagnosis",
            class_names=CLASS_NAMES,
            match_iou=0.5,
        )
        by_size = diagnosis.get("size_recall", {})
        tiny = by_size.get("tiny", {})
        small = by_size.get("small", {})
        gt = int(tiny.get("gt_count", 0) or 0) + int(small.get("gt_count", 0) or 0)
        tp = int(tiny.get("tp_count", 0) or 0) + int(small.get("tp_count", 0) or 0)
        return {
            "available": True,
            "formula_used": "preferred",
            "small_object_recall": tp / max(1, gt),
            "tiny_gt": int(tiny.get("gt_count", 0) or 0),
            "small_gt": int(small.get("gt_count", 0) or 0),
            "small_object_gt": gt,
            "small_object_tp": tp,
            "prediction_batch": batch,
        }
    except Exception as exc:
        return {"available": False, "formula_used": "fallback", "small_object_recall": None, "error": str(exc)}


def compute_short_train_score(overall: dict[str, Any], small_object: dict[str, Any]) -> dict[str, Any]:
    map50 = float(overall["map50"])
    map50_95 = float(overall["map50_95"])
    recall = float(overall["recall"])
    small_recall = small_object.get("small_object_recall")
    if small_object.get("available") and small_recall is not None:
        score = 0.30 * map50 + 0.35 * map50_95 + 0.20 * recall + 0.15 * float(small_recall)
        formula = "0.30*mAP50 + 0.35*mAP50-95 + 0.20*Recall + 0.15*small_object_recall"
    else:
        score = 0.35 * map50 + 0.40 * map50_95 + 0.25 * recall
        formula = "0.35*mAP50 + 0.40*mAP50-95 + 0.25*Recall"
    return {"score": float(score), "formula": formula}


def compare_to_diag_policy_001(val_metrics: dict[str, Any], score: dict[str, Any], diag_policy: dict[str, Any]) -> dict[str, Any]:
    ref_metrics = diag_policy["metrics"]
    ref_by_id = {int(row["class_id"]): row for row in ref_metrics["per_class"]}
    per_class = []
    for row in val_metrics["per_class"]:
        ref = ref_by_id[int(row["class_id"])]
        per_class.append(
            {
                "class_id": int(row["class_id"]),
                "class_name": CLASS_NAMES.get(int(row["class_id"]), str(row["class_id"])),
                "class_aware_recall": float(row["recall"]),
                "diag_policy_001_recall": float(ref["recall"]),
                "delta_recall": float(row["recall"]) - float(ref["recall"]),
                "class_aware_ap50": float(row["ap50"]),
                "diag_policy_001_ap50": float(ref["ap50"]),
                "delta_ap50": float(row["ap50"]) - float(ref["ap50"]),
            }
        )
    delta_score = score["score"] - float(ref_metrics["short_train_score"])
    return {
        "diag_policy_001_policy_id": "diag_policy_001",
        "diag_policy_001_metrics": {
            "precision": ref_metrics["precision"],
            "recall": ref_metrics["recall"],
            "map50": ref_metrics["map50"],
            "map50_95": ref_metrics["map50_95"],
            "short_train_score": ref_metrics["short_train_score"],
        },
        "delta": {
            "precision": val_metrics["overall"]["precision"] - ref_metrics["precision"],
            "recall": val_metrics["overall"]["recall"] - ref_metrics["recall"],
            "map50": val_metrics["overall"]["map50"] - ref_metrics["map50"],
            "map50_95": val_metrics["overall"]["map50_95"] - ref_metrics["map50_95"],
            "short_train_score": delta_score,
        },
        "per_class": per_class,
        "improved_classes": [row for row in per_class if row["delta_recall"] > 0 or row["delta_ap50"] > 0],
        "declined_classes": [row for row in per_class if row["delta_recall"] < 0 or row["delta_ap50"] < 0],
        "class_aware_beats_diag_policy_001": delta_score > 0,
        "summary": {
            "class_aware_short_train_score": score["score"],
            "diag_policy_001_short_train_score": ref_metrics["short_train_score"],
            "delta_short_train_score": delta_score,
            "class_aware_beats_diag_policy_001": delta_score > 0,
        },
    }


def load_diag_policy_001() -> dict[str, Any]:
    payload = read_json(DIAG_POLICY_001_RESULTS)
    for trial in payload["trials"]:
        if trial["policy_id"] == "diag_policy_001":
            return trial
    raise KeyError("diag_policy_001 not found in top3 short-training results")


def find_records(images_dir: Path, labels_dir: Path) -> list[YoloRecord]:
    records = []
    for image_path in sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        relative = image_path.relative_to(images_dir)
        records.append(YoloRecord(image_path=image_path, label_path=labels_dir / relative.with_suffix(".txt"), relative_path=relative))
    if not records:
        raise ValueError(f"no images found under {images_dir}")
    return records


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def load_labels(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    labels = []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        labels.append(int(float(parts[0])))
        x, y, w, h = [float(v) for v in parts[1:5]]
        boxes.append([(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])
    return np.asarray(labels, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def save_sample(image_path: Path, label_path: Path, image: np.ndarray, labels: np.ndarray, boxes: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise IOError(f"failed to write image: {image_path}")
    h, w = image.shape[:2]
    yolo = xyxy_to_yolo(boxes, w, h)
    lines = [
        f"{int(label)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
        for label, box in zip(labels, yolo)
    ]
    label_path.write_text("\n".join(lines), encoding="utf-8")


def xyxy_to_yolo(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.zeros_like(boxes, dtype=np.float32)
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2.0) / width
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2.0) / height
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
    return out


def bbox_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, eps)).astype(np.float32)


def parse_yolo_val_log(path: Path) -> dict[str, Any]:
    text = strip_ansi(path.read_text(encoding="utf-8", errors="replace"))
    rows: list[dict[str, Any]] = []
    overall: dict[str, Any] | None = None
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        values = numeric_tail(parts, 6)
        if values is None:
            continue
        images, instances, precision, recall, ap50, ap50_95 = values
        if parts[0] == "all":
            overall = {
                "images": int(images),
                "instances": int(instances),
                "precision": precision,
                "recall": recall,
                "map50": ap50,
                "map50_95": ap50_95,
            }
            continue
        if overall is None or len(rows) >= len(CLASS_NAMES):
            continue
        class_id = len(rows)
        rows.append(
            {
                "class_id": class_id,
                "name": CLASS_NAMES.get(class_id, str(class_id)),
                "log_name": parts[0],
                "images": int(images),
                "instances": int(instances),
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }
        )
    if overall is None:
        raise RuntimeError(f"could not parse YOLO val metrics from {path}")
    if len(rows) != len(CLASS_NAMES):
        raise RuntimeError(f"expected {len(CLASS_NAMES)} class rows from {path}, parsed {len(rows)}")
    return {"overall": overall, "per_class": rows}


def numeric_tail(parts: list[str], count: int) -> list[float] | None:
    values: list[float] = []
    for item in reversed(parts):
        try:
            values.append(float(item))
        except ValueError:
            continue
        if len(values) == count:
            return list(reversed(values))
    return None


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def read_training_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def metrics_from_row(row: dict[str, str]) -> dict[str, Any]:
        return {
            "epoch": int(float(row["epoch"])),
            "time_seconds": float(row["time"]),
            "precision": float(row["metrics/precision(B)"]),
            "recall": float(row["metrics/recall(B)"]),
            "map50": float(row["metrics/mAP50(B)"]),
            "map50_95": float(row["metrics/mAP50-95(B)"]),
        }

    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "last_epoch": metrics_from_row(rows[-1]),
        "best_epoch_by_map50_95": metrics_from_row(best),
        "epoch_count": len(rows),
    }


def write_class_aware_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    dataset = payload["dataset"]
    policy = payload["policy"]
    lines = [
        "# Class-Aware Policy Short-Training Report",
        "",
        "- Scope: 5 epoch top1 short-training only; no formal 50 epoch training was run.",
        f"- Policy ID: `{payload['policy_id']}`",
        f"- Dataset: `{dataset['data_yaml']}`",
        f"- Train images / bboxes: `{dataset['train_images']}` / `{dataset['train_bboxes']}`",
        f"- Val images / bboxes: `{dataset['val_images']}` / `{dataset['val_bboxes']}`",
        f"- Modified augmented images: `{dataset['augmented_modified_images']}`",
        f"- Unmodified augmented duplicates: `{dataset['augmented_unmodified_images']}`",
        f"- copy_paste new boxes: `{dataset['copy_paste_new_boxes']}`",
        f"- Precision: `{metrics['precision']:.3f}`",
        f"- Recall: `{metrics['recall']:.3f}`",
        f"- mAP50: `{metrics['map50']:.3f}`",
        f"- mAP50-95: `{metrics['map50_95']:.3f}`",
        f"- short_train_score: `{metrics['short_train_score']:.6f}`",
        f"- score formula: `{metrics['short_train_score_formula']}`",
        f"- OOM: `{str(payload['oom']).lower()}`",
        "",
        "## Branches",
        "",
    ]
    for branch in policy["branches"]:
        lines.append(f"- `{branch['branch_name']}`: target=`{', '.join(branch['target_classes'])}`, ops=`{', '.join(op['name'] for op in branch['ops'])}`")
    lines.extend(
        [
            "",
            "## Per-Class Recall/AP50",
            "",
            "| class id | class | Recall | AP50 |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in metrics["per_class"]:
        lines.append(f"| {row['class_id']} | {row['name']} | {row['recall']:.3f} | {row['ap50']:.3f} |")
    write_markdown(path, lines)


def write_comparison_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    ref = payload["comparison_to_diag_policy_001"]["diag_policy_001_metrics"]
    delta = payload["comparison_to_diag_policy_001"]["delta"]
    lines = [
        "# Class-Aware vs diag_policy_001 Short-Training",
        "",
        "| policy | Precision | Recall | mAP50 | mAP50-95 | short_train_score |",
        "|---|---:|---:|---:|---:|---:|",
        f"| class_aware_policy_001 | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['map50']:.3f} | {metrics['map50_95']:.3f} | {metrics['short_train_score']:.6f} |",
        f"| diag_policy_001 | {ref['precision']:.3f} | {ref['recall']:.3f} | {ref['map50']:.3f} | {ref['map50_95']:.3f} | {ref['short_train_score']:.6f} |",
        f"| delta | {delta['precision']:+.3f} | {delta['recall']:+.3f} | {delta['map50']:+.3f} | {delta['map50_95']:+.3f} | {delta['short_train_score']:+.6f} |",
        "",
        f"- class-aware policy beats diag_policy_001: `{str(payload['comparison_to_diag_policy_001']['class_aware_beats_diag_policy_001']).lower()}`",
        f"- Recommend formal 50 epoch rerun: `{str(payload['recommend_formal_50epoch']).lower()}`",
        "",
        "## Per-Class Recall/AP50 Delta",
        "",
        "| class | delta Recall | delta AP50 | class-aware Recall | diag_policy_001 Recall | class-aware AP50 | diag_policy_001 AP50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["comparison_to_diag_policy_001"]["per_class"]:
        lines.append(
            f"| {row['class_name']} | {row['delta_recall']:+.3f} | {row['delta_ap50']:+.3f} | "
            f"{row['class_aware_recall']:.3f} | {row['diag_policy_001_recall']:.3f} | "
            f"{row['class_aware_ap50']:.3f} | {row['diag_policy_001_ap50']:.3f} |"
        )
    write_markdown(path, lines)


def update_state_docs(payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    cmp = payload["comparison_to_diag_policy_001"]
    section = "\n".join(
        [
            "## Class-Aware Policy Short-Training Validation",
            "",
            f"- Run ID: `{RUN_ID}`",
            "- Scope: top1 class-aware mixed policy short-training only; no formal 50 epoch training.",
            f"- Policy: `{payload['policy_id']}`",
            f"- Train images / bboxes: `{payload['dataset']['train_images']}` / `{payload['dataset']['train_bboxes']}`",
            f"- Precision: `{metrics['precision']:.3f}`",
            f"- Recall: `{metrics['recall']:.3f}`",
            f"- mAP50: `{metrics['map50']:.3f}`",
            f"- mAP50-95: `{metrics['map50_95']:.3f}`",
            f"- short_train_score: `{metrics['short_train_score']:.6f}`",
            f"- Beats diag_policy_001 short-training score `0.609204`: `{str(cmp['class_aware_beats_diag_policy_001']).lower()}`",
            f"- Recommend formal 50 epoch rerun: `{str(payload['recommend_formal_50epoch']).lower()}`",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/class_aware_shorttrain_report.md`",
            f"- Comparison: `outputs/experiments/{RUN_ID}/reports/class_aware_vs_diag_policy_001_shorttrain.md`",
            f"- Metrics JSON: `outputs/experiments/{RUN_ID}/metrics/class_aware_shorttrain_metrics.json`",
        ]
    )
    for path in [PROJECT_ROOT / "PROJECT_STATE.md", PROJECT_ROOT / "CODEX_HANDOFF.md", PROJECT_ROOT / "EXPERIMENT_LOG.md"]:
        upsert_section(path, "CLASS_AWARE_POLICY_SHORTTRAIN", section)


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


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def has_oom(*paths: Path) -> bool:
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace").lower() for path in paths if path.exists())
    return any(pattern in text for pattern in ["out of memory", "cuda oom", "cuda out of memory"])


def count_images(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def count_label_rows(root: Path) -> int:
    total = 0
    for path in root.rglob("*.txt"):
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def flatten_stem(path: Path) -> str:
    return "__".join(path.with_suffix("").parts)


def lerp(low: float, high: float, t: float) -> float:
    return low + (high - low) * max(0.0, min(1.0, float(t)))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


if __name__ == "__main__":
    main()
