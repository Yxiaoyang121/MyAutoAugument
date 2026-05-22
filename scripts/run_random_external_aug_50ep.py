from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.augmentations import apply_augmentation  # noqa: E402
from AutoAugment.policies.search_space import default_detection_search_space  # noqa: E402
from AutoAugment.utils import (  # noqa: E402
    IMAGE_EXTENSIONS,
    find_yolo_records_from_dirs,
    flatten_relative_stem,
    load_yolo_sample,
    save_yolo_sample,
)


RUN_ID = "20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep"
BASELINE_ID = "20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep"
DIAGAUG_ID = "20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep"
YOLO_DEFAULT_ID = "20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep"
CLASS_AWARE_SHORT_ID = "20260518_tiled1024_safe_no_ok_position_class_aware_policy_shorttrain"

DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
DATA_YAML = DATASET_ROOT / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID

BASELINE_METRICS = PROJECT_ROOT / "outputs" / "experiments" / BASELINE_ID / "reports" / "baseline_50ep_metrics.json"
DIAGAUG_METRICS = PROJECT_ROOT / "outputs" / "experiments" / DIAGAUG_ID / "reports" / "diagaug_50ep_metrics.json"
YOLO_DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "experiments" / YOLO_DEFAULT_ID / "reports" / "yolo_default_aug_50ep_metrics.json"
CLASS_AWARE_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / CLASS_AWARE_SHORT_ID
    / "metrics"
    / "class_aware_shorttrain_metrics.json"
)

YOLO = Path("D:/Anaconda/envs/pytorch/Scripts/yolo.exe")
MODEL = "yolo11n.pt"
EPOCHS = 50
IMGSZ = 1024
DEFAULT_BATCH = 2
WORKERS = 0
DEVICE = "0"
SEED = 42
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


def main() -> None:
    args = parse_args()
    configure_environment()
    prepare_output_dir(OUTPUT_DIR, force=args.force)

    baseline = read_json(BASELINE_METRICS)
    diagaug = read_json(DIAGAUG_METRICS)
    yolo_default = read_json(YOLO_DEFAULT_METRICS)
    class_aware = read_json(CLASS_AWARE_METRICS)
    class_names = load_class_names(baseline)

    train_records = find_yolo_records_from_dirs(DATASET_ROOT / "images" / "train", DATASET_ROOT / "labels" / "train", missing_label="empty")
    val_records = find_yolo_records_from_dirs(DATASET_ROOT / "images" / "val", DATASET_ROOT / "labels" / "val", missing_label="empty")
    random_policy = generate_random_policy(seed=SEED)
    write_json(OUTPUT_DIR / "random_policy.json", random_policy)

    print("[random-external] building original + 1x random augmented dataset")
    dataset_report = build_random_augmented_dataset(
        random_policy=random_policy,
        train_records=train_records,
        val_records=val_records,
        class_names=class_names,
    )
    write_json(OUTPUT_DIR / "random_augmented_dataset_report.json", dataset_report)
    write_proxy_safety_report(OUTPUT_DIR / "proxy_safety_report.md", dataset_report)

    batch = DEFAULT_BATCH
    oom = False
    print("[random-external] training 50 epochs")
    train_result = run_train(dataset_report["data_yaml"], batch=batch)
    if train_result["returncode"] != 0 and has_oom(OUTPUT_DIR / "logs" / "final_train_batch2.stdout.log", OUTPUT_DIR / "logs" / "final_train_batch2.stderr.log"):
        oom = True
        batch = 1
        remove_train_dir()
        print("[random-external] OOM at batch=2; retrying batch=1")
        train_result = run_train(dataset_report["data_yaml"], batch=batch)
    if train_result["returncode"] != 0:
        raise RuntimeError(f"random external 50 epoch training failed; see {OUTPUT_DIR / 'logs'}")

    print("[random-external] validating best.pt")
    val_result = run_val(dataset_report["data_yaml"], batch=batch)
    if val_result["returncode"] != 0 and has_oom(OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stdout.log", OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stderr.log"):
        oom = True
        batch = 1
        val_result = run_val(dataset_report["data_yaml"], batch=batch)
    if val_result["returncode"] != 0:
        raise RuntimeError(f"random external validation failed; see {OUTPUT_DIR / 'logs'}")

    val_metrics = parse_yolo_val_log(OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stdout.log", class_names)
    train_summary = read_training_summary(OUTPUT_DIR / "train" / "results.csv")
    payload = build_metrics_payload(
        random_policy=random_policy,
        dataset_report=dataset_report,
        val_metrics=val_metrics,
        train_result=train_result,
        val_result=val_result,
        train_summary=train_summary,
        baseline=baseline,
        diagaug=diagaug,
        yolo_default=yolo_default,
        class_aware=class_aware,
        batch=batch,
        oom=oom,
        class_names=class_names,
    )
    write_json(OUTPUT_DIR / "reports" / "random_external_aug_50ep_metrics.json", payload)
    write_random_report(OUTPUT_DIR / "reports" / "random_external_aug_50ep_report.md", payload)
    write_comparison_report(OUTPUT_DIR / "reports" / "compare_baseline_yolo_default_diagaug_random.md", payload)
    update_state_docs(payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random external augmentation 50 epoch YOLO control experiment.")
    parser.add_argument("--force", action="store_true", help="Remove the existing random external output directory before running.")
    return parser.parse_args()


def configure_environment() -> None:
    if YOLO.parent.exists():
        os.environ["PATH"] = str(YOLO.parent) + os.pathsep + os.environ.get("PATH", "")
    yolo_config = PROJECT_ROOT / "outputs" / "Ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def prepare_output_dir(path: Path, *, force: bool) -> None:
    resolved = path.resolve()
    experiments_root = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    if resolved.exists():
        if not force:
            raise FileExistsError(f"output directory already exists; rerun with --force only if replacing this run: {resolved}")
        if not str(resolved).startswith(str(experiments_root)):
            raise RuntimeError(f"refusing to remove output outside experiments: {resolved}")
        shutil.rmtree(resolved)
    for subdir in ["reports", "metrics", "logs"]:
        (resolved / subdir).mkdir(parents=True, exist_ok=True)


def generate_random_policy(*, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    space = default_detection_search_space(operation_count_range=(4, 4), allow_repeated_operations=False)
    policy = space.sample_policy(rng, name=f"random_external_policy_seed{seed:02d}").to_dict()
    return {
        "policy_id": f"random_external_policy_seed{seed:02d}",
        "name": policy["name"],
        "type": "random_external_policy",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "generation_method": "default_detection_search_space(operation_count_range=(4,4), allow_repeated_operations=False)",
        "operation_pool": [operation.name for operation in space.operations],
        "operations": policy["operations"],
        "notes": [
            "This policy is not diagnosis-driven.",
            "It is used as a random external augmentation control against DiagAug.",
            "YOLO built-in augmentations are disabled during training.",
        ],
    }


def build_random_augmented_dataset(
    *,
    random_policy: dict[str, Any],
    train_records: list[Any],
    val_records: list[Any],
    class_names: dict[int, str],
) -> dict[str, Any]:
    dataset_dir = OUTPUT_DIR / "dataset" / "final_dataset"
    images_train = dataset_dir / "images" / "train"
    labels_train = dataset_dir / "labels" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_val = dataset_dir / "labels" / "val"
    for directory in [images_train, labels_train, images_val, labels_val]:
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    nc = len(class_names)
    op_counts: Counter[str] = Counter()
    image_save_failures = 0
    label_valid_failures = 0
    hard_filter_reasons: list[str] = []
    class_counts_original: Counter[int] = Counter()
    class_counts_augmented: Counter[int] = Counter()
    original_bbox_count = 0
    augmented_bbox_raw_count = 0
    augmented_bbox_valid_count = 0
    invalid_bbox_count = 0
    class_id_checked_count = 0
    class_id_valid_count = 0
    augmented_modified_images = 0
    augmented_unmodified_images = 0

    for record in train_records:
        sample = load_yolo_sample(record)
        class_counts_original.update(int(value) for value in sample["labels"].tolist())
        original_bbox_count += int(len(sample["labels"]))
        stem = flatten_relative_stem(record.relative_path)
        ext = record.image_path.suffix.lower() or ".jpg"
        original_target = {"image": sample["image"], "labels": sample["labels"], "bboxes": sample["bboxes"]}
        try:
            save_yolo_sample(original_target, images_train / f"orig_{stem}{ext}", labels_train / f"orig_{stem}.txt")
        except Exception as exc:
            image_save_failures += 1
            hard_filter_reasons.append(f"original image save failed for {record.relative_path}: {exc}")
            continue

        augmented, audit = apply_random_policy_with_audit(sample, random_policy, rng)
        op_counts.update(audit["ops_applied"])
        if audit["ops_applied"]:
            augmented_modified_images += 1
        else:
            augmented_unmodified_images += 1

        filtered, safety = validate_and_filter_sample(augmented, nc=nc)
        augmented_bbox_raw_count += safety["raw_bbox_count"]
        augmented_bbox_valid_count += safety["valid_bbox_count"]
        invalid_bbox_count += safety["invalid_bbox_count"]
        class_id_checked_count += safety["class_id_checked_count"]
        class_id_valid_count += safety["class_id_valid_count"]
        class_counts_augmented.update(int(value) for value in filtered["labels"].tolist())
        if safety["class_id_invalid_count"] > 0:
            label_valid_failures += safety["class_id_invalid_count"]
            hard_filter_reasons.append(f"class id out of range in augmented sample {record.relative_path}")
        try:
            save_yolo_sample(filtered, images_train / f"aug_{stem}{ext}", labels_train / f"aug_{stem}.txt")
        except Exception as exc:
            image_save_failures += 1
            hard_filter_reasons.append(f"augmented image save failed for {record.relative_path}: {exc}")

    for record in val_records:
        sample = load_yolo_sample(record)
        try:
            save_yolo_sample(sample, images_val / record.relative_path, labels_val / record.relative_path.with_suffix(".txt"))
        except Exception as exc:
            image_save_failures += 1
            hard_filter_reasons.append(f"validation image save failed for {record.relative_path}: {exc}")

    data_yaml = write_data_yaml(dataset_dir / "data.yaml", dataset_dir=dataset_dir, class_names=class_names)
    hard_filter_pass = image_save_failures == 0 and label_valid_failures == 0
    bbox_valid_rate = augmented_bbox_valid_count / max(1, augmented_bbox_raw_count)
    original_bbox_retention = augmented_bbox_valid_count / max(1, original_bbox_count)
    class_id_valid_rate = class_id_valid_count / max(1, class_id_checked_count)
    exposure_score = augmented_modified_images / max(1, len(train_records))
    distribution_change = compute_class_distribution_change(class_counts_original, class_counts_augmented, class_names)
    safety_score = compute_safety_score(
        bbox_valid_rate=bbox_valid_rate,
        original_bbox_retention=original_bbox_retention,
        class_id_valid_rate=class_id_valid_rate,
        image_save_failures=image_save_failures,
    )
    proxy_score = compute_proxy_score(
        random_policy=random_policy,
        exposure_score=exposure_score,
        original_bbox_retention=original_bbox_retention,
        distribution_l1=distribution_change["l1_distance"],
    )

    report = {
        "stage": "random_external_augmented_dataset_builder",
        "status": "completed",
        "dataset_dir": str(dataset_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "source_data_yaml": rel(DATA_YAML),
        "random_policy_json": rel(OUTPUT_DIR / "random_policy.json"),
        "policy_id": random_policy["policy_id"],
        "original_train_count": len(train_records),
        "augmented_train_count": len(train_records),
        "total_train_images": count_images(images_train),
        "val_count": len(val_records),
        "train_images": count_images(images_train),
        "train_bboxes": count_label_rows(labels_train),
        "val_images": count_images(images_val),
        "val_bboxes": count_label_rows(labels_val),
        "original_train_bboxes": original_bbox_count,
        "augmented_train_raw_bboxes": augmented_bbox_raw_count,
        "augmented_train_valid_bboxes": augmented_bbox_valid_count,
        "invalid_bbox_count": invalid_bbox_count,
        "bbox_valid_rate": bbox_valid_rate,
        "original_bbox_retention": original_bbox_retention,
        "class_id_valid_rate": class_id_valid_rate,
        "image_save_failures": image_save_failures,
        "label_valid_failures": label_valid_failures,
        "augmented_modified_images": augmented_modified_images,
        "augmented_unmodified_images": augmented_unmodified_images,
        "operation_application_counts": dict(op_counts),
        "hard_filter_pass": hard_filter_pass,
        "hard_filter_reasons": hard_filter_reasons,
        "safety_soft_penalty_reasons": safety_soft_penalty_reasons(
            bbox_valid_rate=bbox_valid_rate,
            original_bbox_retention=original_bbox_retention,
            exposure_score=exposure_score,
            distribution_l1=distribution_change["l1_distance"],
        ),
        "proxy_score": proxy_score,
        "safety_score": safety_score,
        "combined_proxy_safety_score": 0.55 * proxy_score + 0.45 * safety_score,
        "exposure_score": exposure_score,
        "class_distribution_change": distribution_change,
        "images_train": str(images_train.resolve()),
        "labels_train": str(labels_train.resolve()),
        "images_val": str(images_val.resolve()),
        "labels_val": str(labels_val.resolve()),
    }
    if not hard_filter_pass:
        write_json(OUTPUT_DIR / "random_augmented_dataset_report.json", report)
        raise RuntimeError(f"random external dataset failed hard safety filter: {hard_filter_reasons[:3]}")
    return report


def apply_random_policy_with_audit(sample: dict[str, Any], policy: dict[str, Any], rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
    current_image = np.asarray(sample["image"]).copy()
    current_labels = np.asarray(sample["labels"], dtype=np.int64).copy()
    current_bboxes = np.asarray(sample["bboxes"], dtype=np.float32).reshape(-1, 4).copy()
    ops_applied: list[str] = []
    op_draws: list[dict[str, Any]] = []
    for operation in policy.get("operations", []):
        name = str(operation["name"]).lower()
        prob = float(operation.get("prob", 1.0))
        strength = float(operation.get("strength", 1.0))
        draw = float(rng.random())
        applied = draw <= prob
        op_draws.append({"name": name, "prob": prob, "draw": draw, "applied": applied})
        if not applied:
            continue
        current_image, current_labels, current_bboxes = apply_augmentation(
            name,
            current_image,
            current_labels,
            current_bboxes,
            params=operation.get("params", {}) or {},
            strength=strength,
            rng=rng,
        )
        ops_applied.append(name)
    return (
        {"image": current_image, "labels": current_labels, "bboxes": current_bboxes},
        {"ops_applied": ops_applied, "op_draws": op_draws},
    )


def validate_and_filter_sample(sample: dict[str, Any], *, nc: int) -> tuple[dict[str, Any], dict[str, int]]:
    image = np.asarray(sample["image"])
    if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0 or not np.isfinite(image).all():
        raise ValueError("invalid augmented image array")
    height, width = image.shape[:2]
    labels = np.asarray(sample["labels"], dtype=np.int64).reshape(-1)
    boxes = np.asarray(sample["bboxes"], dtype=np.float32).reshape(-1, 4)
    if len(labels) != len(boxes):
        raise ValueError("labels and bboxes length mismatch")
    raw_count = int(len(labels))
    if raw_count == 0:
        return {"image": image, "labels": labels, "bboxes": boxes}, {
            "raw_bbox_count": 0,
            "valid_bbox_count": 0,
            "invalid_bbox_count": 0,
            "class_id_checked_count": 0,
            "class_id_valid_count": 0,
            "class_id_invalid_count": 0,
        }
    class_valid = (labels >= 0) & (labels < nc)
    finite = np.isfinite(boxes).all(axis=1)
    clipped = boxes.astype(np.float32, copy=True)
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    positive = (clipped[:, 2] - clipped[:, 0] >= 2.0) & (clipped[:, 3] - clipped[:, 1] >= 2.0)
    keep = class_valid & finite & positive
    filtered = {"image": image, "labels": labels[keep], "bboxes": clipped[keep]}
    return filtered, {
        "raw_bbox_count": raw_count,
        "valid_bbox_count": int(keep.sum()),
        "invalid_bbox_count": int((~keep).sum()),
        "class_id_checked_count": raw_count,
        "class_id_valid_count": int(class_valid.sum()),
        "class_id_invalid_count": int((~class_valid).sum()),
    }


def write_data_yaml(path: Path, *, dataset_dir: Path, class_names: dict[int, str]) -> Path:
    lines = [
        f"path: {dataset_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for class_id in sorted(class_names):
        name = str(class_names[class_id]).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  {class_id}: "{name}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    return run_command(command, "final_train", batch)


def run_val(data_yaml: str, *, batch: int) -> dict[str, Any]:
    command = [
        str(YOLO),
        "detect",
        "val",
        f"model={OUTPUT_DIR / 'train' / 'weights' / 'best.pt'}",
        f"data={data_yaml}",
        f"imgsz={IMGSZ}",
        f"batch={batch}",
        f"workers={WORKERS}",
        f"device={DEVICE}",
        f"project={OUTPUT_DIR}",
        "name=val",
        "exist_ok=True",
    ]
    return run_command(command, "final_val", batch)


def run_command(command: list[str], prefix: str, batch: int) -> dict[str, Any]:
    logs_dir = OUTPUT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_text = subprocess.list2cmdline(command)
    stem = f"{prefix}_batch{batch}"
    (logs_dir / f"{stem}.command.txt").write_text(command_text + "\n", encoding="utf-8")
    stdout_path = logs_dir / f"{stem}.stdout.log"
    stderr_path = logs_dir / f"{stem}.stderr.log"
    start = time.time()
    start_iso = datetime.now(UTC).isoformat(timespec="seconds")
    (logs_dir / f"{stem}.start_utc.txt").write_text(start_iso + "\n", encoding="utf-8")
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open("w", encoding="utf-8", errors="replace") as stderr:
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, env=os.environ.copy())
        returncode = process.wait()
    end_iso = datetime.now(UTC).isoformat(timespec="seconds")
    wall = time.time() - start
    (logs_dir / f"{stem}.end_utc.txt").write_text(end_iso + "\n", encoding="utf-8")
    (logs_dir / f"{stem}.exitcode").write_text(str(returncode) + "\n", encoding="utf-8")
    return {
        "command": command_text,
        "returncode": returncode,
        "wall_seconds": wall,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "start_utc": start_iso,
        "end_utc": end_iso,
    }


def remove_train_dir() -> None:
    train_dir = OUTPUT_DIR / "train"
    if train_dir.exists():
        resolved = train_dir.resolve()
        if not str(resolved).startswith(str(OUTPUT_DIR.resolve())):
            raise RuntimeError(f"refusing to remove unexpected train dir: {resolved}")
        shutil.rmtree(train_dir)


def build_metrics_payload(
    *,
    random_policy: dict[str, Any],
    dataset_report: dict[str, Any],
    val_metrics: dict[str, Any],
    train_result: dict[str, Any],
    val_result: dict[str, Any],
    train_summary: dict[str, Any],
    baseline: dict[str, Any],
    diagaug: dict[str, Any],
    yolo_default: dict[str, Any],
    class_aware: dict[str, Any],
    batch: int,
    oom: bool,
    class_names: dict[int, str],
) -> dict[str, Any]:
    random_overall = val_metrics["overall"]
    random_per_class = val_metrics["per_class"]
    baseline_overall = baseline["validation"]["overall"]
    baseline_per_class = baseline["validation"]["per_class"]
    diagaug_overall = diagaug["final_metrics"]
    diagaug_per_class = diagaug["final_per_class"]
    yolo_overall = yolo_default["final_metrics"]
    yolo_per_class = yolo_default["final_per_class"]
    comparisons = {
        "vs_baseline": build_comparison(baseline_overall, baseline_per_class, random_overall, random_per_class, "baseline", "random_external"),
        "vs_diagaug": build_comparison(diagaug_overall, diagaug_per_class, random_overall, random_per_class, "diagaug", "random_external"),
        "vs_yolo_default": build_comparison(yolo_overall, yolo_per_class, random_overall, random_per_class, "yolo_default", "random_external"),
        "four_way_per_class": build_four_way_per_class(
            baseline_per_class,
            yolo_per_class,
            diagaug_per_class,
            random_per_class,
            class_names,
        ),
    }
    summary = {
        "precision": random_overall["precision"],
        "recall": random_overall["recall"],
        "map50": random_overall["map50"],
        "map50_95": random_overall["map50_95"],
        "delta_vs_baseline": comparisons["vs_baseline"]["overall_delta"],
        "delta_vs_diagaug": comparisons["vs_diagaug"]["overall_delta"],
        "delta_vs_yolo_default": comparisons["vs_yolo_default"]["overall_delta"],
        "best_pt": str((OUTPUT_DIR / "train" / "weights" / "best.pt").resolve()),
        "oom": oom,
        "training_wall_seconds": train_result["wall_seconds"],
    }
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": rel(DATA_YAML),
        "model": MODEL,
        "random_policy": random_policy,
        "random_augmented_dataset": dataset_report,
        "train_settings": {
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": batch,
            "workers": WORKERS,
            "device": DEVICE,
            "seed": SEED,
            "yolo_builtin_augmentations_disabled": True,
            "disabled_augmentation_params": DISABLED_YOLO_AUGS,
        },
        "commands": {
            "final_train_command": train_result["command"],
            "final_val_command": val_result["command"],
            "final_train_exitcode": train_result["returncode"],
            "final_val_exitcode": val_result["returncode"],
        },
        "time": {
            "train_start_utc": train_result["start_utc"],
            "train_end_utc": train_result["end_utc"],
            "val_start_utc": val_result["start_utc"],
            "val_end_utc": val_result["end_utc"],
            "training_wall_seconds": train_result["wall_seconds"],
            "validation_wall_seconds": val_result["wall_seconds"],
            "training_loop_time_seconds": train_summary.get("last_epoch", {}).get("time_seconds"),
        },
        "artifacts": {
            "best_pt": str((OUTPUT_DIR / "train" / "weights" / "best.pt").resolve()),
            "last_pt": str((OUTPUT_DIR / "train" / "weights" / "last.pt").resolve()),
            "random_policy_json": rel(OUTPUT_DIR / "random_policy.json"),
            "random_augmented_dataset_report": rel(OUTPUT_DIR / "random_augmented_dataset_report.json"),
            "proxy_safety_report": rel(OUTPUT_DIR / "proxy_safety_report.md"),
            "train_log": rel(OUTPUT_DIR / "logs" / f"final_train_batch{batch}.stdout.log"),
            "val_log": rel(OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stdout.log"),
            "reports_dir": rel(OUTPUT_DIR / "reports"),
        },
        "final_metrics": random_overall,
        "final_per_class": random_per_class,
        "baseline_metrics": baseline_overall,
        "baseline_per_class": baseline_per_class,
        "diagaug_metrics": diagaug_overall,
        "diagaug_per_class": diagaug_per_class,
        "yolo_default_metrics": yolo_overall,
        "yolo_default_per_class": yolo_per_class,
        "comparisons": comparisons,
        "class_aware_shorttrain_context": {
            "run_id": CLASS_AWARE_SHORT_ID,
            "scope": "5 epoch short-training only; not a formal 50 epoch result.",
            "short_train_score": class_aware["metrics"]["short_train_score"],
            "beats_diag_policy_001_shorttrain": class_aware["comparison_to_diag_policy_001"]["class_aware_beats_diag_policy_001"],
            "conclusion": "class_aware_policy_001 did not beat diag_policy_001 in 5 epoch short-training, so no class-aware 50 epoch run was performed.",
        },
        "training_results": train_summary,
        "oom": oom,
        "summary": summary,
        "notes": [
            "Random external augmentation is a control group, not diagnosis-driven.",
            "The augmented training set keeps all original train images and adds one random-augmented copy per train image.",
            "YOLO built-in augmentations were disabled to match DiagAug final training.",
            "Class-aware policy results are short-training only and are not treated as a formal 50 epoch comparison.",
        ],
    }


def build_comparison(
    reference_overall: dict[str, Any],
    reference_per_class: list[dict[str, Any]],
    target_overall: dict[str, Any],
    target_per_class: list[dict[str, Any]],
    reference_name: str,
    target_name: str,
) -> dict[str, Any]:
    ref_by_id = {int(row["class_id"]): row for row in reference_per_class}
    per_class = []
    for row in target_per_class:
        class_id = int(row["class_id"])
        ref = ref_by_id[class_id]
        per_class.append(
            {
                "class_id": class_id,
                "name": row["name"],
                f"{reference_name}_recall": float(ref["recall"]),
                f"{target_name}_recall": float(row["recall"]),
                "delta_recall": float(row["recall"]) - float(ref["recall"]),
                f"{reference_name}_ap50": float(ref["ap50"]),
                f"{target_name}_ap50": float(row["ap50"]),
                "delta_ap50": float(row["ap50"]) - float(ref["ap50"]),
            }
        )
    return {
        "overall_delta": {
            "precision": float(target_overall["precision"]) - float(reference_overall["precision"]),
            "recall": float(target_overall["recall"]) - float(reference_overall["recall"]),
            "map50": float(target_overall["map50"]) - float(reference_overall["map50"]),
            "map50_95": float(target_overall["map50_95"]) - float(reference_overall["map50_95"]),
        },
        "per_class_delta": per_class,
    }


def build_four_way_per_class(
    baseline: list[dict[str, Any]],
    yolo_default: list[dict[str, Any]],
    diagaug: list[dict[str, Any]],
    random_external: list[dict[str, Any]],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    by_name = {
        "baseline": {int(row["class_id"]): row for row in baseline},
        "yolo_default": {int(row["class_id"]): row for row in yolo_default},
        "diagaug": {int(row["class_id"]): row for row in diagaug},
        "random_external": {int(row["class_id"]): row for row in random_external},
    }
    rows = []
    for class_id, name in class_names.items():
        rows.append(
            {
                "class_id": class_id,
                "name": name,
                "baseline_recall": by_name["baseline"][class_id]["recall"],
                "baseline_ap50": by_name["baseline"][class_id]["ap50"],
                "yolo_default_recall": by_name["yolo_default"][class_id]["recall"],
                "yolo_default_ap50": by_name["yolo_default"][class_id]["ap50"],
                "diagaug_recall": by_name["diagaug"][class_id]["recall"],
                "diagaug_ap50": by_name["diagaug"][class_id]["ap50"],
                "random_recall": by_name["random_external"][class_id]["recall"],
                "random_ap50": by_name["random_external"][class_id]["ap50"],
            }
        )
    return rows


def write_random_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["final_metrics"]
    dataset = payload["random_augmented_dataset"]
    policy = payload["random_policy"]
    lines = [
        "# Random External Augmentation 50 Epoch Report",
        "",
        "- Scope: random external augmentation control group, 50 epoch training.",
        f"- Dataset: `{payload['dataset']}`",
        f"- Augmented dataset: `{dataset['dataset_dir']}`",
        f"- Train images / bboxes: `{dataset['train_images']}` / `{dataset['train_bboxes']}`",
        f"- Original train images: `{dataset['original_train_count']}`",
        f"- Random augmented train images: `{dataset['augmented_train_count']}`",
        f"- Val images / bboxes: `{dataset['val_images']}` / `{dataset['val_bboxes']}`",
        f"- Random policy: `{policy['policy_id']}`",
        f"- Operations: `{format_ops(policy['operations'])}`",
        f"- bbox_valid_rate: `{dataset['bbox_valid_rate']:.6f}`",
        f"- original_bbox_retention: `{dataset['original_bbox_retention']:.6f}`",
        f"- safety_score: `{dataset['safety_score']:.6f}`",
        f"- Precision: `{metrics['precision']:.3f}`",
        f"- Recall: `{metrics['recall']:.3f}`",
        f"- mAP50: `{metrics['map50']:.3f}`",
        f"- mAP50-95: `{metrics['map50_95']:.3f}`",
        f"- best.pt: `{payload['artifacts']['best_pt']}`",
        f"- OOM: `{str(payload['oom']).lower()}`",
        f"- Training wall seconds: `{payload['time']['training_wall_seconds']:.1f}`",
        "",
        "## Random Policy",
        "",
        "| operation | prob | strength | params |",
        "|---|---:|---:|---|",
    ]
    for op in policy["operations"]:
        lines.append(f"| {op['name']} | {float(op['prob']):.4f} | {float(op['strength']):.4f} | `{json.dumps(op.get('params', {}), ensure_ascii=False)}` |")
    lines.extend(
        [
            "",
            "## Per-Class Recall/AP50",
            "",
            "| class id | class | Recall | AP50 |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in payload["final_per_class"]:
        lines.append(f"| {row['class_id']} | {row['name']} | {row['recall']:.3f} | {row['ap50']:.3f} |")
    lines.extend(
        [
            "",
            "## Class-Aware Shorttrain Context",
            "",
            "- `class_aware_policy_001` was evaluated only as a 5 epoch short-training policy selection check.",
            "- It did not beat `diag_policy_001`; therefore this random external 50 epoch control is not compared as a formal 50 epoch class-aware result.",
        ]
    )
    write_markdown(path, lines)


def write_comparison_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["final_metrics"]
    baseline = payload["baseline_metrics"]
    yolo_default = payload["yolo_default_metrics"]
    diagaug = payload["diagaug_metrics"]
    rows = [
        ("No-YOLO-Aug baseline", baseline),
        ("YOLO default aug", yolo_default),
        ("DiagAug", diagaug),
        ("Random external aug", metrics),
    ]
    lines = [
        "# Baseline vs YOLO Default vs DiagAug vs Random External",
        "",
        "| run | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in rows:
        lines.append(f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['map50']:.3f} | {row['map50_95']:.3f} |")
    lines.extend(
        [
            "",
            "## Overall Delta For Random External",
            "",
            "| reference | delta Precision | delta Recall | delta mAP50 | delta mAP50-95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, label in [("vs_baseline", "baseline"), ("vs_yolo_default", "YOLO default"), ("vs_diagaug", "DiagAug")]:
        delta = payload["comparisons"][key]["overall_delta"]
        lines.append(f"| {label} | {delta['precision']:+.3f} | {delta['recall']:+.3f} | {delta['map50']:+.3f} | {delta['map50_95']:+.3f} |")
    lines.extend(
        [
            "",
            "## Per-Class Recall/AP50",
            "",
            "| class | baseline R/AP50 | YOLO default R/AP50 | DiagAug R/AP50 | random R/AP50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["comparisons"]["four_way_per_class"]:
        lines.append(
            f"| {row['name']} | {row['baseline_recall']:.3f}/{row['baseline_ap50']:.3f} | "
            f"{row['yolo_default_recall']:.3f}/{row['yolo_default_ap50']:.3f} | "
            f"{row['diagaug_recall']:.3f}/{row['diagaug_ap50']:.3f} | "
            f"{row['random_recall']:.3f}/{row['random_ap50']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Random external augmentation uses the same offline expansion size as DiagAug: original train images plus one augmented copy per train image.",
            "- YOLO built-in augmentation is disabled for both DiagAug final training and this random external control.",
            "- Class-aware policy evaluation remains a 5 epoch short-training result only; it is not a formal 50 epoch comparator here.",
        ]
    )
    write_markdown(path, lines)


def write_proxy_safety_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Random External Policy Proxy/Safety Report",
        "",
        "- Scope: pre-training proxy/safety audit for one randomly sampled external augmentation policy.",
        f"- hard_filter_pass: `{str(report['hard_filter_pass']).lower()}`",
        f"- proxy_score: `{report['proxy_score']:.6f}`",
        f"- safety_score: `{report['safety_score']:.6f}`",
        f"- combined_proxy_safety_score: `{report['combined_proxy_safety_score']:.6f}`",
        f"- bbox_valid_rate: `{report['bbox_valid_rate']:.6f}`",
        f"- original_bbox_retention: `{report['original_bbox_retention']:.6f}`",
        f"- class_id_valid_rate: `{report['class_id_valid_rate']:.6f}`",
        f"- exposure_score: `{report['exposure_score']:.6f}`",
        f"- image_save_failures: `{report['image_save_failures']}`",
        f"- label_valid_failures: `{report['label_valid_failures']}`",
        "",
        "## Operation Applications",
        "",
        "| operation | applied images |",
        "|---|---:|",
    ]
    for op, count in sorted(report["operation_application_counts"].items()):
        lines.append(f"| {op} | {count} |")
    lines.extend(["", "## Safety Soft Penalties", ""])
    if report["safety_soft_penalty_reasons"]:
        lines.extend(f"- {reason}" for reason in report["safety_soft_penalty_reasons"])
    else:
        lines.append("- None.")
    if report["hard_filter_reasons"]:
        lines.extend(["", "## Hard Filter Reasons", ""])
        lines.extend(f"- {reason}" for reason in report["hard_filter_reasons"])
    write_markdown(path, lines)


def update_state_docs(payload: dict[str, Any]) -> None:
    metrics = payload["final_metrics"]
    dataset = payload["random_augmented_dataset"]
    delta_diag = payload["comparisons"]["vs_diagaug"]["overall_delta"]
    section = "\n".join(
        [
            "## Random External Augmentation 50 Epoch Control",
            "",
            f"- Run ID: `{RUN_ID}`",
            "- Scope: random external augmentation control group; not diagnosis-driven.",
            "- Training set: original train images + 1x random augmented train images.",
            "- YOLO built-in augmentations: disabled to match DiagAug final training.",
            f"- Random policy: `{payload['random_policy']['policy_id']}` with ops `{format_ops(payload['random_policy']['operations'])}`",
            f"- Train images / bboxes: `{dataset['train_images']}` / `{dataset['train_bboxes']}`",
            f"- Safety: hard_filter_pass=`{str(dataset['hard_filter_pass']).lower()}`, bbox_valid_rate=`{dataset['bbox_valid_rate']:.6f}`",
            f"- Precision: `{metrics['precision']:.3f}`",
            f"- Recall: `{metrics['recall']:.3f}`",
            f"- mAP50: `{metrics['map50']:.3f}`",
            f"- mAP50-95: `{metrics['map50_95']:.3f}`",
            f"- Delta vs DiagAug: P `{delta_diag['precision']:+.3f}`, R `{delta_diag['recall']:+.3f}`, mAP50 `{delta_diag['map50']:+.3f}`, mAP50-95 `{delta_diag['map50_95']:+.3f}`",
            f"- OOM: `{str(payload['oom']).lower()}`",
            f"- best.pt: `{payload['artifacts']['best_pt']}`",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/random_external_aug_50ep_report.md`",
            f"- Comparison: `outputs/experiments/{RUN_ID}/reports/compare_baseline_yolo_default_diagaug_random.md`",
            f"- Metrics JSON: `outputs/experiments/{RUN_ID}/reports/random_external_aug_50ep_metrics.json`",
        ]
    )
    for path in [PROJECT_ROOT / "PROJECT_STATE.md", PROJECT_ROOT / "CODEX_HANDOFF.md", PROJECT_ROOT / "EXPERIMENT_LOG.md"]:
        upsert_section(path, "RANDOM_EXTERNAL_AUG_50EP", section)


def load_class_names(baseline: dict[str, Any]) -> dict[int, str]:
    return {int(row["class_id"]): str(row["name"]) for row in baseline["validation"]["per_class"]}


def compute_class_distribution_change(original: Counter[int], augmented: Counter[int], class_names: dict[int, str]) -> dict[str, Any]:
    total_original = sum(original.values())
    total_augmented = sum(augmented.values())
    rows = []
    l1 = 0.0
    for class_id in sorted(class_names):
        original_ratio = original[class_id] / max(1, total_original)
        augmented_ratio = augmented[class_id] / max(1, total_augmented)
        l1 += abs(augmented_ratio - original_ratio)
        rows.append(
            {
                "class_id": class_id,
                "name": class_names[class_id],
                "original_count": int(original[class_id]),
                "augmented_count": int(augmented[class_id]),
                "original_ratio": original_ratio,
                "augmented_ratio": augmented_ratio,
                "delta_ratio": augmented_ratio - original_ratio,
            }
        )
    return {"l1_distance": l1, "per_class": rows}


def compute_safety_score(*, bbox_valid_rate: float, original_bbox_retention: float, class_id_valid_rate: float, image_save_failures: int) -> float:
    score = 0.42 * bbox_valid_rate + 0.32 * min(1.0, original_bbox_retention) + 0.20 * class_id_valid_rate
    if image_save_failures:
        score -= 0.25
    return float(max(0.0, min(1.0, score)))


def compute_proxy_score(*, random_policy: dict[str, Any], exposure_score: float, original_bbox_retention: float, distribution_l1: float) -> float:
    op_names = {str(op["name"]) for op in random_policy["operations"]}
    diversity = len(op_names) / 4.0
    distribution_score = max(0.0, 1.0 - distribution_l1)
    return float(max(0.0, min(1.0, 0.35 * exposure_score + 0.25 * diversity + 0.25 * min(1.0, original_bbox_retention) + 0.15 * distribution_score)))


def safety_soft_penalty_reasons(*, bbox_valid_rate: float, original_bbox_retention: float, exposure_score: float, distribution_l1: float) -> list[str]:
    reasons = []
    if bbox_valid_rate < 0.99:
        reasons.append(f"bbox_valid_rate below 0.99: {bbox_valid_rate:.6f}")
    if original_bbox_retention < 0.98:
        reasons.append(f"original_bbox_retention below 0.98: {original_bbox_retention:.6f}")
    if exposure_score < 0.50:
        reasons.append(f"low operation exposure because sampled op probabilities applied to only {exposure_score:.6f} of augmented images")
    if distribution_l1 > 0.05:
        reasons.append(f"class distribution changed with L1 distance {distribution_l1:.6f}")
    return reasons


def parse_yolo_val_log(path: Path, class_names: dict[int, str]) -> dict[str, Any]:
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
        if overall is None or len(rows) >= len(class_names):
            continue
        class_id = len(rows)
        rows.append(
            {
                "class_id": class_id,
                "name": class_names.get(class_id, str(class_id)),
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
    if len(rows) != len(class_names):
        raise RuntimeError(f"expected {len(class_names)} class rows from {path}, parsed {len(rows)}")
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


def format_ops(operations: list[dict[str, Any]]) -> str:
    return ", ".join(f"{op['name']}(p={float(op['prob']):.3f},s={float(op['strength']):.3f})" for op in operations)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


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


if __name__ == "__main__":
    main()
