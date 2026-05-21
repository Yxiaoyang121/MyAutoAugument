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
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RUN_ID = "20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis"
BASELINE_RUN_ID = "20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep"
TOP3_RUN_ID = "20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline"
DIAGAUG_POLICY_ID = "diag_policy_001"

DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "experiments" / BASELINE_RUN_ID / "train" / "weights" / "best.pt"
DEFAULT_DATA_YAML = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position" / "data.yaml"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID
SOURCE_DIAGNOSIS = PROJECT_ROOT / "outputs" / "experiments" / TOP3_RUN_ID / "diagnosis" / "diagnosis.json"
SOURCE_PROXY_RANKING = PROJECT_ROOT / "outputs" / "experiments" / TOP3_RUN_ID / "proxy" / "proxy_ranking.json"
PREFERRED_PYTHON = Path("D:/Anaconda/envs/pytorch/python.exe")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
PHOTOMETRIC_PREFIXES = ("clahe", "contrast_up", "gamma_brighten", "brightness_up", "combined_photometric")
DIAGAUG_OPS = {"clahe", "contrast", "gamma", "brightness"}


@dataclass(frozen=True)
class YoloImageRecord:
    image_path: Path
    label_path: Path
    relative_path: Path


@dataclass(frozen=True)
class YoloSplit:
    train_records: list[YoloImageRecord]
    val_records: list[YoloImageRecord]
    train_images_dir: Path
    train_labels_dir: Path
    val_images_dir: Path
    val_labels_dir: Path


@dataclass(frozen=True)
class TransformSpec:
    name: str
    family: str
    params: dict[str, Any]


@dataclass(frozen=True)
class TransformResult:
    image: np.ndarray
    pred_to_original: Callable[[np.ndarray], np.ndarray]
    original_to_transformed: Callable[[np.ndarray], np.ndarray]
    notes: str


def main() -> None:
    args = parse_args()
    maybe_reexec_preferred_python(args.device)
    configure_environment()
    run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Counterfactual diagnosis for baseline missed YOLO defects.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--data-yaml", default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-fn-samples", type=int, default=200)
    parser.add_argument("--save-debug-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def maybe_reexec_preferred_python(device: str | int | None) -> None:
    device_text = str(device or "").lower()
    if device_text in {"", "cpu", "none"}:
        return
    if os.environ.get("COUNTERFACTUAL_DIAG_REEXEC") == "1":
        return
    if not PREFERRED_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    preferred = PREFERRED_PYTHON.resolve()
    if current == preferred:
        return
    os.environ["COUNTERFACTUAL_DIAG_REEXEC"] = "1"
    env = os.environ.copy()
    completed = subprocess.run([str(preferred), *sys.argv], env=env)
    raise SystemExit(completed.returncode)


def configure_environment() -> None:
    yolo_config = PROJECT_ROOT / "outputs" / "Ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def run(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    started = time.time()
    model_path = resolve_path(args.model)
    data_yaml = resolve_path(args.data_yaml)
    dataset_root = resolve_path(args.dataset_root) if args.dataset_root else resolve_dataset_root(data_yaml)
    output_dir = resolve_path(args.output_dir)
    prepare_output_dir(output_dir)

    class_names = load_class_names_from_data_yaml(data_yaml)
    split = resolve_yolo_train_val_records(dataset_root, seed=args.seed)
    transforms = counterfactual_transforms()
    run_config = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "counterfactual diagnosis only; no YOLO train, no final training, no training dataset construction",
        "baseline_run_id": BASELINE_RUN_ID,
        "baseline_best_pt": str(model_path.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "val_images_dir": str(split.val_images_dir.resolve()),
        "val_labels_dir": str(split.val_labels_dir.resolve()),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "workers": int(args.workers),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_fn_samples": int(args.max_fn_samples),
        "save_debug_images": bool(args.save_debug_images),
        "seed": int(args.seed),
        "transforms": [{"name": item.name, "family": item.family, "params": item.params} for item in transforms],
    }
    write_json(output_dir / "run_config.json", run_config)

    print(f"[counterfactual] loading model: {model_path}")
    model = YOLO(str(model_path))
    records = list(split.val_records)
    print(f"[counterfactual] predicting baseline on {len(records)} val tiles")
    baseline_predictions = predict_records(
        model=model,
        records=records,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        workers=args.workers,
    )
    baseline_errors = collect_baseline_errors(
        records=records,
        predictions=baseline_predictions,
        class_names=class_names,
        match_iou=args.iou,
        weak_iou=0.3,
    )
    write_json(output_dir / "baseline_predictions.json", baseline_predictions)
    write_json(output_dir / "baseline_error_instances.json", baseline_errors)
    baseline_summary = summarize_baseline_errors(baseline_errors)
    write_json(output_dir / "baseline_error_summary.json", baseline_summary)

    fn_records = [item for item in baseline_errors if item["original_status"] == "FN"]
    sampled_fn = sample_fn_records(fn_records, max_samples=args.max_fn_samples, seed=args.seed)
    print(f"[counterfactual] baseline FN={len(fn_records)}, sampled FN={len(sampled_fn)}")

    all_rows: list[dict[str, Any]] = []
    debug_records: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for index, transform in enumerate(transforms, start=1):
        print(f"[counterfactual] transform {index}/{len(transforms)}: {transform.name}")
        rows = evaluate_transform(
            model=model,
            fn_records=sampled_fn,
            transform=transform,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            workers=args.workers,
        )
        all_rows.extend(rows)
        if args.save_debug_images:
            debug_records.extend(
                save_debug_images(
                    rows=rows,
                    output_dir=output_dir / "debug_images",
                    transform=transform,
                    rng=rng,
                    max_failed=30,
                )
            )

    write_instance_outputs(output_dir, all_rows)
    summary = build_counterfactual_summary(
        rows=all_rows,
        baseline_summary=baseline_summary,
        transforms=transforms,
        class_names=class_names,
        output_dir=output_dir,
        started=started,
        debug_records=debug_records,
    )
    write_json(output_dir / "counterfactual_summary.json", summary)
    write_summary_md(output_dir / "counterfactual_summary.md", summary)

    policy_ranking = build_counterfactual_policy_ranking(summary)
    write_json(output_dir / "counterfactual_policy_ranking.json", policy_ranking)
    write_policy_ranking_md(output_dir / "counterfactual_policy_ranking.md", policy_ranking)

    write_report(output_dir / "reports" / "counterfactual_diagnosis_report.md", summary, policy_ranking)
    update_state_docs(summary)
    print(json.dumps(summary["headline"], ensure_ascii=False, indent=2))


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (PROJECT_ROOT / value).resolve()


def resolve_dataset_root(data_yaml: Path) -> Path:
    path_value: str | None = None
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("path:"):
            path_value = stripped.split(":", 1)[1].strip().strip("'\"")
            break
    if path_value:
        root = Path(path_value)
        return root if root.is_absolute() else (data_yaml.parent / root).resolve()
    return data_yaml.parent.resolve()


def resolve_yolo_train_val_records(dataset_root: str | Path, *, seed: int = 42) -> YoloSplit:
    root = resolve_path(dataset_root)
    train_images = root / "images" / "train"
    val_images = root / "images" / "val"
    train_labels = root / "labels" / "train"
    val_labels = root / "labels" / "val"
    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(f"expected standard YOLO train/val layout under {root}")
    return YoloSplit(
        train_records=find_yolo_records_from_dirs(train_images, train_labels),
        val_records=find_yolo_records_from_dirs(val_images, val_labels),
        train_images_dir=train_images,
        train_labels_dir=train_labels,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
    )


def find_yolo_records_from_dirs(images_dir: Path, labels_dir: Path) -> list[YoloImageRecord]:
    records: list[YoloImageRecord] = []
    for image_path in sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        relative = image_path.relative_to(images_dir)
        records.append(YoloImageRecord(image_path=image_path, label_path=labels_dir / relative.with_suffix(".txt"), relative_path=relative))
    if not records:
        raise ValueError(f"no images found under {images_dir}")
    return records


def load_yolo_labels(label_path: str | Path, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(label_path)
    if not path.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    labels: list[int] = []
    yolo_boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"YOLO label line must contain at least 5 columns: {line}")
        labels.append(int(float(parts[0])))
        yolo_boxes.append([float(value) for value in parts[1:5]])
    if not yolo_boxes:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return np.asarray(labels, dtype=np.int64), yolo_to_xyxy(np.asarray(yolo_boxes, dtype=np.float32), image_width, image_height)


def yolo_to_xyxy(boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 4)
    out = np.zeros_like(arr, dtype=np.float32)
    out[:, 0] = (arr[:, 0] - arr[:, 2] / 2.0) * image_width
    out[:, 1] = (arr[:, 1] - arr[:, 3] / 2.0) * image_height
    out[:, 2] = (arr[:, 0] + arr[:, 2] / 2.0) * image_width
    out[:, 3] = (arr[:, 1] + arr[:, 3] / 2.0) * image_height
    return out


def bbox_iou(boxes_a: np.ndarray, boxes_b: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, eps)).astype(np.float32)


def roi_quality(image: np.ndarray, bbox: np.ndarray | list[float]) -> dict[str, Any]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return {"brightness_mean": 0.0, "contrast_std": 0.0, "low_contrast": True, "too_dark": True, "too_bright": False}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    brightness = float(gray.mean())
    contrast = float(gray.std())
    return {
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "low_contrast": contrast < 20.0,
        "too_dark": brightness < 60.0,
        "too_bright": brightness > 200.0,
    }


def assign_size_bucket(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def is_near_edge(bbox: np.ndarray | list[float], width: int, height: int, threshold: float = 0.05) -> bool:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (
        x1 <= threshold * width
        or y1 <= threshold * height
        or (width - x2) <= threshold * width
        or (height - y2) <= threshold * height
    )


def load_class_names_from_data_yaml(path: str | Path) -> dict[int, str]:
    yaml_path = Path(path)
    names: dict[int, str] = {}
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("names:"):
            in_names = True
            inline = stripped[len("names:") :].strip()
            if inline.startswith("[") and inline.endswith("]"):
                values = [item.strip().strip("'\"") for item in inline[1:-1].split(",") if item.strip()]
                return {index: value for index, value in enumerate(values)}
            continue
        if in_names:
            match = re.match(r"(\d+)\s*:\s*['\"]?(.+?)['\"]?$", stripped)
            if match:
                names[int(match.group(1))] = match.group(2).strip().strip("'\"")
            elif stripped.startswith("-"):
                names[len(names)] = stripped[1:].strip().strip("'\"")
            elif not line.startswith(" "):
                break
    return names


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_output_dir(path: Path) -> None:
    resolved = path.resolve()
    experiments_root = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    if resolved.exists():
        if not str(resolved).startswith(str(experiments_root)):
            raise RuntimeError(f"refusing to remove output outside experiments: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def counterfactual_transforms() -> list[TransformSpec]:
    return [
        TransformSpec("clahe_clip2", "clahe", {"clip_limit": 2.0, "tile_grid_size": 8}),
        TransformSpec("clahe_clip3", "clahe", {"clip_limit": 3.0, "tile_grid_size": 8}),
        TransformSpec("contrast_up_115", "contrast", {"alpha": 1.15}),
        TransformSpec("contrast_up_130", "contrast", {"alpha": 1.30}),
        TransformSpec("gamma_brighten_075", "gamma", {"gamma": 0.75}),
        TransformSpec("gamma_brighten_085", "gamma", {"gamma": 0.85}),
        TransformSpec("brightness_up_15", "brightness", {"beta": 15}),
        TransformSpec("brightness_up_30", "brightness", {"beta": 30}),
        TransformSpec("sharpen_mild", "sharpen", {"amount": 0.75, "sigma": 1.0}),
        TransformSpec(
            "combined_photometric",
            "combined_photometric",
            {"clahe_clip_limit": 2.0, "contrast_alpha": 1.15, "gamma": 0.85},
        ),
        TransformSpec("zoom_in_context", "scale_context", {"context_scale": 5.0, "min_crop": 256}),
    ]


def predict_records(
    *,
    model: Any,
    records: list[YoloImageRecord],
    imgsz: int,
    batch: int,
    device: str | int,
    conf: float,
    iou: float,
    workers: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for chunk in chunks(records, max(1, batch)):
        sources = [str(record.image_path) for record in chunk]
        results = model.predict(
            source=sources,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=str(device),
            batch=max(1, batch),
            workers=workers,
            verbose=False,
            save=False,
        )
        for record, result in zip(chunk, results):
            output.append(prediction_record_from_result(record, result))
    return output


def predict_images(
    *,
    model: Any,
    images: list[np.ndarray],
    imgsz: int,
    batch: int,
    device: str | int,
    conf: float,
    iou: float,
    workers: int,
) -> list[dict[str, Any]]:
    if not images:
        return []
    results = model.predict(
        source=images,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=str(device),
        batch=max(1, batch),
        workers=workers,
        verbose=False,
        save=False,
    )
    output = []
    for result in results:
        output.append(prediction_arrays_from_result(result))
    return output


def prediction_record_from_result(record: YoloImageRecord, result: Any) -> dict[str, Any]:
    pred = prediction_arrays_from_result(result)
    return {
        "image_path": str(record.image_path.resolve()),
        "label_path": str(record.label_path.resolve()),
        "relative_path": record.relative_path.as_posix(),
        **pred,
    }


def prediction_arrays_from_result(result: Any) -> dict[str, Any]:
    boxes_obj = result.boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return {"pred_labels": [], "pred_boxes": [], "pred_confs": []}
    return {
        "pred_labels": [int(value) for value in boxes_obj.cls.detach().cpu().numpy().tolist()],
        "pred_boxes": [[float(v) for v in row] for row in boxes_obj.xyxy.detach().cpu().numpy().tolist()],
        "pred_confs": [float(value) for value in boxes_obj.conf.detach().cpu().numpy().tolist()],
    }


def collect_baseline_errors(
    *,
    records: list[YoloImageRecord],
    predictions: list[dict[str, Any]],
    class_names: dict[int, str],
    match_iou: float,
    weak_iou: float,
) -> list[dict[str, Any]]:
    by_path = {Path(item["image_path"]).resolve(): item for item in predictions}
    all_rows: list[dict[str, Any]] = []
    for record in records:
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {record.image_path}")
        height, width = image.shape[:2]
        gt_labels, gt_boxes = load_yolo_labels(record.label_path, width, height)
        pred = by_path[record.image_path.resolve()]
        pred_labels = np.asarray(pred["pred_labels"], dtype=np.int64)
        pred_boxes = np.asarray(pred["pred_boxes"], dtype=np.float32).reshape(-1, 4)
        pred_confs = np.asarray(pred["pred_confs"], dtype=np.float32)
        all_rows.extend(
            match_image_errors(
                image=image,
                record=record,
                gt_labels=gt_labels,
                gt_boxes=gt_boxes,
                pred_labels=pred_labels,
                pred_boxes=pred_boxes,
                pred_confs=pred_confs,
                class_names=class_names,
                match_iou=match_iou,
                weak_iou=weak_iou,
            )
        )
    return all_rows


def match_image_errors(
    *,
    image: np.ndarray,
    record: YoloImageRecord,
    gt_labels: np.ndarray,
    gt_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_confs: np.ndarray,
    class_names: dict[int, str],
    match_iou: float,
    weak_iou: float,
) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    ious = bbox_iou(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    gt_status: dict[int, dict[str, Any]] = {}
    pred_order = sorted(range(len(pred_boxes)), key=lambda index: float(pred_confs[index]), reverse=True)

    for pred_index in pred_order:
        candidates = [
            gt_index
            for gt_index in range(len(gt_boxes))
            if gt_index not in matched_gt and int(gt_labels[gt_index]) == int(pred_labels[pred_index])
        ]
        if not candidates:
            continue
        best_gt = max(candidates, key=lambda gt_index: float(ious[pred_index, gt_index]))
        best_iou = float(ious[pred_index, best_gt])
        if best_iou >= match_iou:
            matched_gt.add(best_gt)
            matched_pred.add(pred_index)
            gt_status[best_gt] = {"status": "TP", "pred_index": pred_index, "iou": best_iou, "confidence": float(pred_confs[pred_index])}

    for pred_index in pred_order:
        if pred_index in matched_pred:
            continue
        candidates = [
            gt_index
            for gt_index in range(len(gt_boxes))
            if gt_index not in matched_gt and int(gt_labels[gt_index]) == int(pred_labels[pred_index])
        ]
        if not candidates:
            continue
        best_gt = max(candidates, key=lambda gt_index: float(ious[pred_index, gt_index]))
        best_iou = float(ious[pred_index, best_gt])
        if best_iou >= weak_iou:
            matched_gt.add(best_gt)
            matched_pred.add(pred_index)
            gt_status[best_gt] = {
                "status": "localization_weak",
                "pred_index": pred_index,
                "iou": best_iou,
                "confidence": float(pred_confs[pred_index]),
            }

    rows: list[dict[str, Any]] = []
    for gt_index in range(len(gt_boxes)):
        label = int(gt_labels[gt_index])
        box = gt_boxes[gt_index].astype(np.float32)
        status = gt_status.get(gt_index, {"status": "FN", "pred_index": None, "iou": None, "confidence": None})
        best_same = best_same_class_prediction(label, box, pred_labels, pred_boxes, pred_confs)
        rows.append(
            {
                **base_gt_features(
                    image=image,
                    image_path=record.image_path,
                    relative_path=record.relative_path.as_posix(),
                    gt_index=gt_index,
                    class_id=label,
                    class_name=class_names.get(label, str(label)),
                    gt_bbox=box,
                    width=width,
                    height=height,
                ),
                "original_status": status["status"],
                "matched_pred_index": status["pred_index"],
                "matched_iou": status["iou"],
                "matched_conf": status["confidence"],
                "before_iou": best_same["iou"],
                "before_conf": best_same["confidence"],
                "before_pred_bbox": best_same["bbox"],
            }
        )

    for pred_index in range(len(pred_boxes)):
        if pred_index in matched_pred:
            continue
        rows.append(
            {
                "image_path": str(record.image_path.resolve()),
                "relative_path": record.relative_path.as_posix(),
                "gt_index": None,
                "class_id": int(pred_labels[pred_index]),
                "class_name": class_names.get(int(pred_labels[pred_index]), str(int(pred_labels[pred_index]))),
                "gt_bbox": None,
                "pred_bbox": [float(v) for v in pred_boxes[pred_index]],
                "original_status": "FP",
                "matched_iou": None,
                "matched_conf": float(pred_confs[pred_index]),
            }
        )
    return rows


def base_gt_features(
    *,
    image: np.ndarray,
    image_path: Path,
    relative_path: str,
    gt_index: int,
    class_id: int,
    class_name: str,
    gt_bbox: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(value) for value in gt_bbox]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_ratio = area / max(1.0, float(width * height))
    quality = roi_quality(image, gt_bbox)
    tile_position = classify_tile_position(gt_bbox, width, height)
    return {
        "image_path": str(image_path.resolve()),
        "relative_path": relative_path,
        "gt_index": int(gt_index),
        "class_id": int(class_id),
        "class_name": class_name,
        "gt_bbox": [float(v) for v in gt_bbox],
        "area": float(area),
        "area_ratio": float(area_ratio),
        "area_bin": assign_size_bucket(area_ratio),
        "brightness": float(quality["brightness_mean"]),
        "contrast": float(quality["contrast_std"]),
        "local_std": float(quality["contrast_std"]),
        "is_dark": bool(quality["too_dark"]),
        "is_low_contrast": bool(quality["low_contrast"]),
        "tile_position": tile_position,
        "touches_tile_border": touches_tile_border(gt_bbox, width, height),
        "near_edge": is_near_edge(gt_bbox, width, height),
        "image_width": int(width),
        "image_height": int(height),
    }


def classify_tile_position(bbox: np.ndarray, width: int, height: int) -> str:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    cx = ((x1 + x2) / 2.0) / max(1.0, float(width))
    cy = ((y1 + y2) / 2.0) / max(1.0, float(height))
    near_x = cx < 0.2 or cx > 0.8 or x1 <= 0.05 * width or (width - x2) <= 0.05 * width
    near_y = cy < 0.2 or cy > 0.8 or y1 <= 0.05 * height or (height - y2) <= 0.05 * height
    if near_x and near_y:
        return "corner"
    if near_x or near_y:
        return "edge"
    return "center"


def touches_tile_border(bbox: np.ndarray, width: int, height: int) -> bool:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1 <= 1.0 or y1 <= 1.0 or x2 >= width - 1.0 or y2 >= height - 1.0


def best_same_class_prediction(
    class_id: int,
    gt_bbox: np.ndarray,
    pred_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_confs: np.ndarray,
) -> dict[str, Any]:
    if len(pred_boxes) == 0:
        return {"iou": None, "confidence": None, "bbox": None, "predicted_class": None}
    candidates = [index for index, label in enumerate(pred_labels) if int(label) == int(class_id)]
    if not candidates:
        return {"iou": None, "confidence": None, "bbox": None, "predicted_class": None}
    candidate_boxes = pred_boxes[candidates]
    ious = bbox_iou(candidate_boxes, gt_bbox.reshape(1, 4)).reshape(-1)
    best_local = int(np.argmax(ious))
    best_index = candidates[best_local]
    return {
        "iou": float(ious[best_local]),
        "confidence": float(pred_confs[best_index]),
        "bbox": [float(v) for v in pred_boxes[best_index]],
        "predicted_class": int(pred_labels[best_index]),
    }


def best_any_prediction(
    gt_bbox: np.ndarray,
    pred_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_confs: np.ndarray,
) -> dict[str, Any]:
    if len(pred_boxes) == 0:
        return {"iou": None, "confidence": None, "bbox": None, "predicted_class": None}
    ious = bbox_iou(pred_boxes, gt_bbox.reshape(1, 4)).reshape(-1)
    best_index = int(np.argmax(ious))
    return {
        "iou": float(ious[best_index]),
        "confidence": float(pred_confs[best_index]),
        "bbox": [float(v) for v in pred_boxes[best_index]],
        "predicted_class": int(pred_labels[best_index]),
    }


def sample_fn_records(records: list[dict[str, Any]], *, max_samples: int, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or max_samples >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    indices = sorted(int(index) for index in rng.choice(len(records), size=max_samples, replace=False))
    return [records[index] for index in indices]


def evaluate_transform(
    *,
    model: Any,
    fn_records: list[dict[str, Any]],
    transform: TransformSpec,
    imgsz: int,
    batch: int,
    device: str | int,
    conf: float,
    iou: float,
    workers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in chunks(fn_records, max(1, batch)):
        transformed_images: list[np.ndarray] = []
        transformed_meta: list[tuple[dict[str, Any], TransformResult]] = []
        for fn in chunk:
            image = cv2.imread(fn["image_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read image: {fn['image_path']}")
            result = apply_counterfactual_transform(image, np.asarray(fn["gt_bbox"], dtype=np.float32), transform)
            transformed_images.append(result.image)
            transformed_meta.append((fn, result))
        predictions = predict_images(
            model=model,
            images=transformed_images,
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=conf,
            iou=iou,
            workers=workers,
        )
        for (fn, result), pred in zip(transformed_meta, predictions):
            rows.append(evaluate_instance(fn, transform, result, pred, match_iou=iou, conf=conf))
    return rows


def evaluate_instance(
    fn: dict[str, Any],
    transform: TransformSpec,
    transform_result: TransformResult,
    pred: dict[str, Any],
    *,
    match_iou: float,
    conf: float,
) -> dict[str, Any]:
    gt_bbox = np.asarray(fn["gt_bbox"], dtype=np.float32)
    pred_labels = np.asarray(pred["pred_labels"], dtype=np.int64)
    pred_boxes_transformed = np.asarray(pred["pred_boxes"], dtype=np.float32).reshape(-1, 4)
    pred_confs = np.asarray(pred["pred_confs"], dtype=np.float32)
    pred_boxes_original = transform_result.pred_to_original(pred_boxes_transformed)
    same = best_same_class_prediction(int(fn["class_id"]), gt_bbox, pred_labels, pred_boxes_original, pred_confs)
    any_pred = best_any_prediction(gt_bbox, pred_labels, pred_boxes_original, pred_confs)
    recovered_iou = same["iou"]
    recovered_conf = same["confidence"]
    recovered = (
        recovered_iou is not None
        and recovered_conf is not None
        and float(recovered_iou) >= match_iou
        and float(recovered_conf) >= conf
    )
    transformed_same = best_same_class_prediction(
        int(fn["class_id"]),
        transform_result.original_to_transformed(gt_bbox).astype(np.float32),
        pred_labels,
        pred_boxes_transformed,
        pred_confs,
    )
    before_iou = none_to_zero(fn.get("before_iou"))
    before_conf = none_to_zero(fn.get("before_conf"))
    gt_bbox_transform = transform_result.original_to_transformed(np.asarray(fn["gt_bbox"], dtype=np.float32)).reshape(-1, 4)[0]
    return {
        "image_path": fn["image_path"],
        "relative_path": fn["relative_path"],
        "gt_index": fn["gt_index"],
        "class_id": fn["class_id"],
        "class_name": fn["class_name"],
        "gt_bbox": fn["gt_bbox"],
        "area": fn["area"],
        "area_bin": fn["area_bin"],
        "brightness": fn["brightness"],
        "contrast": fn["contrast"],
        "local_std": fn["local_std"],
        "is_dark": fn["is_dark"],
        "is_low_contrast": fn["is_low_contrast"],
        "tile_position": fn["tile_position"],
        "touches_tile_border": fn["touches_tile_border"],
        "original_status": "FN",
        "transform_name": transform.name,
        "transform_family": transform.family,
        "transform_params": transform.params,
        "recovered": bool(recovered),
        "recovered_transform": transform.name if recovered else None,
        "recovered_iou": recovered_iou,
        "recovered_conf": recovered_conf,
        "predicted_class_after_transform": same["predicted_class"] if same["predicted_class"] is not None else any_pred["predicted_class"],
        "best_iou_after_transform": same["iou"],
        "best_conf_after_transform": same["confidence"],
        "best_any_iou_after_transform": any_pred["iou"],
        "best_any_conf_after_transform": any_pred["confidence"],
        "best_any_class_after_transform": any_pred["predicted_class"],
        "before_iou": fn.get("before_iou"),
        "before_conf": fn.get("before_conf"),
        "before_pred_bbox": fn.get("before_pred_bbox"),
        "iou_gain": none_to_zero(same["iou"]) - before_iou,
        "conf_gain": none_to_zero(same["confidence"]) - before_conf,
        "pred_bbox_after_transform_original_coords": same["bbox"],
        "pred_bbox_after_transform_image_coords": transformed_same["bbox"],
        "gt_bbox_transform_coords": [float(v) for v in gt_bbox_transform],
        "notes": transform_result.notes,
    }


def none_to_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def apply_counterfactual_transform(image: np.ndarray, gt_bbox: np.ndarray, transform: TransformSpec) -> TransformResult:
    height, width = image.shape[:2]

    def identity(boxes: np.ndarray) -> np.ndarray:
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            return arr.reshape(0, 4)
        return arr.reshape(-1, 4).astype(np.float32)

    if transform.family == "clahe":
        out = apply_clahe(image, clip_limit=float(transform.params["clip_limit"]), tile_grid_size=int(transform.params["tile_grid_size"]))
        return TransformResult(out, identity, identity, f"clahe clip_limit={transform.params['clip_limit']} tile=8")
    if transform.family == "contrast":
        out = adjust_contrast(image, alpha=float(transform.params["alpha"]))
        return TransformResult(out, identity, identity, f"contrast alpha={transform.params['alpha']}")
    if transform.family == "gamma":
        out = adjust_gamma(image, gamma=float(transform.params["gamma"]))
        return TransformResult(out, identity, identity, f"gamma brighten gamma={transform.params['gamma']}")
    if transform.family == "brightness":
        out = adjust_brightness(image, beta=float(transform.params["beta"]))
        return TransformResult(out, identity, identity, f"brightness beta={transform.params['beta']}")
    if transform.family == "sharpen":
        out = sharpen_mild(image, amount=float(transform.params["amount"]), sigma=float(transform.params["sigma"]))
        return TransformResult(out, identity, identity, "mild unsharp mask")
    if transform.family == "combined_photometric":
        out = apply_clahe(image, clip_limit=float(transform.params["clahe_clip_limit"]), tile_grid_size=8)
        out = adjust_contrast(out, alpha=float(transform.params["contrast_alpha"]))
        out = adjust_gamma(out, gamma=float(transform.params["gamma"]))
        return TransformResult(out, identity, identity, "clahe + mild contrast + gamma")
    if transform.family == "scale_context":
        crop, meta = zoom_context(image, gt_bbox, context_scale=float(transform.params["context_scale"]), min_crop=int(transform.params["min_crop"]))

        def pred_to_original(boxes: np.ndarray) -> np.ndarray:
            arr = np.asarray(boxes, dtype=np.float32)
            if arr.size == 0:
                return arr.reshape(0, 4)
            arr = arr.reshape(-1, 4).copy()
            arr[:, [0, 2]] = arr[:, [0, 2]] / width * meta["crop_w"] + meta["x1"]
            arr[:, [1, 3]] = arr[:, [1, 3]] / height * meta["crop_h"] + meta["y1"]
            return arr.astype(np.float32)

        def original_to_transformed(boxes: np.ndarray) -> np.ndarray:
            arr = np.asarray(boxes, dtype=np.float32)
            if arr.size == 0:
                return arr.reshape(0, 4)
            arr = arr.reshape(-1, 4).copy()
            arr[:, [0, 2]] = (arr[:, [0, 2]] - meta["x1"]) / max(1.0, meta["crop_w"]) * width
            arr[:, [1, 3]] = (arr[:, [1, 3]] - meta["y1"]) / max(1.0, meta["crop_h"]) * height
            return arr.astype(np.float32)

        return TransformResult(crop, pred_to_original, original_to_transformed, json.dumps(meta, ensure_ascii=False))
    raise ValueError(f"unknown transform family: {transform.family}")


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


def zoom_context(image: np.ndarray, gt_bbox: np.ndarray, *, context_scale: float, min_crop: int) -> tuple[np.ndarray, dict[str, float]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in gt_bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    side = max(float(min_crop), max(box_w, box_h) * context_scale)
    side = min(side, float(max(width, height)))
    crop_w = min(float(width), side)
    crop_h = min(float(height), side)
    crop_x1 = max(0.0, min(float(width) - crop_w, cx - crop_w / 2.0))
    crop_y1 = max(0.0, min(float(height) - crop_h, cy - crop_h / 2.0))
    crop_x2 = min(float(width), crop_x1 + crop_w)
    crop_y2 = min(float(height), crop_y1 + crop_h)
    ix1, iy1, ix2, iy2 = [int(round(v)) for v in [crop_x1, crop_y1, crop_x2, crop_y2]]
    ix2 = max(ix1 + 1, min(width, ix2))
    iy2 = max(iy1 + 1, min(height, iy2))
    crop = image[iy1:iy2, ix1:ix2]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    meta = {"x1": float(ix1), "y1": float(iy1), "x2": float(ix2), "y2": float(iy2), "crop_w": float(ix2 - ix1), "crop_h": float(iy2 - iy1)}
    return resized, meta


def write_instance_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "counterfactual_instances.json", rows)
    csv_path = output_dir / "counterfactual_instances.csv"
    fieldnames = [
        "image_path",
        "class_id",
        "class_name",
        "gt_bbox",
        "area",
        "area_bin",
        "brightness",
        "contrast",
        "local_std",
        "is_dark",
        "is_low_contrast",
        "tile_position",
        "touches_tile_border",
        "original_status",
        "transform_name",
        "recovered",
        "recovered_iou",
        "recovered_conf",
        "predicted_class_after_transform",
        "before_iou",
        "before_conf",
        "iou_gain",
        "conf_gain",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            serial["gt_bbox"] = json.dumps(row.get("gt_bbox"), ensure_ascii=False)
            writer.writerow(serial)


def build_counterfactual_summary(
    *,
    rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    transforms: list[TransformSpec],
    class_names: dict[int, str],
    output_dir: Path,
    started: float,
    debug_records: list[dict[str, Any]],
) -> dict[str, Any]:
    transform_stats = []
    for transform in transforms:
        subset = [row for row in rows if row["transform_name"] == transform.name]
        transform_stats.append(transform_stat(transform, subset, class_names))
    transform_stats = sorted(transform_stats, key=lambda item: (-item["recovery_rate"], item["transform"]))
    highest = transform_stats[0] if transform_stats else {}
    policy_support = build_diag_policy_support(rows)
    unrecovered = unrecovered_class_summary(rows, class_names)
    source_diagnosis = read_json(SOURCE_DIAGNOSIS) if SOURCE_DIAGNOSIS.exists() else {}
    triggered = [issue.get("type") for issue in source_diagnosis.get("issues", [])]
    supported = policy_support["any_diag_policy_001_op_recovery_rate"] > 0.0
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - started, 3),
        "no_training_executed": True,
        "baseline_summary": baseline_summary,
        "tested_fn_count": len({row_key(row) for row in rows}),
        "transform_count": len(transforms),
        "transform_stats": transform_stats,
        "highest_recovery_transform": highest,
        "diag_policy_001_support": policy_support,
        "source_diagnosis_comparison": {
            "source_diagnosis_path": str(SOURCE_DIAGNOSIS),
            "triggered_issues": triggered,
            "contains_low_contrast_missed_defect": "low_contrast_missed_defect" in triggered,
            "counterfactual_supports_low_contrast_issue": supported,
            "interpretation": "supports diag_policy_001" if supported else "does not support diag_policy_001 from recovery evidence",
        },
        "most_recovered_classes": most_recovered_classes(rows, class_names),
        "unrecovered_classes": unrecovered,
        "debug_images_dir": str((output_dir / "debug_images").resolve()),
        "debug_image_count": len(debug_records),
        "debug_records": debug_records,
        "headline": {
            "baseline_fn_count": baseline_summary["fn_count"],
            "tested_fn_count": len({row_key(row) for row in rows}),
            "highest_recovery_transform": highest.get("transform"),
            "highest_recovery_rate": highest.get("recovery_rate"),
            "diag_policy_001_supported": supported,
        },
    }


def row_key(row: dict[str, Any]) -> str:
    return f"{row['image_path']}::{row['gt_index']}"


def transform_stat(transform: TransformSpec, subset: list[dict[str, Any]], class_names: dict[int, str]) -> dict[str, Any]:
    tested = len(subset)
    recovered = sum(1 for row in subset if row["recovered"])
    per_class: dict[str, dict[str, Any]] = {}
    class_ids = sorted({int(row["class_id"]) for row in subset})
    for class_id in class_ids:
        rows = [row for row in subset if int(row["class_id"]) == class_id]
        rec = sum(1 for row in rows if row["recovered"])
        per_class[str(class_id)] = {
            "class_id": class_id,
            "class_name": class_names.get(class_id, str(class_id)),
            "tested_fn_count": len(rows),
            "recovered_count": rec,
            "recovery_rate": rec / max(1, len(rows)),
        }
    low_rows = [row for row in subset if bool(row["is_low_contrast"])]
    dark_rows = [row for row in subset if bool(row["is_dark"])]
    small_rows = [row for row in subset if row["area_bin"] in {"tiny", "small"}]
    return {
        "transform": transform.name,
        "family": transform.family,
        "params": transform.params,
        "tested_fn_count": tested,
        "recovered_count": recovered,
        "recovery_rate": recovered / max(1, tested),
        "per_class": per_class,
        "low_contrast_fn_recovery_rate": rate(low_rows),
        "dark_fn_recovery_rate": rate(dark_rows),
        "small_object_recovery_rate": rate(small_rows),
        "average_conf_gain": float(np.mean([row["conf_gain"] for row in subset])) if subset else 0.0,
        "average_iou_gain": float(np.mean([row["iou_gain"] for row in subset])) if subset else 0.0,
        "main_affected_classes": [
            item["class_name"]
            for item in sorted(per_class.values(), key=lambda value: (-value["recovered_count"], value["class_id"]))
            if item["recovered_count"] > 0
        ][:5],
    }


def rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["recovered"]) / len(rows)


def build_diag_policy_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_instance[row_key(row)].append(row)
    supported_instances = []
    support_by_op: dict[str, dict[str, Any]] = {}
    for op, prefixes in {
        "clahe": ("clahe",),
        "contrast": ("contrast_up",),
        "gamma": ("gamma_brighten",),
        "brightness": ("brightness_up",),
        "combined_photometric": ("combined_photometric",),
    }.items():
        op_rows = [row for row in rows if row["transform_name"].startswith(prefixes)]
        unique_tested = {row_key(row) for row in op_rows}
        unique_recovered = {row_key(row) for row in op_rows if row["recovered"]}
        support_by_op[op] = {
            "tested_fn_count": len(unique_tested),
            "recovered_unique_fn_count": len(unique_recovered),
            "unique_recovery_rate": len(unique_recovered) / max(1, len(unique_tested)),
        }
    for key, items in by_instance.items():
        if any(row["recovered"] and row["transform_name"].startswith(PHOTOMETRIC_PREFIXES) for row in items):
            supported_instances.append(key)
    return {
        "policy_id": DIAGAUG_POLICY_ID,
        "operations": sorted(DIAGAUG_OPS),
        "tested_fn_count": len(by_instance),
        "any_diag_policy_001_op_recovered_unique_fn_count": len(supported_instances),
        "any_diag_policy_001_op_recovery_rate": len(supported_instances) / max(1, len(by_instance)),
        "support_by_operation": support_by_op,
        "interpretation": "photometric counterfactual recovery supports low_contrast_missed_defect policy"
        if supported_instances
        else "no recovered FN under photometric counterfactuals",
    }


def most_recovered_classes(rows: list[dict[str, Any]], class_names: dict[int, str]) -> list[dict[str, Any]]:
    by_class: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if row["recovered"] and row["transform_name"].startswith(PHOTOMETRIC_PREFIXES):
            by_class[int(row["class_id"])].add(row_key(row))
    out = []
    for class_id, keys in by_class.items():
        out.append({"class_id": class_id, "class_name": class_names.get(class_id, str(class_id)), "recovered_unique_fn_count": len(keys)})
    return sorted(out, key=lambda item: (-item["recovered_unique_fn_count"], item["class_id"]))


def unrecovered_class_summary(rows: list[dict[str, Any]], class_names: dict[int, str]) -> list[dict[str, Any]]:
    by_instance: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_instance[row_key(row)].append(row)
    counts: dict[int, int] = defaultdict(int)
    tested: dict[int, int] = defaultdict(int)
    for items in by_instance.values():
        class_id = int(items[0]["class_id"])
        tested[class_id] += 1
        if not any(row["recovered"] for row in items):
            counts[class_id] += 1
    out = []
    for class_id, count in counts.items():
        out.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "unrecovered_unique_fn_count": count,
                "tested_unique_fn_count": tested[class_id],
                "unrecovered_rate": count / max(1, tested[class_id]),
            }
        )
    return sorted(out, key=lambda item: (-item["unrecovered_unique_fn_count"], item["class_id"]))


def summarize_baseline_errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_rows = [row for row in rows if row.get("original_status") in {"TP", "FN", "localization_weak"}]
    tp = sum(1 for row in rows if row.get("original_status") == "TP")
    fp = sum(1 for row in rows if row.get("original_status") == "FP")
    fn = sum(1 for row in rows if row.get("original_status") == "FN")
    weak = sum(1 for row in rows if row.get("original_status") == "localization_weak")
    return {
        "gt_count": len(gt_rows),
        "tp_count": tp,
        "fp_count": fp,
        "fn_count": fn,
        "weak_localization_count": weak,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, len(gt_rows)),
    }


def save_debug_images(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    transform: TransformSpec,
    rng: np.random.Generator,
    max_failed: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    success_rows = [row for row in rows if row["recovered"]]
    failed_rows = [row for row in rows if not row["recovered"]]
    sampled_failed: list[dict[str, Any]] = []
    if failed_rows and max_failed > 0:
        count = min(max_failed, len(failed_rows))
        indices = sorted(int(index) for index in rng.choice(len(failed_rows), size=count, replace=False))
        sampled_failed = [failed_rows[index] for index in indices]
    for row in success_rows + sampled_failed:
        path = save_one_debug_image(row=row, output_dir=output_dir, transform=transform, failed=not row["recovered"])
        if path:
            records.append({"path": str(path.resolve()), "transform": transform.name, "recovered": bool(row["recovered"]), "class_name": row["class_name"]})
    return records


def save_one_debug_image(row: dict[str, Any], output_dir: Path, transform: TransformSpec, *, failed: bool) -> Path | None:
    original = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
    if original is None:
        return None
    gt_bbox = np.asarray(row["gt_bbox"], dtype=np.float32)
    transformed = apply_counterfactual_transform(original, gt_bbox, transform)
    original_vis = original.copy()
    transformed_vis = transformed.image.copy()
    draw_box(original_vis, gt_bbox, (0, 255, 0), "GT")
    if row.get("before_pred_bbox"):
        draw_box(original_vis, np.asarray(row["before_pred_bbox"], dtype=np.float32), (0, 180, 255), "before")
    draw_box(transformed_vis, np.asarray(row["gt_bbox_transform_coords"], dtype=np.float32), (0, 255, 0), "GT")
    if row.get("pred_bbox_after_transform_image_coords"):
        color = (0, 0, 255) if row["recovered"] else (0, 180, 255)
        draw_box(transformed_vis, np.asarray(row["pred_bbox_after_transform_image_coords"], dtype=np.float32), color, "pred")
    label = (
        f"{row['class_name']} {transform.name} "
        f"rec={row['recovered']} iou={fmt(row.get('recovered_iou'))} conf={fmt(row.get('recovered_conf'))}"
    )
    canvas = stack_debug(original_vis, transformed_vis, label)
    status = "failed" if failed else "recovered"
    filename = "{status}_{cls}_{transform}_iou{ious}_conf{conf}_{stem}_{idx}.jpg".format(
        status=status,
        cls=sanitize_filename(str(row["class_name"]))[:40],
        transform=sanitize_filename(transform.name),
        ious=fmt(row.get("recovered_iou")).replace(".", "p"),
        conf=fmt(row.get("recovered_conf")).replace(".", "p"),
        stem=sanitize_filename(Path(row["image_path"]).stem)[:50],
        idx=row.get("gt_index"),
    )
    path = output_dir / filename
    cv2.imwrite(str(path), canvas)
    return path


def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int], text: str) -> None:
    if box.size != 4:
        return
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def stack_debug(original: np.ndarray, transformed: np.ndarray, label: str) -> np.ndarray:
    height = min(original.shape[0], transformed.shape[0])
    original = cv2.resize(original, (height, height), interpolation=cv2.INTER_AREA)
    transformed = cv2.resize(transformed, (height, height), interpolation=cv2.INTER_AREA)
    header = np.full((52, height * 2, 3), 255, dtype=np.uint8)
    cv2.putText(header, "original + GT", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(header, "counterfactual + pred", (height + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(header, label[:160], (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    body = np.hstack([original, transformed])
    return np.vstack([header, body])


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return safe.strip("._") or "class"


def fmt(value: Any) -> str:
    if value is None:
        return "0.000"
    return f"{float(value):.3f}"


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Counterfactual Diagnosis Summary",
        "",
        "- Scope: prediction-only counterfactual diagnosis; no training was run.",
        f"- Baseline FN count: `{summary['baseline_summary']['fn_count']}`",
        f"- Tested FN count: `{summary['tested_fn_count']}`",
        f"- Highest recovery transform: `{summary['highest_recovery_transform'].get('transform')}` "
        f"rate=`{summary['highest_recovery_transform'].get('recovery_rate', 0.0):.4f}`",
        f"- diag_policy_001 photometric unique FN recovery rate: "
        f"`{summary['diag_policy_001_support']['any_diag_policy_001_op_recovery_rate']:.4f}`",
        "",
        "## Transform Recovery",
        "",
        "| transform | tested FN | recovered | recovery_rate | main affected classes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in summary["transform_stats"]:
        classes = ", ".join(item["main_affected_classes"]) if item["main_affected_classes"] else "-"
        lines.append(
            f"| `{item['transform']}` | {item['tested_fn_count']} | {item['recovered_count']} | "
            f"{item['recovery_rate']:.4f} | {classes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path: Path, summary: dict[str, Any], policy_ranking: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    support = summary["diag_policy_001_support"]
    source = summary["source_diagnosis_comparison"]
    highest = summary["highest_recovery_transform"]
    most_classes = summary["most_recovered_classes"][:8]
    unrecovered = summary["unrecovered_classes"][:8]
    lines = [
        "# Counterfactual Diagnosis Report",
        "",
        "## Scope",
        "",
        "- This run is prediction-only counterfactual diagnosis.",
        "- It did not run YOLO train, final 50 epoch training, top3 short-training, or training dataset construction.",
        "- Flow: baseline best.pt -> val FN detection -> counterfactual transforms on FN tiles -> re-predict -> recovery statistics.",
        "",
        "## Direct Answers",
        "",
        f"1. Low-contrast/brightness transforms recovered `{support['any_diag_policy_001_op_recovered_unique_fn_count']}` "
        f"of `{support['tested_fn_count']}` tested FN instances "
        f"(`{support['any_diag_policy_001_op_recovery_rate']:.4f}` unique recovery rate).",
        f"2. Highest recovery transform: `{highest.get('transform')}` with recovery_rate `{highest.get('recovery_rate', 0.0):.4f}`.",
        "3. CLAHE / contrast / gamma / brightness are counted as direct photometric support for "
        f"`{DIAGAUG_POLICY_ID}`; per-op rates are listed below.",
        f"4. diag_policy_001 support: `{source['interpretation']}`.",
        "5. copy_paste cannot be directly validated by this counterfactual because it changes training-set object frequency and placement; "
        "there is no new object inserted into the model weights during prediction-only inference.",
        "6. Scale/size evidence is represented by `zoom_in_context`; if its recovery is materially higher than photometric transforms, "
        "that indicates a scale/context issue.",
        "7. Photometric-recovered classes are listed under class recovery.",
        "8. Classes with high unrecovered counts may need copy-paste, class-balanced augmentation, more samples, or annotation review.",
        f"9. Recommendation: `{'continue prioritizing diag_policy_001' if source['counterfactual_supports_low_contrast_issue'] else 'do not rely on diag_policy_001 from counterfactual evidence alone'}`.",
        "",
        "## Existing Diagnosis Comparison",
        "",
        f"- Source diagnosis: `{source['source_diagnosis_path']}`",
        f"- Triggered issues: `{', '.join(str(item) for item in source['triggered_issues'])}`",
        f"- Contains `low_contrast_missed_defect`: `{source['contains_low_contrast_missed_defect']}`",
        f"- Counterfactual supports low-contrast issue: `{source['counterfactual_supports_low_contrast_issue']}`",
        "",
        "## Transform Recovery Table",
        "",
        "| transform | tested FN | recovered | recovery_rate | low_contrast_rate | dark_rate | small_object_rate | main affected classes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary["transform_stats"]:
        classes = ", ".join(item["main_affected_classes"]) if item["main_affected_classes"] else "-"
        lines.append(
            f"| `{item['transform']}` | {item['tested_fn_count']} | {item['recovered_count']} | "
            f"{item['recovery_rate']:.4f} | {item['low_contrast_fn_recovery_rate']:.4f} | "
            f"{item['dark_fn_recovery_rate']:.4f} | {item['small_object_recovery_rate']:.4f} | {classes} |"
        )
    lines.extend(["", "## diag_policy_001 Operation Support", ""])
    for op, item in support["support_by_operation"].items():
        lines.append(
            f"- `{op}`: recovered `{item['recovered_unique_fn_count']}` / `{item['tested_fn_count']}` "
            f"unique FN, rate `{item['unique_recovery_rate']:.4f}`"
        )
    lines.extend(["", "## Class Recovery", ""])
    if most_classes:
        for item in most_classes:
            lines.append(f"- Recovered: `{item['class_name']}` unique FN `{item['recovered_unique_fn_count']}`")
    else:
        lines.append("- No class had photometric recovered FN.")
    lines.extend(["", "## Still Unrecovered", ""])
    for item in unrecovered:
        lines.append(
            f"- `{item['class_name']}` unrecovered `{item['unrecovered_unique_fn_count']}` / "
            f"`{item['tested_unique_fn_count']}` (`{item['unrecovered_rate']:.4f}`)"
        )
    lines.extend(
        [
            "",
            "## Counterfactual-Adjusted Policy Ranking",
            "",
            "- Ranking starts from the existing proxy/safety ranking.",
            "- Directly testable photometric and scale operations receive a recovery-rate adjustment.",
            "- copy_paste is not hard rejected or directly penalized solely because prediction-only counterfactuals cannot validate new object synthesis.",
            "",
            "| rank | policy_id | original_rank | base_score | recovery_score | adjusted_score | direct_ops | notes |",
            "|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in policy_ranking.get("ranking", []):
        lines.append(
            f"| {row['counterfactual_rank']} | `{row['policy_id']}` | {row.get('original_rank')} | "
            f"{row['base_score']:.4f} | {row['counterfactual_recovery_score']:.4f} | "
            f"{row['counterfactual_adjusted_score']:.4f} | {', '.join(row['directly_tested_ops']) or '-'} | "
            f"{row['notes']} |"
        )
    lines.extend(["", f"Debug images: `{summary['debug_images_dir']}`", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_counterfactual_policy_ranking(summary: dict[str, Any]) -> dict[str, Any]:
    if not SOURCE_PROXY_RANKING.exists():
        return {"source_proxy_ranking": str(SOURCE_PROXY_RANKING), "ranking": []}
    source_rows = read_json(SOURCE_PROXY_RANKING)
    rates = transform_rate_lookup(summary)
    ranking = []
    for index, row in enumerate(source_rows, start=1):
        policy = row.get("policy", row)
        ops = [str(op.get("name", "")).lower() for op in policy.get("operations", [])]
        recovery_scores = []
        tested_ops = []
        notes = []
        for op in ops:
            mapped = op_recovery_score(op, rates)
            if mapped is None:
                if op == "copy_paste":
                    notes.append("copy_paste not directly testable by prediction-only counterfactual")
                continue
            tested_ops.append(op)
            recovery_scores.append(mapped)
        recovery_score = float(np.mean(recovery_scores)) if recovery_scores else 0.0
        base = float(row.get("combined_proxy_safety_score", row.get("proxy_score", 0.0)) or 0.0)
        adjustment = 0.20 * recovery_score if recovery_scores else 0.0
        adjusted = base + adjustment
        ranking.append(
            {
                "policy_id": row.get("policy_id", policy.get("policy_id", policy.get("name"))),
                "original_rank": int(row.get("rank", index) or index),
                "base_score": base,
                "counterfactual_recovery_score": recovery_score,
                "counterfactual_adjustment": adjustment,
                "counterfactual_adjusted_score": adjusted,
                "directly_tested_ops": tested_ops,
                "non_direct_ops": [op for op in ops if op not in tested_ops],
                "contains_copy_paste": "copy_paste" in ops,
                "notes": "; ".join(notes) if notes else "directly testable operations adjusted by FN recovery",
                "policy": policy,
            }
        )
    ranking.sort(key=lambda item: (-item["counterfactual_adjusted_score"], item["original_rank"]))
    for rank, row in enumerate(ranking, start=1):
        row["counterfactual_rank"] = rank
    return {
        "source_proxy_ranking": str(SOURCE_PROXY_RANKING),
        "formula": "counterfactual_adjusted_score = combined_proxy_safety_score + 0.20 * mean_direct_op_recovery_rate",
        "copy_paste_rule": "copy_paste is not directly testable by prediction-only counterfactual diagnosis and is not hard rejected.",
        "ranking": ranking,
    }


def transform_rate_lookup(summary: dict[str, Any]) -> dict[str, float]:
    return {item["transform"]: float(item["recovery_rate"]) for item in summary.get("transform_stats", [])}


def op_recovery_score(op: str, rates: dict[str, float]) -> float | None:
    mapping = {
        "clahe": ["clahe_clip2", "clahe_clip3", "combined_photometric"],
        "contrast": ["contrast_up_115", "contrast_up_130", "combined_photometric"],
        "gamma": ["gamma_brighten_075", "gamma_brighten_085", "combined_photometric"],
        "brightness": ["brightness_up_15", "brightness_up_30"],
        "sharpen": ["sharpen_mild"],
        "scale": ["zoom_in_context"],
        "mild-scale": ["zoom_in_context"],
    }
    keys = mapping.get(op)
    if not keys:
        return None
    values = [rates[key] for key in keys if key in rates]
    if not values:
        return None
    return max(values)


def write_policy_ranking_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Counterfactual-Adjusted Policy Ranking",
        "",
        f"- Formula: `{payload.get('formula')}`",
        f"- copy_paste rule: {payload.get('copy_paste_rule')}",
        "",
        "| rank | policy_id | original_rank | base_score | recovery_score | adjusted_score | direct_ops |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("ranking", []):
        lines.append(
            f"| {row['counterfactual_rank']} | `{row['policy_id']}` | {row['original_rank']} | "
            f"{row['base_score']:.4f} | {row['counterfactual_recovery_score']:.4f} | "
            f"{row['counterfactual_adjusted_score']:.4f} | {', '.join(row['directly_tested_ops']) or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_state_docs(summary: dict[str, Any]) -> None:
    block = counterfactual_doc_block(summary)
    for rel in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
        update_marked_block(PROJECT_ROOT / rel, "COUNTERFACTUAL_DIAGNOSIS", block)


def counterfactual_doc_block(summary: dict[str, Any]) -> str:
    highest = summary["highest_recovery_transform"]
    support = summary["diag_policy_001_support"]
    return "\n".join(
        [
            "<!-- COUNTERFACTUAL_DIAGNOSIS_START -->",
            "## Counterfactual Diagnosis for Baseline Missed Defects",
            "",
            f"- Run ID: `{RUN_ID}`",
            "- Scope: prediction-only counterfactual diagnosis; no training, no 50 epoch run, no top3 short-training.",
            "- Purpose: validate whether baseline FN cases respond to low-contrast/brightness-style transforms.",
            f"- Baseline FN count: `{summary['baseline_summary']['fn_count']}`",
            f"- Tested FN count: `{summary['tested_fn_count']}`",
            f"- Highest recovery transform: `{highest.get('transform')}` recovery_rate=`{highest.get('recovery_rate', 0.0):.4f}`",
            f"- diag_policy_001 unique photometric FN recovery rate: `{support['any_diag_policy_001_op_recovery_rate']:.4f}`",
            f"- Supports low_contrast_missed_defect -> diag_policy_001: `{summary['source_diagnosis_comparison']['counterfactual_supports_low_contrast_issue']}`",
            "- copy_paste is not directly testable by prediction-only counterfactual inference.",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/counterfactual_diagnosis_report.md`",
            f"- Summary JSON: `outputs/experiments/{RUN_ID}/counterfactual_summary.json`",
            f"- Instance table: `outputs/experiments/{RUN_ID}/counterfactual_instances.csv`",
            "<!-- COUNTERFACTUAL_DIAGNOSIS_END -->",
        ]
    )


def update_marked_block(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    if start in text and end in text:
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
        text = pattern.sub(block, text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


if __name__ == "__main__":
    main()
