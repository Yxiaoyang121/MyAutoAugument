from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from AutoAugment.bbox.iou import bbox_iou
from AutoAugment.formats.yolo import load_yolo_labels


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png")
SIZE_BUCKETS = ("tiny", "small", "medium", "large")


def analyze_yolo_errors(
    *,
    images_dir: str | Path,
    labels_dir: str | Path,
    predictions_dir: str | Path,
    class_names: dict[int, str] | None = None,
    match_iou: float = 0.5,
    localization_weak_iou: float = 0.3,
) -> dict[str, Any]:
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    predictions_root = Path(predictions_dir)
    if not images_root.exists():
        raise FileNotFoundError(f"images directory does not exist: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"labels directory does not exist: {labels_root}")
    if not predictions_root.exists():
        raise FileNotFoundError(f"predictions directory does not exist: {predictions_root}")

    image_paths = sorted(path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"no images found under {images_root}")
    names = class_names or {}
    object_records: list[dict[str, Any]] = []
    per_image_records: list[dict[str, Any]] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        prediction_path = predictions_root / relative.with_suffix(".txt")
        gt_labels, gt_boxes = load_yolo_labels(label_path, width, height)
        pred_labels, pred_boxes, pred_confs = load_yolo_predictions(prediction_path, width, height)
        records = match_detections(
            gt_labels=gt_labels,
            gt_boxes=gt_boxes,
            pred_labels=pred_labels,
            pred_boxes=pred_boxes,
            pred_confs=pred_confs,
            image=image,
            image_path=image_path,
            class_names=names,
            match_iou=match_iou,
            localization_weak_iou=localization_weak_iou,
        )
        object_records.extend(records)
        per_image_records.append(_per_image_summary(image_path, records))
    summary = summarize_error_records(object_records, class_names=names)
    return {
        "summary": summary,
        "per_image_errors": per_image_records,
        "per_object_errors": object_records,
    }


def load_yolo_predictions(label_path: str | Path, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(label_path)
    if not path.exists():
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    labels: list[int] = []
    boxes: list[list[float]] = []
    confs: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"YOLO prediction line must contain at least 5 columns: {line}")
        labels.append(int(float(parts[0])))
        x_center, y_center, box_width, box_height = [float(value) for value in parts[1:5]]
        confs.append(float(parts[5]) if len(parts) >= 6 else 1.0)
        x1 = (x_center - box_width / 2.0) * image_width
        y1 = (y_center - box_height / 2.0) * image_height
        x2 = (x_center + box_width / 2.0) * image_width
        y2 = (y_center + box_height / 2.0) * image_height
        boxes.append([x1, y1, x2, y2])
    if not boxes:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32),
        np.asarray(confs, dtype=np.float32),
    )


def match_detections(
    *,
    gt_labels: np.ndarray,
    gt_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_confs: np.ndarray,
    image: np.ndarray,
    image_path: str | Path,
    class_names: dict[int, str] | None = None,
    match_iou: float = 0.5,
    localization_weak_iou: float = 0.3,
) -> list[dict[str, Any]]:
    names = class_names or {}
    height, width = image.shape[:2]
    ious = bbox_iou(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    records: list[dict[str, Any]] = []

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
            matched_pred.add(pred_index)
            matched_gt.add(best_gt)
            records.append(
                _object_record(
                    image=image,
                    image_path=image_path,
                    class_id=int(gt_labels[best_gt]),
                    class_name=names.get(int(gt_labels[best_gt]), str(int(gt_labels[best_gt]))),
                    error_type="TP",
                    gt_bbox=gt_boxes[best_gt],
                    pred_bbox=pred_boxes[pred_index],
                    confidence=float(pred_confs[pred_index]),
                    iou=best_iou,
                    width=width,
                    height=height,
                )
            )

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
        if best_iou >= localization_weak_iou:
            matched_pred.add(pred_index)
            matched_gt.add(best_gt)
            records.append(
                _object_record(
                    image=image,
                    image_path=image_path,
                    class_id=int(gt_labels[best_gt]),
                    class_name=names.get(int(gt_labels[best_gt]), str(int(gt_labels[best_gt]))),
                    error_type="localization_weak",
                    gt_bbox=gt_boxes[best_gt],
                    pred_bbox=pred_boxes[pred_index],
                    confidence=float(pred_confs[pred_index]),
                    iou=best_iou,
                    width=width,
                    height=height,
                )
            )

    for gt_index in range(len(gt_boxes)):
        if gt_index in matched_gt:
            continue
        records.append(
            _object_record(
                image=image,
                image_path=image_path,
                class_id=int(gt_labels[gt_index]),
                class_name=names.get(int(gt_labels[gt_index]), str(int(gt_labels[gt_index]))),
                error_type="FN",
                gt_bbox=gt_boxes[gt_index],
                pred_bbox=None,
                confidence=None,
                iou=None,
                width=width,
                height=height,
            )
        )

    for pred_index in range(len(pred_boxes)):
        if pred_index in matched_pred:
            continue
        class_id = int(pred_labels[pred_index])
        records.append(
            _object_record(
                image=image,
                image_path=image_path,
                class_id=class_id,
                class_name=names.get(class_id, str(class_id)),
                error_type="FP",
                gt_bbox=None,
                pred_bbox=pred_boxes[pred_index],
                confidence=float(pred_confs[pred_index]),
                iou=None,
                width=width,
                height=height,
            )
        )
    return records


def _object_record(
    *,
    image: np.ndarray,
    image_path: str | Path,
    class_id: int,
    class_name: str,
    error_type: str,
    gt_bbox: np.ndarray | None,
    pred_bbox: np.ndarray | None,
    confidence: float | None,
    iou: float | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    feature_box = gt_bbox if gt_bbox is not None else pred_bbox
    if feature_box is None:
        raise ValueError("gt_bbox or pred_bbox is required")
    x1, y1, x2, y2 = [float(value) for value in feature_box]
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    area_ratio = (box_width * box_height) / max(1.0, float(width * height))
    quality = roi_quality(image, feature_box)
    return {
        "image_path": str(Path(image_path)),
        "class_id": class_id,
        "class_name": class_name,
        "gt_bbox": _box_to_list(gt_bbox),
        "pred_bbox": _box_to_list(pred_bbox),
        "confidence": confidence,
        "iou": iou,
        "error_type": error_type,
        "bbox_area_ratio": area_ratio,
        "bbox_width_ratio": box_width / max(1.0, float(width)),
        "bbox_height_ratio": box_height / max(1.0, float(height)),
        "size_bucket": assign_size_bucket(area_ratio),
        "center_x": ((x1 + x2) / 2.0) / max(1.0, float(width)),
        "center_y": ((y1 + y2) / 2.0) / max(1.0, float(height)),
        "near_edge": is_near_edge(feature_box, width, height),
        **quality,
    }


def _box_to_list(box: np.ndarray | None) -> list[float] | None:
    if box is None:
        return None
    return [float(value) for value in box]


def roi_quality(image: np.ndarray, bbox: np.ndarray | list[float]) -> dict[str, Any]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "brightness_mean": 0.0,
            "contrast_std": 0.0,
            "low_contrast": True,
            "too_dark": True,
            "too_bright": False,
        }
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


def summarize_error_records(records: list[dict[str, Any]], *, class_names: dict[int, str] | None = None) -> dict[str, Any]:
    names = class_names or {}
    gt_records = [record for record in records if record["error_type"] in {"TP", "FN", "localization_weak"}]
    tp_count = _count(records, "TP")
    fp_count = _count(records, "FP")
    fn_count = _count(records, "FN")
    localization_weak_count = _count(records, "localization_weak")
    summary: dict[str, Any] = {
        "overall": {
            "gt_count": len(gt_records),
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "localization_weak_count": localization_weak_count,
            "precision": tp_count / max(1, tp_count + fp_count),
            "recall": tp_count / max(1, len(gt_records)),
            "fn_rate": fn_count / max(1, len(gt_records)),
        },
        "by_class": [],
        "by_size": {},
        "by_position": {},
        "quality": {},
        "fp_background": {},
    }
    class_ids = sorted(set([record["class_id"] for record in records]) | set(names))
    for class_id in class_ids:
        items = [record for record in records if int(record["class_id"]) == int(class_id)]
        gt_items = [record for record in items if record["error_type"] in {"TP", "FN", "localization_weak"}]
        tp = _count(items, "TP")
        fp = _count(items, "FP")
        fn = _count(items, "FN")
        summary["by_class"].append(
            {
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), str(int(class_id))),
                "gt_count": len(gt_items),
                "tp_count": tp,
                "fp_count": fp,
                "fn_count": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, len(gt_items)),
                "fn_rate": fn / max(1, len(gt_items)),
                "fp_rate": fp / max(1, tp + fp),
            }
        )
    for bucket in SIZE_BUCKETS:
        items = [record for record in gt_records if record["size_bucket"] == bucket]
        tp = _count(items, "TP")
        fn = _count(items, "FN")
        summary["by_size"][bucket] = {
            "gt_count": len(items),
            "tp_count": tp,
            "fn_count": fn,
            "recall": tp / max(1, len(items)),
            "fn_rate": fn / max(1, len(items)),
        }
    summary["by_position"] = _position_summary(gt_records)
    summary["quality"] = {
        "false_negatives": _quality_summary([record for record in records if record["error_type"] == "FN"]),
        "false_positives": _quality_summary([record for record in records if record["error_type"] == "FP"]),
    }
    fp_records = [record for record in records if record["error_type"] == "FP"]
    summary["fp_background"] = {
        "fp_count": len(fp_records),
        "images_with_fp": len({record["image_path"] for record in fp_records}),
        "low_contrast_rate": _rate(fp_records, "low_contrast"),
        "dark_rate": _rate(fp_records, "too_dark"),
        "bright_rate": _rate(fp_records, "too_bright"),
    }
    return summary


def _position_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = {
        "left": lambda item: item["center_x"] < 1 / 3,
        "center": lambda item: 1 / 3 <= item["center_x"] < 2 / 3,
        "right": lambda item: item["center_x"] >= 2 / 3,
        "top": lambda item: item["center_y"] < 1 / 3,
        "middle": lambda item: 1 / 3 <= item["center_y"] < 2 / 3,
        "bottom": lambda item: item["center_y"] >= 2 / 3,
        "edge": lambda item: bool(item["near_edge"]),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, predicate in groups.items():
        items = [record for record in records if predicate(record)]
        tp = _count(items, "TP")
        fn = _count(items, "FN")
        out[name] = {
            "gt_count": len(items),
            "tp_count": tp,
            "fn_count": fn,
            "recall": tp / max(1, len(items)),
            "fn_rate": fn / max(1, len(items)),
        }
    return out


def _quality_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "brightness_mean": float(np.mean([record["brightness_mean"] for record in records])) if records else None,
        "contrast_std_mean": float(np.mean([record["contrast_std"] for record in records])) if records else None,
        "low_contrast_count": sum(1 for record in records if record["low_contrast"]),
        "dark_count": sum(1 for record in records if record["too_dark"]),
        "bright_count": sum(1 for record in records if record["too_bright"]),
        "low_contrast_rate": _rate(records, "low_contrast"),
        "dark_rate": _rate(records, "too_dark"),
        "bright_rate": _rate(records, "too_bright"),
    }


def _rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if bool(record.get(key))) / len(records)


def _count(records: list[dict[str, Any]], error_type: str) -> int:
    return sum(1 for record in records if record["error_type"] == error_type)


def _per_image_summary(image_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image_path": str(image_path),
        "tp_count": _count(records, "TP"),
        "fp_count": _count(records, "FP"),
        "fn_count": _count(records, "FN"),
        "localization_weak_count": _count(records, "localization_weak"),
    }


def load_class_names_from_data_yaml(path: str | Path) -> dict[int, str]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml does not exist: {yaml_path}")
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
            elif not line.startswith(" "):
                break
    return names


def write_analysis_outputs(
    output_dir: str | Path,
    *,
    analysis: dict[str, Any],
    advice: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = analysis["summary"]
    _write_json(output / "error_summary.json", summary)
    _write_json(output / "augmentation_advice.json", advice)
    _write_json(output / "advisor_search_space.json", advice["advisor_search_space"])
    _write_csv(output / "per_image_errors.csv", analysis["per_image_errors"])
    _write_csv(output / "per_object_errors.csv", analysis["per_object_errors"])
    _write_summary_csv(output / "error_summary.csv", summary)
    (output / "diagnosis_report.txt").write_text(_diagnosis_text(summary), encoding="utf-8")
    (output / "augmentation_advice.txt").write_text(_advice_text(advice), encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for item in summary.get("by_class", []):
        rows.append({"group_type": "class", "group": item["class_name"], **item})
    for name, item in summary.get("by_size", {}).items():
        rows.append({"group_type": "size", "group": name, **item})
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["group_type", "group"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _diagnosis_text(summary: dict[str, Any]) -> str:
    overall = summary.get("overall", {})
    lines = [
        "YOLO validation error diagnosis",
        "",
        f"GT: {overall.get('gt_count', 0)}",
        f"TP: {overall.get('tp_count', 0)}",
        f"FP: {overall.get('fp_count', 0)}",
        f"FN: {overall.get('fn_count', 0)}",
        f"Localization weak: {overall.get('localization_weak_count', 0)}",
        f"Precision: {overall.get('precision', 0):.4f}",
        f"Recall: {overall.get('recall', 0):.4f}",
        "",
        "Class diagnosis:",
    ]
    for item in summary.get("by_class", []):
        lines.append(
            f"- {item['class_name']}: gt={item['gt_count']} tp={item['tp_count']} fp={item['fp_count']} "
            f"fn={item['fn_count']} recall={item['recall']:.4f} fn_rate={item['fn_rate']:.4f}"
        )
    lines.append("")
    lines.append("Size diagnosis:")
    for name, item in summary.get("by_size", {}).items():
        lines.append(f"- {name}: gt={item['gt_count']} tp={item['tp_count']} fn={item['fn_count']} fn_rate={item['fn_rate']:.4f}")
    return "\n".join(lines) + "\n"


def _advice_text(advice: dict[str, Any]) -> str:
    lines = ["Augmentation advice", ""]
    for issue in advice.get("issues", []):
        lines.append(f"- {issue.get('issue')} severity={issue.get('severity')}")
        for recommendation in issue.get("recommendations", []):
            lines.append(f"  - {recommendation}")
    if not advice.get("issues"):
        lines.append("- No dominant failure mode found.")
    lines.append("")
    lines.append("Advisor search space:")
    lines.append(json.dumps(advice.get("advisor_search_space", {}), ensure_ascii=False, indent=2))
    return "\n".join(lines) + "\n"
