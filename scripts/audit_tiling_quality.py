from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.formats.yolo import load_yolo_labels
from AutoAugment.utils import flatten_relative_stem


TILED_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full"
REPORT_JSON = TILED_ROOT / "tiled_dataset_report.json"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audits" / "tiling_quality"
AUDIT_JSON = AUDIT_DIR / "tiling_quality_audit.json"
AUDIT_MD = AUDIT_DIR / "tiling_quality_audit.md"
DEBUG_DIR = AUDIT_DIR / "debug_truncated_bboxes"
VISIBILITY_THRESHOLDS = [0.5, 0.7, 0.8, 0.9]
BORDER_MARGIN = 2.0


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    source_root = Path(report["summary"]["source_dataset"])
    names = load_names(TILED_ROOT / "data.yaml")
    records = collect_bbox_records(report, source_root)
    summary = summarize_records(records, names)
    debug_count = write_debug_visualizations(records, names, limit=50)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(TILED_ROOT.resolve()),
        "source_dataset": str(source_root.resolve()),
        "tiled_report_json": str(REPORT_JSON.resolve()),
        "border_margin": BORDER_MARGIN,
        "summary": {**summary, "debug_visualization_count": debug_count, "debug_visualization_dir": str(DEBUG_DIR.resolve())},
        "per_class": build_per_class(records, names),
        "unmatched_bbox_count": sum(1 for record in records if not record["matched_original"]),
        "debug_samples": [
            slim_record(record)
            for record in records
            if record["border_truncated"] or (record["touches_tile_boundary"] and record["visibility"] < 0.9)
        ][:200],
    }
    AUDIT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    AUDIT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(str(AUDIT_JSON))
    print(str(AUDIT_MD))


def load_names(path: Path) -> dict[int, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    raw = data.get("names", [])
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    return {}


def collect_bbox_records(report: dict[str, Any], source_root: Path) -> list[dict[str, Any]]:
    source_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    for tile_info in report["tiles"]:
        split = tile_info["split"]
        relative_path = Path(tile_info["relative_path"])
        tile_index = int(tile_info["tile_index"])
        tile = tuple(int(value) for value in tile_info["tile"])
        tile_width = tile[2] - tile[0]
        tile_height = tile[3] - tile[1]
        stem = f"{flatten_relative_stem(relative_path)}_tile_{tile_index:04d}"
        label_path = TILED_ROOT / "labels" / split / f"{stem}.txt"
        image_path = TILED_ROOT / "images" / split / f"{stem}.jpg"
        if not label_path.exists():
            continue
        labels, tile_boxes = load_yolo_labels(label_path, tile_width, tile_height)
        original_labels, original_boxes = load_original_labels(source_root, split, relative_path, source_cache)
        for label, tile_box in zip(labels, tile_boxes):
            class_id = int(label)
            source_box = np.asarray(
                [
                    float(tile_box[0]) + tile[0],
                    float(tile_box[1]) + tile[1],
                    float(tile_box[2]) + tile[0],
                    float(tile_box[3]) + tile[1],
                ],
                dtype=np.float32,
            )
            match = match_original_bbox(source_box, class_id, original_labels, original_boxes)
            if match is None:
                visibility = 0.0
                border_truncated = False
                original_box = None
                matched = False
            else:
                original_box = match
                visibility = clipped_visibility(source_box, original_box)
                border_truncated = is_border_truncated(original_box, source_box, BORDER_MARGIN)
                matched = True
            records.append(
                {
                    "split": split,
                    "tile_image": str(image_path),
                    "relative_path": str(relative_path),
                    "tile_index": tile_index,
                    "tile": list(tile),
                    "class_id": class_id,
                    "tile_bbox": [float(value) for value in tile_box],
                    "source_bbox": [float(value) for value in source_box],
                    "original_bbox": [float(value) for value in original_box] if original_box is not None else None,
                    "visibility": float(visibility),
                    "touches_tile_boundary": touches_tile_boundary(tile_box, tile_width, tile_height, BORDER_MARGIN),
                    "border_truncated": bool(border_truncated),
                    "matched_original": matched,
                }
            )
    return records


def load_original_labels(
    source_root: Path,
    split: str,
    relative_path: Path,
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key = (split, str(relative_path))
    if key in cache:
        return cache[key]
    image_path = source_root / "images" / split / relative_path
    label_path = source_root / "labels" / split / relative_path.with_suffix(".txt")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read source image: {image_path}")
    height, width = image.shape[:2]
    labels, boxes = load_yolo_labels(label_path, width, height)
    cache[key] = (labels, boxes)
    return labels, boxes


def match_original_bbox(
    source_box: np.ndarray,
    class_id: int,
    original_labels: np.ndarray,
    original_boxes: np.ndarray,
) -> np.ndarray | None:
    source_area = bbox_area(source_box)
    if source_area <= 0:
        return None
    best_box: np.ndarray | None = None
    best_score = 0.0
    for label, original_box in zip(original_labels, original_boxes):
        if int(label) != class_id:
            continue
        intersection = intersection_area(source_box, original_box)
        score = intersection / source_area
        if score > best_score:
            best_score = score
            best_box = np.asarray(original_box, dtype=np.float32)
    return best_box if best_score >= 0.5 else best_box


def clipped_visibility(source_box: np.ndarray, original_box: np.ndarray) -> float:
    original_area = bbox_area(original_box)
    if original_area <= 0:
        return 0.0
    return float(np.clip(intersection_area(source_box, original_box) / original_area, 0.0, 1.0))


def summarize_records(records: list[dict[str, Any]], names: dict[int, str]) -> dict[str, Any]:
    total = len(records)
    threshold_counts = {
        f"visibility_lt_{threshold}": sum(1 for record in records if float(record["visibility"]) < threshold)
        for threshold in VISIBILITY_THRESHOLDS
    }
    boundary_count = sum(1 for record in records if bool(record["touches_tile_boundary"]))
    truncated_count = sum(1 for record in records if bool(record["border_truncated"]))
    severe_truncated_count = sum(
        1 for record in records if bool(record["border_truncated"]) and float(record["visibility"]) < 0.7
    )
    return {
        "total_bbox_count": total,
        **threshold_counts,
        "touches_tile_boundary_count": boundary_count,
        "touches_tile_boundary_ratio": boundary_count / total if total else 0.0,
        "border_truncated_bbox_count": truncated_count,
        "border_truncated_bbox_ratio": truncated_count / total if total else 0.0,
        "severe_truncated_visibility_lt_0.7_count": severe_truncated_count,
        "severe_truncated_visibility_lt_0.7_ratio": severe_truncated_count / total if total else 0.0,
        "class_count": len(names),
    }


def build_per_class(records: list[dict[str, Any]], names: dict[int, str]) -> list[dict[str, Any]]:
    by_class: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_class.setdefault(int(record["class_id"]), []).append(record)
    rows: list[dict[str, Any]] = []
    for class_id in sorted(set(names) | set(by_class)):
        items = by_class.get(class_id, [])
        visibilities = [float(item["visibility"]) for item in items]
        truncated = [item for item in items if bool(item["border_truncated"])]
        rows.append(
            {
                "class_id": class_id,
                "name": names.get(class_id, str(class_id)),
                "bbox_count": len(items),
                "truncated_bbox_count": len(truncated),
                "truncated_bbox_ratio": len(truncated) / len(items) if items else 0.0,
                "touches_tile_boundary_count": sum(1 for item in items if bool(item["touches_tile_boundary"])),
                "average_visibility": float(np.mean(visibilities)) if visibilities else None,
                "visibility_lt_0.5": sum(1 for item in items if float(item["visibility"]) < 0.5),
                "visibility_lt_0.7": sum(1 for item in items if float(item["visibility"]) < 0.7),
                "visibility_lt_0.8": sum(1 for item in items if float(item["visibility"]) < 0.8),
                "visibility_lt_0.9": sum(1 for item in items if float(item["visibility"]) < 0.9),
            }
        )
    return rows


def write_debug_visualizations(records: list[dict[str, Any]], names: dict[int, str], *, limit: int) -> int:
    candidates = [
        record
        for record in records
        if bool(record["border_truncated"]) or (bool(record["touches_tile_boundary"]) and float(record["visibility"]) < 0.9)
    ]
    rng = np.random.default_rng(42)
    if len(candidates) > limit:
        indices = rng.choice(len(candidates), size=limit, replace=False)
        candidates = [candidates[int(index)] for index in indices]
    written = 0
    for index, record in enumerate(candidates, start=1):
        image = cv2.imread(record["tile_image"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in record["tile_bbox"]]
        color = (0, 0, 255) if record["border_truncated"] else (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{record['class_id']} {names.get(int(record['class_id']), '')} v={float(record['visibility']):.2f}"
        cv2.putText(image, text, (max(0, x1), max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        out = DEBUG_DIR / f"{index:03d}_{record['split']}_{Path(record['tile_image']).stem}.jpg"
        if cv2.imwrite(str(out), image):
            written += 1
    return written


def slim_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": record["split"],
        "relative_path": record["relative_path"],
        "tile_index": record["tile_index"],
        "class_id": record["class_id"],
        "visibility": record["visibility"],
        "touches_tile_boundary": record["touches_tile_boundary"],
        "border_truncated": record["border_truncated"],
        "tile_bbox": record["tile_bbox"],
        "original_bbox": record["original_bbox"],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Tiling Quality Audit",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Source dataset: `{payload['source_dataset']}`",
        f"- Tiled report JSON: `{payload['tiled_report_json']}`",
        f"- Border margin: `{payload['border_margin']}`",
        "",
        "## Summary",
        "",
        f"- Total bboxes: {summary['total_bbox_count']}",
        f"- Visibility < 0.5: {summary['visibility_lt_0.5']}",
        f"- Visibility < 0.7: {summary['visibility_lt_0.7']}",
        f"- Visibility < 0.8: {summary['visibility_lt_0.8']}",
        f"- Visibility < 0.9: {summary['visibility_lt_0.9']}",
        f"- Bboxes touching tile boundary: {summary['touches_tile_boundary_count']} ({summary['touches_tile_boundary_ratio']:.4f})",
        f"- Border-truncated bboxes: {summary['border_truncated_bbox_count']} ({summary['border_truncated_bbox_ratio']:.4f})",
        f"- Severe truncated bboxes with visibility < 0.7: {summary['severe_truncated_visibility_lt_0.7_count']} ({summary['severe_truncated_visibility_lt_0.7_ratio']:.4f})",
        f"- Debug visualizations: `{summary['debug_visualization_dir']}` ({summary['debug_visualization_count']} images)",
        f"- Unmatched bboxes: {payload['unmatched_bbox_count']}",
        "",
        "## Per Class",
        "",
        "| class id | name | bbox count | truncated | truncated ratio | touches boundary | avg visibility | vis <0.5 | vis <0.7 | vis <0.8 | vis <0.9 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_class"]:
        avg = "" if row["average_visibility"] is None else f"{row['average_visibility']:.4f}"
        lines.append(
            f"| {row['class_id']} | {escape(row['name'])} | {row['bbox_count']} | {row['truncated_bbox_count']} | "
            f"{row['truncated_bbox_ratio']:.4f} | {row['touches_tile_boundary_count']} | {avg} | "
            f"{row['visibility_lt_0.5']} | {row['visibility_lt_0.7']} | {row['visibility_lt_0.8']} | {row['visibility_lt_0.9']} |"
        )
    return "\n".join(lines) + "\n"


def touches_tile_boundary(box: np.ndarray, tile_width: int, tile_height: int, margin: float) -> bool:
    return (
        float(box[0]) <= margin
        or float(box[1]) <= margin
        or float(tile_width) - float(box[2]) <= margin
        or float(tile_height) - float(box[3]) <= margin
    )


def is_border_truncated(original_box: np.ndarray, clipped_box: np.ndarray, margin: float) -> bool:
    return (
        float(clipped_box[0]) > float(original_box[0]) + margin
        or float(clipped_box[1]) > float(original_box[1]) + margin
        or float(clipped_box[2]) < float(original_box[2]) - margin
        or float(clipped_box[3]) < float(original_box[3]) - margin
    )


def intersection_area(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
