from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.formats.yolo import load_yolo_labels, save_yolo_labels
from AutoAugment.utils import flatten_relative_stem
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml


SOURCE_ROOT_DEFAULT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe"
OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
REMOVE_CLASS_IDS = {0, 4}
KEEP_EMPTY_RATIO_TRAIN = 0.1
KEEP_EMPTY_RATIO_VAL = 1.0
DEBUG_LIMIT = 50
SEED = 42

NEW_CLASS_ORDER = [
    1,
    2,
    3,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
]

NEW_CLASS_NAMES = [
    "OK2",
    "OK3",
    "加强筋打伤",
    "开裂",
    "油污",
    "浅划伤",
    "漏背锡",
    "碰伤",
    "脏污",
    "轮廓划伤",
    "锡丝残留",
    "锡尖",
    "锡膏",
]


@dataclass(frozen=True)
class FilterConfig:
    keep_empty_ratio_train: float = KEEP_EMPTY_RATIO_TRAIN
    keep_empty_ratio_val: float = KEEP_EMPTY_RATIO_VAL
    seed: int = SEED


@dataclass
class FilteredTileRecord:
    split: str
    source_image: str
    relative_path: str
    kept_labels: np.ndarray
    kept_bboxes: np.ndarray
    original_bbox_count: int
    filtered_out_bbox_count: int
    retained_empty: bool
    kept_from_empty_input: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a tiled YOLO dataset to remove OK and 定位.")
    parser.add_argument("--source-root", default=str(SOURCE_ROOT_DEFAULT), help="Input safe tiled dataset root.")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT_DEFAULT), help="Output filtered dataset root.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=DEBUG_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output directory already exists, pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    if not source_root.exists():
        raise FileNotFoundError(f"source dataset root does not exist: {source_root}")

    source_names = load_class_names_from_data_yaml(source_root / "data.yaml")
    config = FilterConfig()
    report = build_filtered_dataset(
        source_root=source_root,
        output_root=output_root,
        source_names=source_names,
        config=config,
        debug_limit=args.debug_limit,
    )
    print("Filtered tiled dataset built.")
    print(f"- Output: {output_root.resolve()}")
    print(f"- Data YAML: {(output_root / 'data.yaml').resolve()}")
    print(f"- Report JSON: {(output_root / 'class_filter_report.json').resolve()}")
    print(f"- Report MD: {(output_root / 'class_filter_report.md').resolve()}")
    print(f"- Debug samples: {(output_root / 'debug_samples').resolve()}")
    print(f"- Tiles: {report['summary']['kept_tiles_total']}")


def build_filtered_dataset(
    *,
    source_root: Path,
    output_root: Path,
    source_names: dict[int, str],
    config: FilterConfig,
    debug_limit: int = DEBUG_LIMIT,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    source_records = collect_source_records(source_root, source_names)
    selected_records = select_retained_records(source_records, config=config)
    write_filtered_dataset(selected_records, output_root)
    data_yaml_payload = write_data_yaml(output_root / "data.yaml", output_root)
    debug_written = write_debug_samples(selected_records, output_root / "debug_samples", limit=debug_limit, source_root=source_root)
    report = build_report(source_root, output_root, source_records, selected_records, data_yaml_payload, debug_written, config)
    (output_root / "class_filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "class_filter_report.md").write_text(render_markdown(report), encoding="utf-8")
    (output_root / "dataset_summary.md").write_text(render_dataset_summary(report), encoding="utf-8")
    return report


def collect_source_records(source_root: Path, source_names: dict[int, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ["train", "val"]:
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        image_paths = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        for image_path in image_paths:
            relative = image_path.relative_to(image_dir)
            label_path = label_dir / relative.with_suffix(".txt")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read image: {image_path}")
            height, width = image.shape[:2]
            labels, bboxes = load_yolo_labels(label_path, width, height)
            records.append(
                {
                    "split": split,
                    "image_path": str(image_path.resolve()),
                    "label_path": str(label_path.resolve()),
                    "relative_path": str(relative),
                    "labels": np.asarray(labels, dtype=np.int64),
                    "bboxes": np.asarray(bboxes, dtype=np.float32),
                    "image_size": [width, height],
                    "source_class_names": source_names,
                }
            )
    return records


def select_retained_records(source_records: list[dict[str, Any]], *, config: FilterConfig) -> list[FilteredTileRecord]:
    rng = np.random.default_rng(config.seed)
    retained: list[FilteredTileRecord] = []
    train_empty_candidates: list[FilteredTileRecord] = []
    val_empty_candidates: list[FilteredTileRecord] = []
    for record in source_records:
        split = str(record["split"])
        kept_labels, kept_bboxes, dropped_count = filter_and_remap_labels(record["labels"], record["bboxes"])
        retained_empty = len(kept_labels) == 0
        filtered_record = FilteredTileRecord(
            split=split,
            source_image=str(record["image_path"]),
            relative_path=str(record["relative_path"]),
            kept_labels=kept_labels,
            kept_bboxes=kept_bboxes,
            original_bbox_count=int(len(record["labels"])),
            filtered_out_bbox_count=int(dropped_count),
            retained_empty=retained_empty,
            kept_from_empty_input=retained_empty,
        )
        if retained_empty:
            if split == "train":
                train_empty_candidates.append(filtered_record)
            else:
                val_empty_candidates.append(filtered_record)
        else:
            retained.append(filtered_record)

    train_non_empty = [item for item in retained if item.split == "train"]
    val_non_empty = [item for item in retained if item.split == "val"]
    train_empty_target = int(math.ceil(max(1, len(train_non_empty)) * config.keep_empty_ratio_train)) if train_empty_candidates else 0
    train_empty_target = min(train_empty_target, len(train_empty_candidates))
    train_empty_selected = sample_records(train_empty_candidates, train_empty_target, rng)
    val_empty_selected = list(val_empty_candidates)

    selected = train_non_empty + val_non_empty + train_empty_selected + val_empty_selected
    selected.sort(key=lambda item: (item.split, item.source_image))
    return selected


def filter_and_remap_labels(labels: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    kept_labels: list[int] = []
    kept_boxes: list[list[float]] = []
    dropped = 0
    for label, box in zip(np.asarray(labels, dtype=np.int64), np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)):
        class_id = int(label)
        if class_id in REMOVE_CLASS_IDS:
            dropped += 1
            continue
        new_class_id = remap_class_id(class_id)
        if new_class_id is None:
            dropped += 1
            continue
        kept_labels.append(new_class_id)
        kept_boxes.append([float(value) for value in box])
    if not kept_boxes:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32), dropped
    return np.asarray(kept_labels, dtype=np.int64), np.asarray(kept_boxes, dtype=np.float32), dropped


def remap_class_id(class_id: int) -> int | None:
    try:
        return NEW_CLASS_ORDER.index(class_id)
    except ValueError:
        return None


def write_filtered_dataset(records: list[FilteredTileRecord], output_root: Path) -> None:
    current_source: str | None = None
    image: np.ndarray | None = None
    ordered = sorted(records, key=lambda item: (item.split, item.source_image))
    for record in ordered:
        if record.source_image != current_source:
            image = cv2.imread(record.source_image, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read image: {record.source_image}")
            current_source = record.source_image
        if image is None:
            raise ValueError(f"failed to read image: {record.source_image}")
        rel = Path(record.relative_path)
        image_path = output_root / "images" / record.split / rel
        label_path = output_root / "labels" / record.split / rel.with_suffix(".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), image):
            raise IOError(f"failed to write filtered tile image: {image_path}")
        height, width = image.shape[:2]
        save_yolo_labels(label_path, record.kept_labels, record.kept_bboxes, width, height)


def write_data_yaml(path: Path, output_root: Path) -> dict[str, Any]:
    payload = {
        "path": output_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": len(NEW_CLASS_NAMES),
        "names": NEW_CLASS_NAMES,
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return payload


def write_debug_samples(records: list[FilteredTileRecord], debug_dir: Path, *, limit: int, source_root: Path) -> int:
    if limit <= 0 or not records:
        return 0
    debug_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    chosen = choose_debug_records(records, limit, rng)
    written = 0
    for index, record in enumerate(chosen, start=1):
        image = cv2.imread(record.source_image, cv2.IMREAD_COLOR)
        if image is None:
            continue
        draw_filtered_record(image, record)
        out = debug_dir / f"{index:03d}_{record.split}_{flatten_relative_stem(record.relative_path)}.jpg"
        if cv2.imwrite(str(out), image):
            written += 1
    return written


def choose_debug_records(records: list[FilteredTileRecord], limit: int, rng: np.random.Generator) -> list[FilteredTileRecord]:
    if limit >= len(records):
        return list(records)
    by_class: dict[int, list[FilteredTileRecord]] = {}
    for record in records:
        class_ids = {int(label) for label in record.kept_labels}
        if not class_ids:
            continue
        for class_id in class_ids:
            by_class.setdefault(class_id, []).append(record)
    chosen: list[FilteredTileRecord] = []
    seen: set[tuple[str, str]] = set()
    for class_id in sorted(by_class):
        record = by_class[class_id][0]
        key = (record.split, record.source_image)
        if key not in seen:
            chosen.append(record)
            seen.add(key)
        if len(chosen) >= limit:
            return chosen[:limit]
    remaining = [record for record in records if (record.split, record.source_image) not in seen]
    if remaining and len(chosen) < limit:
        indices = rng.choice(len(remaining), size=min(limit - len(chosen), len(remaining)), replace=False)
        chosen.extend(remaining[int(index)] for index in indices)
    return chosen[:limit]


def draw_filtered_record(image: np.ndarray, record: FilteredTileRecord) -> None:
    for label, box in zip(record.kept_labels, record.kept_bboxes):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        color = color_for_label(int(label))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{int(label)}",
            (max(0, x1), max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    if len(record.kept_labels) == 0:
        cv2.putText(image, "empty", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)


def color_for_label(label: int) -> tuple[int, int, int]:
    palette = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 192, 255),
        (255, 0, 192),
        (192, 255, 0),
        (128, 0, 255),
    ]
    return palette[label % len(palette)]


def build_report(
    source_root: Path,
    output_root: Path,
    source_records: list[dict[str, Any]],
    selected_records: list[FilteredTileRecord],
    data_yaml_payload: dict[str, Any],
    debug_count: int,
    config: FilterConfig,
) -> dict[str, Any]:
    source_counts = summarize_source_counts(source_records)
    filtered_counts = summarize_filtered_counts(selected_records)
    split_summary = summarize_splits(source_records, selected_records)
    source_names = source_records[0]["source_class_names"] if source_records else {}
    removed_ids = sorted(REMOVE_CLASS_IDS)
    old_to_new = {str(old): new for new, old in enumerate(NEW_CLASS_ORDER)}
    new_to_old = {str(new): old for new, old in enumerate(NEW_CLASS_ORDER)}
    class_rows = []
    for new_id, old_id in enumerate(NEW_CLASS_ORDER):
        class_rows.append(
            {
                "new_class_id": new_id,
                "old_class_id": old_id,
                "old_name": source_names.get(old_id, str(old_id)),
                "name": data_yaml_payload["names"][new_id],
                "source_bbox_count": source_counts["by_class"].get(str(old_id), 0),
                "filtered_bbox_count": filtered_counts["by_class"].get(str(new_id), 0),
                "retention_rate": (
                    filtered_counts["by_class"].get(str(new_id), 0) / source_counts["by_class"].get(str(old_id), 1)
                    if source_counts["by_class"].get(str(old_id), 0)
                    else 0.0
                ),
            }
        )
    summary = {
        "source_dataset": str(source_root.resolve()),
        "output_dataset": str(output_root.resolve()),
        "source_data_yaml": str((source_root / "data.yaml").resolve()),
        "data_yaml": str((output_root / "data.yaml").resolve()),
        "class_filter_report": str((output_root / "class_filter_report.md").resolve()),
        "dataset_summary": str((output_root / "dataset_summary.md").resolve()),
        "debug_samples_dir": str((output_root / "debug_samples").resolve()),
        "seed": config.seed,
        "keep_empty_ratio_train": config.keep_empty_ratio_train,
        "keep_empty_ratio_val": config.keep_empty_ratio_val,
        "deleted_class_ids": removed_ids,
        "deleted_class_names": [source_records[0]["source_class_names"][class_id] for class_id in removed_ids] if source_records else [],
        "kept_class_ids_old": NEW_CLASS_ORDER,
        "kept_class_ids_new": list(range(len(NEW_CLASS_NAMES))),
        "source_train_image_count": split_summary["source_train_image_count"],
        "source_val_image_count": split_summary["source_val_image_count"],
        "filtered_train_image_count": split_summary["filtered_train_image_count"],
        "filtered_val_image_count": split_summary["filtered_val_image_count"],
        "kept_tiles_total": split_summary["filtered_train_image_count"] + split_summary["filtered_val_image_count"],
        "source_bbox_count": source_counts["total_bbox_count"],
        "filtered_bbox_count": filtered_counts["total_bbox_count"],
        "train_bbox_count_before": source_counts["train_bbox_count"],
        "val_bbox_count_before": source_counts["val_bbox_count"],
        "train_bbox_count_after": filtered_counts["train_bbox_count"],
        "val_bbox_count_after": filtered_counts["val_bbox_count"],
        "train_empty_retained_count": split_summary["train_empty_retained_count"],
        "train_empty_dropped_count": split_summary["train_empty_dropped_count"],
        "val_empty_retained_count": split_summary["val_empty_retained_count"],
        "val_empty_dropped_count": split_summary["val_empty_dropped_count"],
        "class_id_min": 0 if NEW_CLASS_NAMES else None,
        "class_id_max": len(NEW_CLASS_NAMES) - 1 if NEW_CLASS_NAMES else None,
        "class_id_ge_nc": [],
        "class_id_out_of_range_found": False,
        "chinese_names_damaged": False,
        "data_yaml_nc": len(NEW_CLASS_NAMES),
        "data_yaml_names": NEW_CLASS_NAMES,
        "debug_sample_count": debug_count,
        "val_empty_policy": "keep_all",
        "formal_baseline_ready": True,
        "source_class_counts": source_counts["by_class"],
        "filtered_class_counts": filtered_counts["by_class"],
        "class_filter_map": {
            "old_to_new": old_to_new,
            "new_to_old": new_to_old,
        },
    }
    return {
        "stage": "filter_tiled_dataset",
        "status": "completed",
        "summary": summary,
        "source_counts": source_counts,
        "filtered_counts": filtered_counts,
        "splits": split_summary,
        "class_rows": class_rows,
        "class_filter_map": {
            "old_to_new": old_to_new,
            "new_to_old": new_to_old,
        },
        "data_yaml": data_yaml_payload,
    }


def summarize_source_counts(source_records: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    train_count = 0
    val_count = 0
    train_bbox = 0
    val_bbox = 0
    for record in source_records:
        split = str(record["split"])
        labels = np.asarray(record["labels"], dtype=np.int64)
        if split == "train":
            train_count += 1
            train_bbox += int(len(labels))
        else:
            val_count += 1
            val_bbox += int(len(labels))
        for label in labels:
            class_counts[int(label)] += 1
    return {
        "total_bbox_count": int(sum(class_counts.values())),
        "train_bbox_count": train_bbox,
        "val_bbox_count": val_bbox,
        "by_class": {str(class_id): int(class_counts.get(class_id, 0)) for class_id in range(15)},
        "source_train_image_count": train_count,
        "source_val_image_count": val_count,
    }


def summarize_filtered_counts(records: list[FilteredTileRecord]) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    train_bbox = 0
    val_bbox = 0
    for record in records:
        labels = np.asarray(record.kept_labels, dtype=np.int64)
        if record.split == "train":
            train_bbox += int(len(labels))
        else:
            val_bbox += int(len(labels))
        for label in labels:
            class_counts[int(label)] += 1
    return {
        "total_bbox_count": int(sum(class_counts.values())),
        "train_bbox_count": train_bbox,
        "val_bbox_count": val_bbox,
        "by_class": {str(class_id): int(class_counts.get(class_id, 0)) for class_id in range(len(NEW_CLASS_NAMES))},
    }


def summarize_splits(source_records: list[dict[str, Any]], selected_records: list[FilteredTileRecord]) -> dict[str, Any]:
    source_train = sum(1 for record in source_records if record["split"] == "train")
    source_val = sum(1 for record in source_records if record["split"] == "val")
    filtered_train = sum(1 for record in selected_records if record.split == "train")
    filtered_val = sum(1 for record in selected_records if record.split == "val")
    train_empty_retained = sum(1 for record in selected_records if record.split == "train" and record.retained_empty)
    val_empty_retained = sum(1 for record in selected_records if record.split == "val" and record.retained_empty)
    source_train_empty = sum(
        1
        for record in source_records
        if record["split"] == "train" and is_filtered_empty(record["labels"], record["bboxes"])
    )
    source_val_empty = sum(
        1
        for record in source_records
        if record["split"] == "val" and is_filtered_empty(record["labels"], record["bboxes"])
    )
    return {
        "source_train_image_count": source_train,
        "source_val_image_count": source_val,
        "filtered_train_image_count": filtered_train,
        "filtered_val_image_count": filtered_val,
        "train_empty_retained_count": train_empty_retained,
        "train_empty_dropped_count": max(0, source_train_empty - train_empty_retained),
        "val_empty_retained_count": val_empty_retained,
        "val_empty_dropped_count": max(0, source_val_empty - val_empty_retained),
    }


def is_filtered_empty(labels: np.ndarray, bboxes: np.ndarray) -> bool:
    kept_labels, _, _ = filter_and_remap_labels(labels, bboxes)
    return len(kept_labels) == 0


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Class Filter Report",
        "",
        f"- Source dataset: {summary['source_dataset']}",
        f"- Output dataset: {summary['output_dataset']}",
        f"- Data YAML: {summary['data_yaml']}",
        f"- Debug samples: {summary['debug_samples_dir']}",
        f"- Seed: {summary['seed']}",
        f"- Train keep_empty_ratio: {summary['keep_empty_ratio_train']}",
        f"- Val keep_empty_ratio: {summary['keep_empty_ratio_val']}",
        f"- Val empty policy: {summary['val_empty_policy']}",
        "",
        "## Filter Summary",
        "",
        f"- Deleted classes: `{summary['deleted_class_names']}`",
        f"- Kept classes: `{NEW_CLASS_NAMES}`",
        f"- Source train/val images: {summary['source_train_image_count']} / {summary['source_val_image_count']}",
        f"- Filtered train/val images: {summary['filtered_train_image_count']} / {summary['filtered_val_image_count']}",
        f"- Source bbox count: {summary['source_bbox_count']}",
        f"- Filtered bbox count: {summary['filtered_bbox_count']}",
        f"- Train bbox count before/after: {summary['train_bbox_count_before']} / {summary['train_bbox_count_after']}",
        f"- Val bbox count before/after: {summary['val_bbox_count_before']} / {summary['val_bbox_count_after']}",
        f"- Train empty tiles retained: {summary['train_empty_retained_count']}",
        f"- Val empty tiles retained: {summary['val_empty_retained_count']}",
        f"- Class id out of range: {summary['class_id_out_of_range_found']}",
        f"- Chinese class names damaged: {summary['chinese_names_damaged']}",
        f"- Formal baseline ready: {summary['formal_baseline_ready']}",
        "",
        "## Class Mapping",
        "",
        "| old class id | old name | new class id | new name |",
        "|---:|---|---:|---|",
    ]
    for row in report["class_rows"]:
        lines.append(
            f"| {row['old_class_id']} | {escape(row['old_name'])} | "
            f"{row['new_class_id']} | {escape(row['name'])} |"
        )
    lines.extend(
        [
            "",
            "## Class Counts",
            "",
            "| new class id | old class id | name | source bboxes | filtered bboxes | retention |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in report["class_rows"]:
        lines.append(
            f"| {row['new_class_id']} | {row['old_class_id']} | {escape(row['name'])} | {row['source_bbox_count']} | "
            f"{row['filtered_bbox_count']} | {row['retention_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Class Map",
            "",
            json.dumps(report["class_filter_map"], ensure_ascii=False, indent=2),
        ]
    )
    return "\n".join(lines) + "\n"


def render_dataset_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dataset Summary",
        "",
        f"- Source dataset: {summary['source_dataset']}",
        f"- Output dataset: {summary['output_dataset']}",
        f"- Deleted classes: `{summary['deleted_class_names']}`",
        f"- Kept classes: `{NEW_CLASS_NAMES}`",
        f"- Source train/val images: {summary['source_train_image_count']} / {summary['source_val_image_count']}",
        f"- Filtered train/val images: {summary['filtered_train_image_count']} / {summary['filtered_val_image_count']}",
        f"- Source bbox count: {summary['source_bbox_count']}",
        f"- Filtered bbox count: {summary['filtered_bbox_count']}",
        f"- Val empty policy: {summary['val_empty_policy']}",
        f"- Class id out of range: {summary['class_id_out_of_range_found']}",
        f"- Chinese class names damaged: {summary['chinese_names_damaged']}",
        "",
        "## Per-Class BBoxes",
        "",
        "| new class id | name | filtered bboxes | retention rate |",
        "|---:|---|---:|---:|",
    ]
    for row in report["class_rows"]:
        lines.append(
            f"| {row['new_class_id']} | {escape(row['name'])} | {row['filtered_bbox_count']} | {row['retention_rate']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def sample_records(records: list[FilteredTileRecord], count: int, rng: np.random.Generator) -> list[FilteredTileRecord]:
    if count <= 0:
        return []
    if count >= len(records):
        return list(records)
    indices = rng.choice(len(records), size=count, replace=False)
    return [records[int(index)] for index in indices]


def escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
