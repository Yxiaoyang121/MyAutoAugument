from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.formats.yolo import save_yolo_labels
from AutoAugment.utils import YoloImageRecord, flatten_relative_stem, load_yolo_sample, resolve_yolo_train_val_records
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml


@dataclass(frozen=True)
class TileConfig:
    tile_size: int = 1024
    overlap: float = 0.2
    min_visibility: float = 0.7
    large_object_min_visibility: float = 0.9
    drop_border_truncated: bool = True
    border_margin: float = 2.0
    require_box_center_inside: bool = True
    classwise_visibility_config: str | None = None
    keep_empty_ratio: float = 0.1
    min_box_area: float = 4.0
    seed: int = 42


@dataclass
class TileCandidate:
    split: str
    source_image: str
    relative_path: str
    tile_index: int
    tile: tuple[int, int, int, int]
    labels: np.ndarray
    bboxes: np.ndarray
    original_bbox_count: int
    dropped_bbox_count: int
    dropped_bbox_reasons: dict[str, int]
    dropped_bbox_details: list[dict[str, Any]]
    retained_bbox_details: list[dict[str, Any]]
    empty: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tiled YOLO dataset from large images.")
    parser.add_argument("--dataset-root", required=True, help="Input YOLO dataset root.")
    parser.add_argument("--data-yaml", default=None, help="Input data.yaml used for class names.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output tiled YOLO dataset root. Use outputs/datasets/tiled/<dataset_id>; full datasets must not use smoke names.",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio in [0, 1).")
    parser.add_argument("--min-visibility", type=float, default=0.7)
    parser.add_argument("--large-object-min-visibility", type=float, default=0.9)
    parser.add_argument("--drop-border-truncated", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--border-margin", type=float, default=2.0)
    parser.add_argument("--require-box-center-inside", nargs="?", const=True, default=True, type=parse_bool)
    parser.add_argument("--classwise-visibility-config", default=None)
    parser.add_argument("--keep-empty-ratio", type=float, default=0.1)
    parser.add_argument("--min-box-area", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-split", type=int, default=None, help="Optional smoke-test cap per split.")
    parser.add_argument("--debug-limit", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TileConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        large_object_min_visibility=args.large_object_min_visibility,
        drop_border_truncated=bool(args.drop_border_truncated),
        border_margin=args.border_margin,
        require_box_center_inside=bool(args.require_box_center_inside),
        classwise_visibility_config=args.classwise_visibility_config,
        keep_empty_ratio=args.keep_empty_ratio,
        min_box_area=args.min_box_area,
        seed=args.seed,
    )
    validate_config(config)
    dataset_root = Path(args.dataset_root)
    data_yaml = Path(args.data_yaml) if args.data_yaml else dataset_root / "data.yaml"
    output = Path(args.output_dir)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output directory already exists, pass --overwrite: {output}")
        shutil.rmtree(output)
    class_names = load_class_names_from_data_yaml(data_yaml) if data_yaml.exists() else {}
    report = build_tiled_dataset(
        dataset_root=dataset_root,
        output_dir=output,
        class_names=class_names,
        config=config,
        max_images_per_split=args.max_images_per_split,
        debug_limit=args.debug_limit,
    )
    print("Tiled YOLO dataset built.")
    print(f"- Output: {output.resolve()}")
    print(f"- Data YAML: {(output / 'data.yaml').resolve()}")
    print(f"- Report JSON: {(output / 'tiled_dataset_report.json').resolve()}")
    print(f"- Report MD: {(output / 'tiled_dataset_report.md').resolve()}")
    print(f"- Debug tiling: {(output / 'debug_tiling').resolve()}")
    print(f"- Tiles: {report['summary']['output_tile_count']}")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def validate_config(config: TileConfig) -> None:
    if config.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if not 0.0 <= config.overlap < 1.0:
        raise ValueError("--overlap must be in [0, 1)")
    if not 0.0 <= config.min_visibility <= 1.0:
        raise ValueError("--min-visibility must be in [0, 1]")
    if not 0.0 <= config.large_object_min_visibility <= 1.0:
        raise ValueError("--large-object-min-visibility must be in [0, 1]")
    if config.border_margin < 0:
        raise ValueError("--border-margin must be non-negative")
    if not 0.0 <= config.keep_empty_ratio <= 1.0:
        raise ValueError("--keep-empty-ratio must be in [0, 1]")
    if config.min_box_area < 0:
        raise ValueError("--min-box-area must be non-negative")


def build_tiled_dataset(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    class_names: dict[int, str],
    config: TileConfig,
    max_images_per_split: int | None = None,
    debug_limit: int = 30,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split = resolve_yolo_train_val_records(dataset_root, seed=config.seed, missing_label="empty")
    rng = np.random.default_rng(config.seed)
    split_records = {
        "train": list(split.train_records),
        "val": list(split.val_records),
    }
    if max_images_per_split is not None:
        split_records = {name: records[: max(1, max_images_per_split)] for name, records in split_records.items()}

    class_rules = build_class_visibility_rules(class_names, config)
    candidates: list[TileCandidate] = []
    all_candidates_by_split: dict[str, list[TileCandidate]] = {}
    for split_name, records in split_records.items():
        split_candidates = collect_tile_candidates(records, split=split_name, config=config, class_rules=class_rules)
        all_candidates_by_split[split_name] = split_candidates
        selected = select_empty_tiles(split_candidates, rng=rng, keep_empty_ratio=config.keep_empty_ratio)
        candidates.extend(selected)

    write_tiles(candidates, output, config)
    debug_written = write_debug_tile_visualizations(
        [candidate for items in all_candidates_by_split.values() for candidate in items],
        output / "debug_tiling",
        rng=rng,
        limit=debug_limit,
        class_names=class_names,
    )
    data_yaml_payload = write_data_yaml(output / "data.yaml", output, class_names, candidates)
    report = build_report(
        dataset_root,
        output,
        config,
        split_records,
        candidates,
        all_candidates_by_split,
        class_names,
        class_rules,
        data_yaml_payload,
        debug_written,
    )
    (output / "tiled_dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report_md(output / "tiled_dataset_report.md", report)
    write_dataset_summary_md(output / "dataset_summary.md", report)
    return report


def collect_tile_candidates(
    records: list[YoloImageRecord],
    *,
    split: str,
    config: TileConfig,
    class_rules: dict[int, dict[str, Any]],
) -> list[TileCandidate]:
    candidates: list[TileCandidate] = []
    for record in records:
        sample = load_yolo_sample(record)
        image = sample["image"]
        height, width = image.shape[:2]
        labels = sample["labels"]
        bboxes = sample["bboxes"]
        tile_index = 0
        for tile in sliding_windows(width, height, config.tile_size, config.overlap):
            kept_labels, kept_boxes, dropped, dropped_reasons, dropped_details, retained_details = crop_bboxes_to_tile(
                bboxes,
                labels,
                tile,
                class_rules=class_rules,
                config=config,
                min_box_area=config.min_box_area,
            )
            candidates.append(
                TileCandidate(
                    split=split,
                    source_image=str(record.image_path.resolve()),
                    relative_path=str(record.relative_path),
                    tile_index=tile_index,
                    tile=tile,
                    labels=kept_labels,
                    bboxes=kept_boxes,
                    original_bbox_count=int(len(labels)),
                    dropped_bbox_count=int(dropped),
                    dropped_bbox_reasons=dropped_reasons,
                    dropped_bbox_details=dropped_details,
                    retained_bbox_details=retained_details,
                    empty=len(kept_labels) == 0,
                )
            )
            tile_index += 1
    return candidates


def select_empty_tiles(
    candidates: list[TileCandidate],
    *,
    rng: np.random.Generator,
    keep_empty_ratio: float,
) -> list[TileCandidate]:
    non_empty = [candidate for candidate in candidates if not candidate.empty]
    empty = [candidate for candidate in candidates if candidate.empty]
    if keep_empty_ratio <= 0.0 or not empty:
        return non_empty
    target_empty = int(math.ceil(max(1, len(non_empty)) * keep_empty_ratio))
    target_empty = min(target_empty, len(empty))
    selected_empty = list(rng.choice(empty, size=target_empty, replace=False)) if target_empty > 0 else []
    return sorted(non_empty + selected_empty, key=lambda item: (item.split, item.source_image, item.tile_index))


def write_tiles(candidates: list[TileCandidate], output: Path, config: TileConfig) -> None:
    current_source: str | None = None
    image: np.ndarray | None = None
    ordered = sorted(candidates, key=lambda item: (item.source_image, item.tile_index, item.split))
    for candidate in ordered:
        if candidate.source_image != current_source:
            image = cv2.imread(candidate.source_image, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read image: {candidate.source_image}")
            current_source = candidate.source_image
        if image is None:
            raise ValueError(f"failed to read image: {candidate.source_image}")
        x1, y1, x2, y2 = candidate.tile
        tile_image = image[y1:y2, x1:x2].copy()
        if tile_image.size == 0:
            continue
        stem = f"{flatten_relative_stem(candidate.relative_path)}_tile_{candidate.tile_index:04d}"
        image_path = output / "images" / candidate.split / f"{stem}.jpg"
        label_path = output / "labels" / candidate.split / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), tile_image):
            raise IOError(f"failed to write tile image: {image_path}")
        save_yolo_labels(label_path, candidate.labels, candidate.bboxes, x2 - x1, y2 - y1)


def sliding_windows(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    x_starts = axis_starts(width, tile_size, stride)
    y_starts = axis_starts(height, tile_size, stride)
    return [(x, y, min(width, x + tile_size), min(height, y + tile_size)) for y in y_starts for x in x_starts]


def axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if tile_size >= length:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def crop_bboxes_to_tile(
    bboxes: np.ndarray,
    labels: np.ndarray,
    tile: tuple[int, int, int, int],
    *,
    class_rules: dict[int, dict[str, Any]],
    config: TileConfig,
    min_box_area: float,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    kept_labels: list[int] = []
    kept_boxes: list[list[float]] = []
    dropped_reasons: Counter[str] = Counter()
    dropped_details: list[dict[str, Any]] = []
    retained_details: list[dict[str, Any]] = []
    x1, y1, x2, y2 = tile
    tile_width = x2 - x1
    tile_height = y2 - y1
    for label, box in zip(np.asarray(labels, dtype=np.int64), np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)):
        class_id = int(label)
        rule = class_rules.get(class_id, default_class_rule(config))
        required_visibility = float(rule["min_visibility"])
        original_area = bbox_area(box)
        if original_area <= 0:
            dropped_reasons["invalid_original_area"] += 1
            dropped_details.append(drop_detail(class_id, box, None, 0.0, "invalid_original_area", required_visibility, tile))
            continue
        intersection = np.asarray(
            [max(box[0], x1), max(box[1], y1), min(box[2], x2), min(box[3], y2)],
            dtype=np.float32,
        )
        visible_area = bbox_area(intersection)
        visibility = visible_area / original_area if original_area > 0 else 0.0
        if visible_area <= 0:
            dropped_reasons["outside_tile"] += 1
            dropped_details.append(drop_detail(class_id, box, None, visibility, "outside_tile", required_visibility, tile))
            continue
        if visible_area < min_box_area:
            dropped_reasons["below_min_area"] += 1
            dropped_details.append(drop_detail(class_id, box, intersection, visibility, "below_min_area", required_visibility, tile))
            continue
        if rule.get("require_box_center_inside", False) and not box_center_inside_tile(box, tile):
            dropped_reasons["center_outside_tile"] += 1
            dropped_details.append(drop_detail(class_id, box, intersection, visibility, "center_outside_tile", required_visibility, tile))
            continue
        border_truncated = is_border_truncated(box, intersection, margin=float(rule.get("border_margin", config.border_margin)))
        if rule.get("drop_border_truncated", False) and border_truncated:
            dropped_reasons["border_truncated"] += 1
            dropped_details.append(drop_detail(class_id, box, intersection, visibility, "border_truncated", required_visibility, tile))
            continue
        if visibility < required_visibility:
            dropped_reasons["below_min_visibility"] += 1
            dropped_details.append(drop_detail(class_id, box, intersection, visibility, "below_min_visibility", required_visibility, tile))
            continue
        out_box = intersection.copy()
        out_box[[0, 2]] -= x1
        out_box[[1, 3]] -= y1
        out_box[0] = float(np.clip(out_box[0], 0, tile_width))
        out_box[2] = float(np.clip(out_box[2], 0, tile_width))
        out_box[1] = float(np.clip(out_box[1], 0, tile_height))
        out_box[3] = float(np.clip(out_box[3], 0, tile_height))
        kept_labels.append(class_id)
        kept_boxes.append([float(value) for value in out_box])
        retained_details.append(
            {
                "class_id": class_id,
                "original_bbox": [float(value) for value in box],
                "tile_bbox": [float(value) for value in out_box],
                "source_bbox": [float(value) for value in intersection],
                "visibility": float(visibility),
                "required_visibility": required_visibility,
                "border_touching": bbox_touches_tile_boundary(out_box, tile_width, tile_height, config.border_margin),
                "border_truncated": bool(border_truncated),
                "center_inside_tile": box_center_inside_tile(box, tile),
            }
        )
    dropped = int(sum(dropped_reasons.values()))
    if not kept_boxes:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32),
            dropped,
            dict(dropped_reasons),
            dropped_details,
            retained_details,
        )
    return (
        np.asarray(kept_labels, dtype=np.int64),
        np.asarray(kept_boxes, dtype=np.float32),
        dropped,
        dict(dropped_reasons),
        dropped_details,
        retained_details,
    )


def bbox_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def default_class_rule(config: TileConfig) -> dict[str, Any]:
    return {
        "min_visibility": config.min_visibility,
        "drop_border_truncated": config.drop_border_truncated,
        "border_margin": config.border_margin,
        "require_box_center_inside": config.require_box_center_inside,
        "class_group": "defect_or_default",
    }


def build_class_visibility_rules(class_names: dict[int, str], config: TileConfig) -> dict[int, dict[str, Any]]:
    large_object_names = {"OK", "OK2", "OK3", "定位"}
    rules: dict[int, dict[str, Any]] = {}
    for class_id, name in class_names.items():
        rule = default_class_rule(config)
        if str(name) in large_object_names:
            rule["min_visibility"] = config.large_object_min_visibility
            rule["class_group"] = "large_structure"
        rules[int(class_id)] = rule
    rules.update(load_classwise_visibility_config(config.classwise_visibility_config, class_names, config))
    return rules


def load_classwise_visibility_config(
    path_value: str | None,
    class_names: dict[int, str],
    config: TileConfig,
) -> dict[int, dict[str, Any]]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"classwise visibility config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    raw_rules = payload.get("classes", payload.get("visibility", payload)) if isinstance(payload, dict) else {}
    if not isinstance(raw_rules, dict):
        raise ValueError("--classwise-visibility-config must contain a mapping")
    name_to_id = {name: class_id for class_id, name in class_names.items()}
    overrides: dict[int, dict[str, Any]] = {}
    for raw_key, raw_value in raw_rules.items():
        class_id = parse_class_rule_key(raw_key, name_to_id)
        rule = default_class_rule(config)
        if isinstance(raw_value, dict):
            if "min_visibility" in raw_value:
                rule["min_visibility"] = float(raw_value["min_visibility"])
            if "drop_border_truncated" in raw_value:
                rule["drop_border_truncated"] = parse_bool(raw_value["drop_border_truncated"])
            if "border_margin" in raw_value:
                rule["border_margin"] = float(raw_value["border_margin"])
            if "require_box_center_inside" in raw_value:
                rule["require_box_center_inside"] = parse_bool(raw_value["require_box_center_inside"])
            if "class_group" in raw_value:
                rule["class_group"] = str(raw_value["class_group"])
        else:
            rule["min_visibility"] = float(raw_value)
        overrides[class_id] = rule
    return overrides


def parse_class_rule_key(raw_key: Any, name_to_id: dict[str, int]) -> int:
    if isinstance(raw_key, int):
        return raw_key
    text = str(raw_key)
    if text.isdigit():
        return int(text)
    if text in name_to_id:
        return int(name_to_id[text])
    raise ValueError(f"classwise visibility config references unknown class: {raw_key}")


def box_center_inside_tile(box: np.ndarray, tile: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = tile
    center_x = (float(box[0]) + float(box[2])) / 2.0
    center_y = (float(box[1]) + float(box[3])) / 2.0
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def is_border_truncated(original_box: np.ndarray, clipped_box: np.ndarray, *, margin: float) -> bool:
    return (
        float(clipped_box[0]) > float(original_box[0]) + margin
        or float(clipped_box[1]) > float(original_box[1]) + margin
        or float(clipped_box[2]) < float(original_box[2]) - margin
        or float(clipped_box[3]) < float(original_box[3]) - margin
    )


def bbox_touches_tile_boundary(box: np.ndarray, tile_width: int, tile_height: int, margin: float) -> bool:
    return (
        float(box[0]) <= margin
        or float(box[1]) <= margin
        or float(tile_width) - float(box[2]) <= margin
        or float(tile_height) - float(box[3]) <= margin
    )


def drop_detail(
    class_id: int,
    original_box: np.ndarray,
    clipped_box: np.ndarray | None,
    visibility: float,
    reason: str,
    required_visibility: float,
    tile: tuple[int, int, int, int],
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "class_id": int(class_id),
        "original_bbox": [float(value) for value in original_box],
        "source_bbox": [float(value) for value in clipped_box] if clipped_box is not None else None,
        "visibility": float(visibility),
        "required_visibility": float(required_visibility),
        "reason": reason,
        "tile": list(tile),
    }
    if clipped_box is not None:
        x1, y1, x2, y2 = tile
        tile_width = x2 - x1
        tile_height = y2 - y1
        tile_box = np.asarray(clipped_box, dtype=np.float32).copy()
        tile_box[[0, 2]] -= x1
        tile_box[[1, 3]] -= y1
        detail["tile_bbox"] = [float(value) for value in tile_box]
        detail["border_touching"] = bbox_touches_tile_boundary(tile_box, tile_width, tile_height, 2.0)
    else:
        detail["tile_bbox"] = None
        detail["border_touching"] = False
    return detail


def write_debug_tiling(
    records: list[YoloImageRecord],
    selected: list[TileCandidate],
    debug_dir: Path,
    split: str,
    *,
    limit: int,
    config: TileConfig,
) -> int:
    if limit <= 0:
        return 0
    debug_dir.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list[TileCandidate]] = {}
    for candidate in selected:
        by_source.setdefault(candidate.source_image, []).append(candidate)
    written = 0
    for record in records:
        if written >= limit:
            break
        source = str(record.image_path.resolve())
        if source not in by_source:
            continue
        sample = load_yolo_sample(record)
        image = sample["image"].copy()
        for box in sample["bboxes"]:
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for candidate in by_source[source]:
            x1, y1, x2, y2 = candidate.tile
            color = (0, 180, 0) if not candidate.empty else (0, 160, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        debug_image = resize_for_debug(image)
        out = debug_dir / f"{split}_{flatten_relative_stem(record.relative_path)}_tiling.jpg"
        cv2.imwrite(str(out), debug_image)
        written += 1
    return written


def write_debug_tile_visualizations(
    candidates: list[TileCandidate],
    debug_dir: Path,
    *,
    rng: np.random.Generator,
    limit: int,
    class_names: dict[int, str],
) -> int:
    if limit <= 0 or not candidates:
        return 0
    debug_dir.mkdir(parents=True, exist_ok=True)
    retained_candidates = [candidate for candidate in candidates if candidate.retained_bbox_details]
    border_dropped = [
        candidate
        for candidate in candidates
        if any(detail.get("reason") == "border_truncated" for detail in candidate.dropped_bbox_details)
    ]
    visibility_dropped = [
        candidate
        for candidate in candidates
        if any(detail.get("reason") in {"below_min_visibility", "center_outside_tile"} for detail in candidate.dropped_bbox_details)
    ]
    chosen = pick_class_coverage_debug_candidates(candidates, limit)
    chosen.extend(sample_debug_candidates(border_dropped, min(max(1, limit // 3), max(0, limit - len(chosen))), rng))
    chosen.extend(sample_debug_candidates(visibility_dropped, min(max(1, limit // 4), max(0, limit - len(chosen))), rng))
    chosen.extend(sample_debug_candidates(retained_candidates, max(0, limit - len(chosen)), rng))
    if len(chosen) < limit:
        chosen.extend(sample_debug_candidates(candidates, max(0, limit - len(chosen)), rng))
    chosen = dedupe_candidates(chosen)[:limit]
    if len(chosen) < limit:
        chosen_keys = {(candidate.split, candidate.source_image, candidate.tile_index) for candidate in chosen}
        for candidate in candidates:
            key = (candidate.split, candidate.source_image, candidate.tile_index)
            if key in chosen_keys:
                continue
            chosen.append(candidate)
            chosen_keys.add(key)
            if len(chosen) >= limit:
                break

    written = 0
    for index, candidate in enumerate(chosen, start=1):
        image = cv2.imread(candidate.source_image, cv2.IMREAD_COLOR)
        if image is None:
            continue
        x1, y1, x2, y2 = candidate.tile
        tile_image = image[y1:y2, x1:x2].copy()
        if tile_image.size == 0:
            continue
        for detail in candidate.retained_bbox_details:
            draw_debug_box(tile_image, detail, color_for_label(int(detail["class_id"])), "keep")
        for detail in candidate.dropped_bbox_details:
            if detail.get("reason") == "outside_tile" or detail.get("tile_bbox") is None:
                continue
            reason = str(detail.get("reason", "drop"))
            color = (0, 0, 255) if reason == "border_truncated" else (0, 165, 255)
            draw_debug_box(tile_image, detail, color, reason)
        if not candidate.retained_bbox_details and not any(detail.get("tile_bbox") for detail in candidate.dropped_bbox_details):
            cv2.putText(tile_image, "empty", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 160, 255), 2, cv2.LINE_AA)
        stem = flatten_relative_stem(candidate.relative_path)[:120]
        out = debug_dir / f"{index:03d}_{candidate.split}_{stem}_tile_{candidate.tile_index:04d}.jpg"
        if cv2.imwrite(str(out), tile_image):
            written += 1
    return written


def pick_class_coverage_debug_candidates(candidates: list[TileCandidate], limit: int) -> list[TileCandidate]:
    chosen: list[TileCandidate] = []
    seen_classes: set[int] = set()
    for candidate in candidates:
        class_ids = {int(detail["class_id"]) for detail in candidate.retained_bbox_details}
        class_ids.update(int(detail["class_id"]) for detail in candidate.dropped_bbox_details if detail.get("reason") != "outside_tile")
        if class_ids - seen_classes:
            chosen.append(candidate)
            seen_classes.update(class_ids)
        if len(chosen) >= limit:
            break
    return chosen


def dedupe_candidates(candidates: list[TileCandidate]) -> list[TileCandidate]:
    seen: set[tuple[str, str, int]] = set()
    unique: list[TileCandidate] = []
    for candidate in candidates:
        key = (candidate.split, candidate.source_image, candidate.tile_index)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def draw_debug_box(image: np.ndarray, detail: dict[str, Any], color: tuple[int, int, int], prefix: str) -> None:
    box = detail.get("tile_bbox")
    if box is None:
        return
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{prefix}:{int(detail['class_id'])} v={float(detail.get('visibility', 0.0)):.2f}"
    cv2.putText(
        image,
        label,
        (max(0, x1), max(15, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def sample_debug_candidates(candidates: list[TileCandidate], count: int, rng: np.random.Generator) -> list[TileCandidate]:
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(candidates)
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indices]


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


def resize_for_debug(image: np.ndarray, max_side: int = 1800) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def write_data_yaml(path: Path, dataset_root: Path, class_names: dict[int, str], candidates: list[TileCandidate]) -> dict[str, Any]:
    max_label = -1
    for candidate in candidates:
        if len(candidate.labels):
            max_label = max(max_label, int(np.max(candidate.labels)))
    nc = max(max_label + 1, len(class_names), 1)
    payload = {
        "path": dataset_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": [str(class_names.get(index, f"class{index}")) for index in range(nc)],
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return payload


def build_report(
    dataset_root: str | Path,
    output: Path,
    config: TileConfig,
    split_records: dict[str, list[YoloImageRecord]],
    candidates: list[TileCandidate],
    all_candidates_by_split: dict[str, list[TileCandidate]],
    class_names: dict[int, str],
    class_rules: dict[int, dict[str, Any]],
    data_yaml_payload: dict[str, Any],
    debug_visualization_count: int,
) -> dict[str, Any]:
    nc = int(data_yaml_payload["nc"])
    names = [str(value) for value in data_yaml_payload["names"]]
    source_split_stats = {split: summarize_yolo_records(records, nc) for split, records in split_records.items()}
    tiled_split_stats = {
        split: summarize_tile_candidates([candidate for candidate in candidates if candidate.split == split], nc)
        for split in ["train", "val"]
    }
    all_candidate_tiles = [candidate for items in all_candidates_by_split.values() for candidate in items]
    source_class_ids = [cid for split in source_split_stats.values() for cid in split["class_ids"]]
    tiled_class_ids = [cid for split in tiled_split_stats.values() for cid in split["class_ids"]]
    selected_drop = summarize_dropped_bboxes(candidates)
    all_candidate_drop = summarize_dropped_bboxes(all_candidate_tiles)
    retained_quality = summarize_retained_bbox_quality(candidates)
    names_quality = inspect_class_names(names, nc)

    per_split: dict[str, dict[str, Any]] = {}
    for split in ["train", "val"]:
        source_stats = source_split_stats.get(split, empty_label_summary(nc))
        tiled_stats = tiled_split_stats.get(split, empty_label_summary(nc))
        per_split[split] = {
            "source_image_count": len(split_records.get(split, [])),
            "tiled_image_count": tiled_stats["image_count"],
            "candidate_tile_count_before_empty_sampling": len(all_candidates_by_split.get(split, [])),
            "empty_tile_retained_count": tiled_stats["empty_tile_count"],
            "original_bbox_count": source_stats["bbox_total"],
            "tiled_bbox_count": tiled_stats["bbox_total"],
            "dropped_bbox_count": all_candidate_drop["by_split"][split]["effective_total_excluding_outside_tile"],
            "dropped_bbox_reasons": all_candidate_drop["by_split"][split]["by_reason"],
            "original_class_counts": source_stats["class_counts"],
            "tiled_class_counts": tiled_stats["class_counts"],
            "original_invalid_row_count": source_stats["invalid_row_count"],
            "tiled_class_id_min": tiled_stats["class_id_min"],
            "tiled_class_id_max": tiled_stats["class_id_max"],
            "tiled_class_id_ge_nc": tiled_stats["class_id_ge_nc"],
        }
    per_class = []
    for class_id in range(nc):
        original_train = int(source_split_stats["train"]["class_counts"].get(str(class_id), 0))
        original_val = int(source_split_stats["val"]["class_counts"].get(str(class_id), 0))
        tiled_train = int(tiled_split_stats["train"]["class_counts"].get(str(class_id), 0))
        tiled_val = int(tiled_split_stats["val"]["class_counts"].get(str(class_id), 0))
        tiled_total = tiled_train + tiled_val
        class_drop = all_candidate_drop["by_class"].get(str(class_id), {})
        dropped_effective = int(class_drop.get("effective_total_excluding_outside_tile", 0))
        retention_denominator = tiled_total + dropped_effective
        per_class.append(
            {
                "class_id": class_id,
                "name": names[class_id] if class_id < len(names) else class_names.get(class_id, f"class{class_id}"),
                "original_train_instances": original_train,
                "original_val_instances": original_val,
                "original_total_instances": original_train + original_val,
                "tiled_train_instances": tiled_train,
                "tiled_val_instances": tiled_val,
                "tiled_total_instances": tiled_total,
                "dropped_instances": dropped_effective,
                "visibility_failed_instances": int(class_drop.get("visibility_failed_total", 0)),
                "dropped_reasons": class_drop.get("by_reason", {}),
                "retention_rate": tiled_total / retention_denominator if retention_denominator else 0.0,
                "required_visibility": class_rules.get(class_id, default_class_rule(config))["min_visibility"],
            }
        )

    original_bbox_count = sum(item["bbox_total"] for item in source_split_stats.values())
    tiled_bbox_count = sum(item["bbox_total"] for item in tiled_split_stats.values())
    source_ge_nc = sorted({cid for cid in source_class_ids if cid >= nc})
    tiled_ge_nc = sorted({cid for cid in tiled_class_ids if cid >= nc})
    source_negative = sorted({cid for cid in source_class_ids if cid < 0})
    tiled_negative = sorted({cid for cid in tiled_class_ids if cid < 0})
    summary = {
        "source_dataset": str(Path(dataset_root).resolve()),
        "output_dataset": str(output.resolve()),
        "data_yaml": str((output / "data.yaml").resolve()),
        "debug_tiling_dir": str((output / "debug_tiling").resolve()),
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "min_visibility": config.min_visibility,
        "large_object_min_visibility": config.large_object_min_visibility,
        "drop_border_truncated": config.drop_border_truncated,
        "border_margin": config.border_margin,
        "require_box_center_inside": config.require_box_center_inside,
        "classwise_visibility_config": config.classwise_visibility_config,
        "keep_empty_ratio": config.keep_empty_ratio,
        "seed": config.seed,
        "source_image_count": sum(len(records) for records in split_records.values()),
        "original_train_image_count": len(split_records.get("train", [])),
        "original_val_image_count": len(split_records.get("val", [])),
        "output_tile_count": len(candidates),
        "tiled_train_image_count": tiled_split_stats["train"]["image_count"],
        "tiled_val_image_count": tiled_split_stats["val"]["image_count"],
        "empty_tile_count": sum(1 for candidate in candidates if candidate.empty),
        "empty_tile_retained_count": sum(1 for candidate in candidates if candidate.empty),
        "original_bbox_count": original_bbox_count,
        "tiled_bbox_count": tiled_bbox_count,
        "bbox_count": tiled_bbox_count,
        "dropped_bbox_count": all_candidate_drop["effective_total_excluding_outside_tile"],
        "dropped_bbox_reasons": all_candidate_drop["effective_by_reason_excluding_outside_tile"],
        "outside_tile_candidate_count": int(all_candidate_drop["by_reason"].get("outside_tile", 0)),
        "visibility_dropped_bbox_count": int(all_candidate_drop["visibility_failed_total"]),
        "primary_below_min_visibility_dropped_bbox_count": int(all_candidate_drop["effective_by_reason_excluding_outside_tile"].get("below_min_visibility", 0)),
        "border_truncated_dropped_bbox_count": int(all_candidate_drop["effective_by_reason_excluding_outside_tile"].get("border_truncated", 0)),
        "center_outside_dropped_bbox_count": int(all_candidate_drop["effective_by_reason_excluding_outside_tile"].get("center_outside_tile", 0)),
        "retained_bbox_quality": retained_quality,
        "obvious_half_target_bbox_found": bool(
            retained_quality["retained_visibility_below_required_count"] > 0
            or retained_quality["retained_border_truncated_count"] > 0
        ),
        "all_candidate_dropped_bbox_count_before_empty_sampling": all_candidate_drop["total"],
        "all_candidate_dropped_bbox_reasons_before_empty_sampling": all_candidate_drop["by_reason"],
        "class_count": nc,
        "data_yaml_nc": nc,
        "data_yaml_names": names,
        "source_class_id_min": min(source_class_ids) if source_class_ids else None,
        "source_class_id_max": max(source_class_ids) if source_class_ids else None,
        "tiled_class_id_min": min(tiled_class_ids) if tiled_class_ids else None,
        "tiled_class_id_max": max(tiled_class_ids) if tiled_class_ids else None,
        "class_id_min": min(tiled_class_ids) if tiled_class_ids else None,
        "class_id_max": max(tiled_class_ids) if tiled_class_ids else None,
        "source_class_id_ge_nc": source_ge_nc,
        "tiled_class_id_ge_nc": tiled_ge_nc,
        "class_id_ge_nc": sorted(set(source_ge_nc) | set(tiled_ge_nc)),
        "class_id_negative": sorted(set(source_negative) | set(tiled_negative)),
        "class_id_out_of_range_found": bool(source_ge_nc or tiled_ge_nc or source_negative or tiled_negative),
        "names_quality": names_quality,
        "chinese_names_damaged": names_quality["chinese_name_damage_found"],
        "debug_visualization_count": debug_visualization_count,
        "is_smoke": False,
        "can_be_formal_baseline_dataset": not bool(source_ge_nc or tiled_ge_nc or source_negative or tiled_negative or names_quality["has_problem"]),
    }
    return {
        "stage": "build_yolo_tiled_dataset",
        "status": "completed",
        "config": asdict(config),
        "summary": summary,
        "source_splits": source_split_stats,
        "tiled_splits": tiled_split_stats,
        "per_split": per_split,
        "per_class": per_class,
        "dropped_bboxes": {
            "selected_tiles": selected_drop,
            "all_candidate_tiles_before_empty_sampling": all_candidate_drop,
        },
        "data_yaml": {
            "path": str((output / "data.yaml").resolve()),
            "nc": nc,
            "names": names,
            "names_quality": names_quality,
        },
        "class_visibility_rules": {str(class_id): rule for class_id, rule in sorted(class_rules.items())},
        "tiles": [
            {
                "split": candidate.split,
                "source_image": candidate.source_image,
                "relative_path": candidate.relative_path,
                "tile_index": candidate.tile_index,
                "tile": list(candidate.tile),
                "bbox_count": int(len(candidate.labels)),
                "dropped_bbox_count": candidate.dropped_bbox_count,
                "dropped_bbox_reasons": candidate.dropped_bbox_reasons,
                "empty": candidate.empty,
            }
            for candidate in candidates
        ],
    }


def summarize_yolo_records(records: list[YoloImageRecord], nc: int) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    class_images: dict[int, set[str]] = {}
    class_ids: list[int] = []
    invalid_rows: list[dict[str, Any]] = []
    for record in records:
        if not record.label_path.exists():
            continue
        try:
            lines = record.label_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            invalid_rows.append({"file": str(record.label_path), "line": None, "issue": f"decode_error: {exc}"})
            continue
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                invalid_rows.append({"file": str(record.label_path), "line": line_number, "issue": "too_few_columns"})
                continue
            try:
                cid_float = float(parts[0])
                cid = int(cid_float)
            except ValueError:
                invalid_rows.append({"file": str(record.label_path), "line": line_number, "issue": "non_numeric_class_id"})
                continue
            if cid != cid_float:
                invalid_rows.append({"file": str(record.label_path), "line": line_number, "issue": "non_integer_class_id", "class_id": parts[0]})
            if cid < 0:
                invalid_rows.append({"file": str(record.label_path), "line": line_number, "issue": "negative_class_id", "class_id": cid})
            if cid >= nc:
                invalid_rows.append({"file": str(record.label_path), "line": line_number, "issue": "class_id_ge_nc", "class_id": cid})
            class_counts[cid] += 1
            class_images.setdefault(cid, set()).add(str(record.relative_path))
            class_ids.append(cid)
    return label_summary(
        image_count=len(records),
        class_counts=class_counts,
        class_images=class_images,
        class_ids=class_ids,
        invalid_rows=invalid_rows,
        nc=nc,
        empty_tile_count=0,
    )


def summarize_tile_candidates(candidates: list[TileCandidate], nc: int) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    class_images: dict[int, set[str]] = {}
    class_ids: list[int] = []
    for candidate in candidates:
        tile_key = f"{candidate.split}/{flatten_relative_stem(candidate.relative_path)}_tile_{candidate.tile_index:04d}"
        for label in candidate.labels:
            cid = int(label)
            class_counts[cid] += 1
            class_images.setdefault(cid, set()).add(tile_key)
            class_ids.append(cid)
    return label_summary(
        image_count=len(candidates),
        class_counts=class_counts,
        class_images=class_images,
        class_ids=class_ids,
        invalid_rows=[],
        nc=nc,
        empty_tile_count=sum(1 for candidate in candidates if candidate.empty),
    )


def label_summary(
    *,
    image_count: int,
    class_counts: Counter[int],
    class_images: dict[int, set[str]],
    class_ids: list[int],
    invalid_rows: list[dict[str, Any]],
    nc: int,
    empty_tile_count: int,
) -> dict[str, Any]:
    return {
        "image_count": image_count,
        "empty_tile_count": empty_tile_count,
        "bbox_total": int(sum(class_counts.values())),
        "class_counts": {str(key): int(class_counts.get(key, 0)) for key in range(nc)},
        "class_counts_all": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "class_image_counts": {str(key): len(class_images.get(key, set())) for key in range(nc)},
        "class_ids": class_ids,
        "class_id_min": min(class_ids) if class_ids else None,
        "class_id_max": max(class_ids) if class_ids else None,
        "class_id_ge_nc": sorted({cid for cid in class_ids if cid >= nc}),
        "class_id_negative": sorted({cid for cid in class_ids if cid < 0}),
        "invalid_row_count": len(invalid_rows),
        "invalid_rows_sample": invalid_rows[:20],
    }


def empty_label_summary(nc: int) -> dict[str, Any]:
    return label_summary(
        image_count=0,
        class_counts=Counter(),
        class_images={},
        class_ids=[],
        invalid_rows=[],
        nc=nc,
        empty_tile_count=0,
    )


def summarize_dropped_bboxes(candidates: list[TileCandidate]) -> dict[str, Any]:
    by_reason: Counter[str] = Counter()
    by_class: dict[int, Counter[str]] = {}
    visibility_failed_total = 0
    visibility_failed_by_class: Counter[int] = Counter()
    by_split = {
        "train": {"total": 0, "by_reason": {}},
        "val": {"total": 0, "by_reason": {}},
    }
    split_reason_counts = {"train": Counter(), "val": Counter()}
    for candidate in candidates:
        for detail in candidate.dropped_bbox_details:
            reason = str(detail.get("reason", "unknown"))
            class_id = int(detail.get("class_id", -1))
            by_reason[reason] += 1
            split_reason_counts.setdefault(candidate.split, Counter())[reason] += 1
            by_class.setdefault(class_id, Counter())[reason] += 1
            if reason != "outside_tile" and float(detail.get("visibility", 0.0)) < float(detail.get("required_visibility", 0.0)):
                visibility_failed_total += 1
                visibility_failed_by_class[class_id] += 1
    for split, counts in split_reason_counts.items():
        by_split[split] = {
            "total": int(sum(counts.values())),
            "by_reason": {reason: int(count) for reason, count in sorted(counts.items())},
            "effective_total_excluding_outside_tile": int(sum(count for reason, count in counts.items() if reason != "outside_tile")),
        }
    effective_by_reason = {reason: int(count) for reason, count in sorted(by_reason.items()) if reason != "outside_tile"}
    return {
        "total": int(sum(by_reason.values())),
        "by_reason": {reason: int(count) for reason, count in sorted(by_reason.items())},
        "effective_total_excluding_outside_tile": int(sum(effective_by_reason.values())),
        "effective_by_reason_excluding_outside_tile": effective_by_reason,
        "visibility_failed_total": int(visibility_failed_total),
        "by_class": {
            str(class_id): {
                "total": int(sum(counts.values())),
                "effective_total_excluding_outside_tile": int(sum(count for reason, count in counts.items() if reason != "outside_tile")),
                "visibility_failed_total": int(visibility_failed_by_class.get(class_id, 0)),
                "by_reason": {reason: int(count) for reason, count in sorted(counts.items())},
            }
            for class_id, counts in sorted(by_class.items())
            if class_id >= 0
        },
        "by_split": by_split,
    }


def summarize_retained_bbox_quality(candidates: list[TileCandidate]) -> dict[str, Any]:
    total = 0
    below_required = 0
    border_truncated = 0
    border_touching = 0
    visibility_thresholds = {threshold: 0 for threshold in [0.5, 0.7, 0.8, 0.9]}
    per_class_visibility: dict[int, list[float]] = {}
    for candidate in candidates:
        for detail in candidate.retained_bbox_details:
            total += 1
            class_id = int(detail["class_id"])
            visibility = float(detail.get("visibility", 0.0))
            per_class_visibility.setdefault(class_id, []).append(visibility)
            if visibility < float(detail.get("required_visibility", 0.0)):
                below_required += 1
            if bool(detail.get("border_truncated")):
                border_truncated += 1
            if bool(detail.get("border_touching")):
                border_touching += 1
            for threshold in visibility_thresholds:
                if visibility < threshold:
                    visibility_thresholds[threshold] += 1
    return {
        "retained_bbox_count": total,
        "retained_visibility_below_required_count": below_required,
        "retained_border_truncated_count": border_truncated,
        "retained_border_touching_count": border_touching,
        "retained_visibility_lt": {str(threshold): int(count) for threshold, count in visibility_thresholds.items()},
        "per_class_average_visibility": {
            str(class_id): float(np.mean(values)) if values else None
            for class_id, values in sorted(per_class_visibility.items())
        },
    }


def inspect_class_names(names: list[str], nc: int) -> dict[str, Any]:
    missing = [index for index in range(nc) if index >= len(names)]
    empty = [index for index, value in enumerate(names[:nc]) if not str(value).strip()]
    replacement_char = [index for index, value in enumerate(names[:nc]) if "\ufffd" in str(value)]
    mojibake_tokens = ["Ã", "Â", "å", "æ", "ç", "é", "闁", "鐎", "閸", "鈧", "脙", "脗", "锟斤拷"]
    suspicious_mojibake = [
        index
        for index, value in enumerate(names[:nc])
        if any(token in str(value) for token in mojibake_tokens)
    ]
    question_mark_in_non_ascii = [
        index
        for index, value in enumerate(names[:nc])
        if "?" in str(value) and any(ord(ch) > 127 for ch in str(value))
    ]
    contains_cjk = [
        index
        for index, value in enumerate(names[:nc])
        if any("\u4e00" <= ch <= "\u9fff" for ch in str(value))
    ]
    chinese_name_damage_found = bool(replacement_char or suspicious_mojibake or question_mark_in_non_ascii)
    return {
        "missing_name_ids": missing,
        "empty_name_ids": empty,
        "replacement_char_name_ids": replacement_char,
        "suspicious_mojibake_name_ids": sorted(set(suspicious_mojibake + question_mark_in_non_ascii)),
        "contains_cjk_name_ids": contains_cjk,
        "chinese_name_damage_found": chinese_name_damage_found,
        "has_problem": bool(missing or empty or chinese_name_damage_found),
    }


def write_report_md(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Tiled YOLO Dataset Report",
        "",
        f"- Source dataset: {summary['source_dataset']}",
        f"- Output dataset: {summary['output_dataset']}",
        f"- Data YAML: {summary['data_yaml']}",
        f"- Debug tiling: {summary['debug_tiling_dir']}",
        f"- tile_size: {summary['tile_size']}",
        f"- overlap: {summary['overlap']}",
        f"- min_visibility: {summary['min_visibility']}",
        f"- large_object_min_visibility: {summary['large_object_min_visibility']}",
        f"- drop_border_truncated: {summary['drop_border_truncated']}",
        f"- border_margin: {summary['border_margin']}",
        f"- require_box_center_inside: {summary['require_box_center_inside']}",
        f"- keep_empty_ratio: {summary['keep_empty_ratio']}",
        f"- seed: {summary['seed']}",
        f"- Original train images: {summary['original_train_image_count']}",
        f"- Original val images: {summary['original_val_image_count']}",
        f"- Tiled train images: {summary['tiled_train_image_count']}",
        f"- Tiled val images: {summary['tiled_val_image_count']}",
        f"- Original bboxes: {summary['original_bbox_count']}",
        f"- Tiled bboxes: {summary['tiled_bbox_count']}",
        f"- Dropped bboxes after intersection candidates: {summary['dropped_bbox_count']}",
        f"- Dropped bbox reasons: `{summary['dropped_bbox_reasons']}`",
        f"- Dropped for insufficient visibility: {summary['visibility_dropped_bbox_count']}",
        f"- Dropped for border truncation: {summary['border_truncated_dropped_bbox_count']}",
        f"- Dropped for center outside tile: {summary['center_outside_dropped_bbox_count']}",
        f"- Empty tiles retained: {summary['empty_tile_retained_count']}",
        f"- Debug tile visualizations: {summary['debug_visualization_count']}",
        f"- Obvious half-target bbox remains: {summary['obvious_half_target_bbox_found']}",
        f"- data.yaml nc: {summary['data_yaml_nc']}",
        f"- Tiled class id min: {summary['tiled_class_id_min']}",
        f"- Tiled class id max: {summary['tiled_class_id_max']}",
        f"- Any class id >= nc: {bool(summary['class_id_ge_nc'])}",
        f"- Any class id out of range: {summary['class_id_out_of_range_found']}",
        f"- Chinese class names damaged: {summary['chinese_names_damaged']}",
        f"- Can be formal baseline dataset: {summary['can_be_formal_baseline_dataset']}",
        "",
        "## Per Split",
        "",
        "| split | original images | tiled images | original bboxes | tiled bboxes | empty retained | dropped bboxes | dropped reasons | class id min | class id max | class id >= nc |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for split, item in report["per_split"].items():
        lines.append(
            f"| {split} | {item['source_image_count']} | {item['tiled_image_count']} | "
            f"{item['original_bbox_count']} | {item['tiled_bbox_count']} | {item['empty_tile_retained_count']} | "
            f"{item['dropped_bbox_count']} | `{item['dropped_bbox_reasons']}` | "
            f"{item['tiled_class_id_min']} | {item['tiled_class_id_max']} | {item['tiled_class_id_ge_nc']} |"
        )
    lines.extend(
        [
            "",
            "## Data YAML",
            "",
            f"- nc: {report['data_yaml']['nc']}",
            f"- names quality: `{report['data_yaml']['names_quality']}`",
            "",
            "| class id | name |",
            "|---:|---|",
        ]
    )
    for index, name in enumerate(report["data_yaml"]["names"]):
        lines.append(f"| {index} | {escape_md(name)} |")
    lines.extend(
        [
            "",
            "## Per-Class Instances",
            "",
            "| class id | name | original train | original val | original total | tiled train | tiled val | tiled total | dropped | retention rate | required visibility |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["per_class"]:
        lines.append(
            "| {class_id} | {name} | {original_train_instances} | {original_val_instances} | "
            "{original_total_instances} | {tiled_train_instances} | {tiled_val_instances} | {tiled_total_instances} | "
            "{dropped_instances} | {retention_rate:.4f} | {required_visibility} |".format(
                class_id=item["class_id"],
                name=escape_md(item["name"]),
                original_train_instances=item["original_train_instances"],
                original_val_instances=item["original_val_instances"],
                original_total_instances=item["original_total_instances"],
                tiled_train_instances=item["tiled_train_instances"],
                tiled_val_instances=item["tiled_val_instances"],
                tiled_total_instances=item["tiled_total_instances"],
                dropped_instances=item["dropped_instances"],
                retention_rate=float(item["retention_rate"]),
                required_visibility=item["required_visibility"],
            )
        )
    lines.extend(
        [
            "",
            "## Dropped BBoxes",
            "",
            "| scope | total | reasons |",
            "|---|---:|---|",
            f"| selected retained tiles | {report['dropped_bboxes']['selected_tiles']['effective_total_excluding_outside_tile']} | `{report['dropped_bboxes']['selected_tiles']['effective_by_reason_excluding_outside_tile']}` |",
            f"| all candidate tiles before empty sampling | {report['dropped_bboxes']['all_candidate_tiles_before_empty_sampling']['effective_total_excluding_outside_tile']} | `{report['dropped_bboxes']['all_candidate_tiles_before_empty_sampling']['effective_by_reason_excluding_outside_tile']}` |",
            "",
            "## Retained BBox Quality",
            "",
            f"- Retained visibility below required: {summary['retained_bbox_quality']['retained_visibility_below_required_count']}",
            f"- Retained border truncated: {summary['retained_bbox_quality']['retained_border_truncated_count']}",
            f"- Retained border touching: {summary['retained_bbox_quality']['retained_border_touching_count']}",
            f"- Retained visibility thresholds: `{summary['retained_bbox_quality']['retained_visibility_lt']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_summary_md(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Dataset Summary",
        "",
        f"- Source dataset: {summary['source_dataset']}",
        f"- Output dataset: {summary['output_dataset']}",
        f"- Original train/val images: {summary['original_train_image_count']} / {summary['original_val_image_count']}",
        f"- Tiled train/val images: {summary['tiled_train_image_count']} / {summary['tiled_val_image_count']}",
        f"- Original bboxes: {summary['original_bbox_count']}",
        f"- Tiled bboxes: {summary['tiled_bbox_count']}",
        f"- Empty tiles retained: {summary['empty_tile_retained_count']}",
        f"- Dropped bboxes after intersection candidates: {summary['dropped_bbox_count']}",
        f"- Dropped bbox reasons: `{summary['dropped_bbox_reasons']}`",
        f"- Dropped for insufficient visibility: {summary['visibility_dropped_bbox_count']}",
        f"- Dropped for border truncation: {summary['border_truncated_dropped_bbox_count']}",
        f"- Obvious half-target bbox remains: {summary['obvious_half_target_bbox_found']}",
        f"- data.yaml nc: {summary['data_yaml_nc']}",
        f"- data.yaml names: `{summary['data_yaml_names']}`",
        f"- Tiled class id min/max: {summary['tiled_class_id_min']} / {summary['tiled_class_id_max']}",
        f"- Class id >= nc: {summary['class_id_ge_nc']}",
        f"- Class id out of range found: {summary['class_id_out_of_range_found']}",
        f"- Chinese class names damaged: {summary['chinese_names_damaged']}",
        f"- Can be formal baseline dataset: {summary['can_be_formal_baseline_dataset']}",
        "",
        "## Per-Class Tiled BBoxes",
        "",
        "| class id | name | tiled train | tiled val | tiled total | dropped | retention rate |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["per_class"]:
        lines.append(
            f"| {item['class_id']} | {escape_md(item['name'])} | "
            f"{item['tiled_train_instances']} | {item['tiled_val_instances']} | {item['tiled_total_instances']} | "
            f"{item['dropped_instances']} | {float(item['retention_rate']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_md(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
