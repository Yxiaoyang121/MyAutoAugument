from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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
    min_visibility: float = 0.3
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
    parser.add_argument("--min-visibility", type=float, default=0.3)
    parser.add_argument("--keep-empty-ratio", type=float, default=0.1)
    parser.add_argument("--min-box-area", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-split", type=int, default=None, help="Optional smoke-test cap per split.")
    parser.add_argument("--debug-limit", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TileConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
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


def validate_config(config: TileConfig) -> None:
    if config.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if not 0.0 <= config.overlap < 1.0:
        raise ValueError("--overlap must be in [0, 1)")
    if not 0.0 <= config.min_visibility <= 1.0:
        raise ValueError("--min-visibility must be in [0, 1]")
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
    debug_limit: int = 12,
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

    candidates: list[TileCandidate] = []
    debug_written = 0
    for split_name, records in split_records.items():
        split_candidates = collect_tile_candidates(records, split=split_name, config=config)
        selected = select_empty_tiles(split_candidates, rng=rng, keep_empty_ratio=config.keep_empty_ratio)
        candidates.extend(selected)
        if debug_written < debug_limit:
            debug_written += write_debug_tiling(
                records,
                selected,
                output / "debug_tiling",
                split_name,
                limit=max(0, debug_limit - debug_written),
                config=config,
            )

    write_tiles(candidates, output, config)
    write_data_yaml(output / "data.yaml", output, class_names, candidates)
    report = build_report(dataset_root, output, config, split_records, candidates, class_names)
    (output / "tiled_dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report_md(output / "tiled_dataset_report.md", report)
    return report


def collect_tile_candidates(records: list[YoloImageRecord], *, split: str, config: TileConfig) -> list[TileCandidate]:
    candidates: list[TileCandidate] = []
    for record in records:
        sample = load_yolo_sample(record)
        image = sample["image"]
        height, width = image.shape[:2]
        labels = sample["labels"]
        bboxes = sample["bboxes"]
        tile_index = 0
        for tile in sliding_windows(width, height, config.tile_size, config.overlap):
            kept_labels, kept_boxes, dropped = crop_bboxes_to_tile(
                bboxes,
                labels,
                tile,
                min_visibility=config.min_visibility,
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
    image_cache: dict[str, np.ndarray] = {}
    for candidate in candidates:
        image = image_cache.get(candidate.source_image)
        if image is None:
            image = cv2.imread(candidate.source_image, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to read image: {candidate.source_image}")
            image_cache[candidate.source_image] = image
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
    min_visibility: float,
    min_box_area: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    kept_labels: list[int] = []
    kept_boxes: list[list[float]] = []
    dropped = 0
    x1, y1, x2, y2 = tile
    tile_width = x2 - x1
    tile_height = y2 - y1
    for label, box in zip(np.asarray(labels, dtype=np.int64), np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)):
        original_area = bbox_area(box)
        if original_area <= 0:
            dropped += 1
            continue
        intersection = np.asarray(
            [max(box[0], x1), max(box[1], y1), min(box[2], x2), min(box[3], y2)],
            dtype=np.float32,
        )
        visible_area = bbox_area(intersection)
        if visible_area <= 0 or visible_area / original_area < min_visibility or visible_area < min_box_area:
            dropped += 1
            continue
        out_box = intersection.copy()
        out_box[[0, 2]] -= x1
        out_box[[1, 3]] -= y1
        out_box[0] = float(np.clip(out_box[0], 0, tile_width))
        out_box[2] = float(np.clip(out_box[2], 0, tile_width))
        out_box[1] = float(np.clip(out_box[1], 0, tile_height))
        out_box[3] = float(np.clip(out_box[3], 0, tile_height))
        kept_labels.append(int(label))
        kept_boxes.append([float(value) for value in out_box])
    if not kept_boxes:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32), dropped
    return np.asarray(kept_labels, dtype=np.int64), np.asarray(kept_boxes, dtype=np.float32), dropped


def bbox_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


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


def resize_for_debug(image: np.ndarray, max_side: int = 1800) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def write_data_yaml(path: Path, dataset_root: Path, class_names: dict[int, str], candidates: list[TileCandidate]) -> None:
    max_label = -1
    for candidate in candidates:
        if len(candidate.labels):
            max_label = max(max_label, int(np.max(candidate.labels)))
    nc = max(max_label + 1, len(class_names), 1)
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {nc}",
        "names:",
    ]
    for index in range(nc):
        name = str(class_names.get(index, f"class{index}")).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  {index}: "{name}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    dataset_root: str | Path,
    output: Path,
    config: TileConfig,
    split_records: dict[str, list[YoloImageRecord]],
    candidates: list[TileCandidate],
    class_names: dict[int, str],
) -> dict[str, Any]:
    per_split: dict[str, dict[str, Any]] = {}
    for split in ["train", "val"]:
        split_candidates = [candidate for candidate in candidates if candidate.split == split]
        per_split[split] = {
            "source_image_count": len(split_records.get(split, [])),
            "tile_count": len(split_candidates),
            "empty_tile_count": sum(1 for candidate in split_candidates if candidate.empty),
            "bbox_count": sum(len(candidate.labels) for candidate in split_candidates),
            "dropped_bbox_count": sum(candidate.dropped_bbox_count for candidate in split_candidates),
        }
    summary = {
        "source_dataset": str(Path(dataset_root).resolve()),
        "output_dataset": str(output.resolve()),
        "data_yaml": str((output / "data.yaml").resolve()),
        "debug_tiling_dir": str((output / "debug_tiling").resolve()),
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "min_visibility": config.min_visibility,
        "keep_empty_ratio": config.keep_empty_ratio,
        "source_image_count": sum(len(records) for records in split_records.values()),
        "output_tile_count": len(candidates),
        "empty_tile_count": sum(1 for candidate in candidates if candidate.empty),
        "bbox_count": sum(len(candidate.labels) for candidate in candidates),
        "dropped_bbox_count": sum(candidate.dropped_bbox_count for candidate in candidates),
        "class_count": max(len(class_names), 1),
    }
    return {
        "stage": "build_yolo_tiled_dataset",
        "status": "completed",
        "config": asdict(config),
        "summary": summary,
        "per_split": per_split,
        "tiles": [
            {
                "split": candidate.split,
                "source_image": candidate.source_image,
                "relative_path": candidate.relative_path,
                "tile_index": candidate.tile_index,
                "tile": list(candidate.tile),
                "bbox_count": int(len(candidate.labels)),
                "dropped_bbox_count": candidate.dropped_bbox_count,
                "empty": candidate.empty,
            }
            for candidate in candidates
        ],
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
        f"- keep_empty_ratio: {summary['keep_empty_ratio']}",
        f"- Source images: {summary['source_image_count']}",
        f"- Output tiles: {summary['output_tile_count']}",
        f"- Empty tiles: {summary['empty_tile_count']}",
        f"- Retained bboxes: {summary['bbox_count']}",
        f"- Dropped bboxes: {summary['dropped_bbox_count']}",
        "",
        "## Per Split",
    ]
    for split, item in report["per_split"].items():
        lines.append(
            f"- {split}: source_images={item['source_image_count']} tiles={item['tile_count']} "
            f"empty={item['empty_tile_count']} bboxes={item['bbox_count']} dropped={item['dropped_bbox_count']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
