from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.formats.yolo import load_yolo_labels, save_yolo_labels
from AutoAugment.utils import flatten_relative_stem, resolve_path


IMAGE_EXTENSIONS = (".bmp", ".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class TileConfig:
    mode: str
    tile_width: int = 1280
    tile_height: int = 800
    overlap: int = 200
    min_box_visibility: float = 0.5
    min_box_area: float = 4.0
    keep_empty: bool = False
    clip_boxes: bool = True


@dataclass(frozen=True)
class SplitPaths:
    train_images: Path
    train_labels: Path
    val_images: Path
    val_labels: Path


@dataclass
class TileRecord:
    split: str
    source_image_path: str
    output_image_path: str
    output_label_path: str
    tile_x1: int
    tile_y1: int
    tile_x2: int
    tile_y2: int
    original_bbox_count: int
    kept_bbox_count: int
    dropped_bbox_count: int
    empty_tile: bool


@dataclass
class TileSummary:
    mode: str
    tile_width: int
    tile_height: int
    overlap: int
    min_box_visibility: float
    min_box_area: float
    keep_empty: bool
    clip_boxes: bool
    source_dataset: str
    output_dataset: str
    original_image_count: int = 0
    output_tile_count: int = 0
    retained_bbox_count: int = 0
    dropped_bbox_count: int = 0
    empty_tile_count: int = 0
    missing_label_image_count: int = 0


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile a YOLO detection dataset before train_yolo search. "
            "This is a manual preprocessing step and is not part of RandomSearch."
        )
    )
    parser.add_argument("--dataset", required=True, help="YOLO dataset root with images/train,val and labels/train,val.")
    parser.add_argument("--output", required=True, help="Output YOLO dataset root.")
    parser.add_argument("--mode", choices=["sliding_window", "object_aware"], default="sliding_window")
    parser.add_argument("--tile-width", type=int, default=1280)
    parser.add_argument("--tile-height", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--min-box-visibility", type=float, default=0.5)
    parser.add_argument("--min-box-area", type=float, default=4.0)
    parser.add_argument("--keep-empty", type=parse_bool, default=False)
    parser.add_argument("--clip-boxes", type=parse_bool, default=True)
    parser.add_argument("--train-images", default=None)
    parser.add_argument("--train-labels", default=None)
    parser.add_argument("--val-images", default=None)
    parser.add_argument("--val-labels", default=None)
    return parser.parse_args()


def validate_config(config: TileConfig) -> None:
    if config.tile_width <= 0 or config.tile_height <= 0:
        raise ValueError("--tile-width and --tile-height must be positive")
    if config.overlap < 0:
        raise ValueError("--overlap must be non-negative")
    if config.mode == "sliding_window" and (config.overlap >= config.tile_width or config.overlap >= config.tile_height):
        raise ValueError("--overlap must be smaller than both tile dimensions")
    if not 0.0 <= config.min_box_visibility <= 1.0:
        raise ValueError("--min-box-visibility must be in [0, 1]")
    if config.min_box_area < 0:
        raise ValueError("--min-box-area must be non-negative")


def resolve_split_paths(
    dataset_root: str | Path,
    *,
    train_images: str | Path | None = None,
    train_labels: str | Path | None = None,
    val_images: str | Path | None = None,
    val_labels: str | Path | None = None,
) -> SplitPaths:
    root = resolve_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    explicit_values = [train_images, train_labels, val_images, val_labels]
    if any(value is not None for value in explicit_values):
        if not all(value is not None for value in explicit_values):
            raise ValueError("--train-images, --train-labels, --val-images and --val-labels must be provided together")
        paths = SplitPaths(
            train_images=resolve_path(train_images),  # type: ignore[arg-type]
            train_labels=resolve_path(train_labels),  # type: ignore[arg-type]
            val_images=resolve_path(val_images),  # type: ignore[arg-type]
            val_labels=resolve_path(val_labels),  # type: ignore[arg-type]
        )
    else:
        paths = SplitPaths(
            train_images=root / "images" / "train",
            train_labels=root / "labels" / "train",
            val_images=root / "images" / "val",
            val_labels=root / "labels" / "val",
        )

    _validate_split_dir(paths.train_images, "train images")
    _validate_split_dir(paths.val_images, "val images")
    _validate_label_dir(paths.train_labels, "train labels")
    _validate_label_dir(paths.val_labels, "val labels")
    return paths


def _validate_split_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} path is not a directory: {path}")


def _validate_label_dir(path: Path, name: str) -> None:
    if path.exists() and path.is_dir():
        return
    parent = path.parent.parent if path.name in {"train", "val"} else path.parent
    typo = parent / "lables" / path.name if parent.exists() else None
    hint = f" Did you mean '{typo}'?" if typo is not None and typo.exists() else ""
    raise FileNotFoundError(f"{name} directory does not exist: {path}.{hint} Labels are required for tiling.")


def list_image_paths(images_dir: Path, image_exts: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in image_exts}
    images = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in exts)
    if not images:
        raise ValueError(f"no images found under {images_dir} with extensions {sorted(exts)}")
    return images


def sliding_windows(width: int, height: int, tile_width: int, tile_height: int, overlap: int) -> list[tuple[int, int, int, int]]:
    x_starts = _axis_starts(width, tile_width, overlap)
    y_starts = _axis_starts(height, tile_height, overlap)
    return [
        (x, y, min(width, x + tile_width), min(height, y + tile_height))
        for y in y_starts
        for x in x_starts
    ]


def object_aware_windows(
    bboxes: np.ndarray,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    *,
    keep_empty: bool = False,
) -> list[tuple[int, int, int, int]]:
    if len(bboxes) == 0:
        if not keep_empty:
            return []
        return [_clamp_window(0, 0, tile_width, tile_height, width, height)]
    windows = []
    seen: set[tuple[int, int, int, int]] = set()
    for box in np.asarray(bboxes, dtype=np.float32):
        center_x = float((box[0] + box[2]) / 2.0)
        center_y = float((box[1] + box[3]) / 2.0)
        x1 = int(round(center_x - tile_width / 2.0))
        y1 = int(round(center_y - tile_height / 2.0))
        window = _clamp_window(x1, y1, tile_width, tile_height, width, height)
        if window not in seen:
            seen.add(window)
            windows.append(window)
    return windows


def crop_bboxes_to_tile(
    bboxes: np.ndarray,
    labels: np.ndarray,
    tile: tuple[int, int, int, int],
    *,
    min_box_visibility: float = 0.5,
    min_box_area: float = 4.0,
    clip_boxes: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(boxes) != len(labels_arr):
        raise ValueError("labels and bboxes must have the same length")
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32), 0

    x1, y1, x2, y2 = tile
    tile_width = x2 - x1
    tile_height = y2 - y1
    kept_boxes: list[list[float]] = []
    kept_labels: list[int] = []
    dropped = 0
    for label, box in zip(labels_arr, boxes):
        original_area = bbox_area(box)
        if original_area <= 0:
            dropped += 1
            continue
        intersection = bbox_tile_intersection(box, tile)
        visible_area = bbox_area(intersection)
        if visible_area <= 0:
            dropped += 1
            continue
        visibility = visible_area / original_area
        if visibility < min_box_visibility or visible_area < min_box_area:
            dropped += 1
            continue
        if clip_boxes:
            output_box = intersection.copy()
        else:
            if box[0] < x1 or box[1] < y1 or box[2] > x2 or box[3] > y2:
                dropped += 1
                continue
            output_box = box.copy()
        output_box[[0, 2]] -= x1
        output_box[[1, 3]] -= y1
        output_box[0] = float(np.clip(output_box[0], 0, tile_width))
        output_box[2] = float(np.clip(output_box[2], 0, tile_width))
        output_box[1] = float(np.clip(output_box[1], 0, tile_height))
        output_box[3] = float(np.clip(output_box[3], 0, tile_height))
        if bbox_area(output_box) < min_box_area:
            dropped += 1
            continue
        kept_labels.append(int(label))
        kept_boxes.append([float(value) for value in output_box])

    if not kept_boxes:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32), dropped
    return np.asarray(kept_labels, dtype=np.int64), np.asarray(kept_boxes, dtype=np.float32), dropped


def bbox_tile_intersection(bbox: np.ndarray, tile: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = tile
    box = np.asarray(bbox, dtype=np.float32)
    return np.asarray(
        [
            max(float(box[0]), float(x1)),
            max(float(box[1]), float(y1)),
            min(float(box[2]), float(x2)),
            min(float(box[3]), float(y2)),
        ],
        dtype=np.float32,
    )


def bbox_area(bbox: np.ndarray) -> float:
    box = np.asarray(bbox, dtype=np.float32)
    width = max(0.0, float(box[2] - box[0]))
    height = max(0.0, float(box[3] - box[1]))
    return width * height


def process_dataset(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    config: TileConfig,
    train_images: str | Path | None = None,
    train_labels: str | Path | None = None,
    val_images: str | Path | None = None,
    val_labels: str | Path | None = None,
) -> tuple[TileSummary, list[TileRecord]]:
    validate_config(config)
    source_root = resolve_path(dataset_root)
    output = resolve_path(output_root)
    split_paths = resolve_split_paths(
        source_root,
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
    )
    output.mkdir(parents=True, exist_ok=True)
    summary = TileSummary(
        mode=config.mode,
        tile_width=config.tile_width,
        tile_height=config.tile_height,
        overlap=config.overlap,
        min_box_visibility=config.min_box_visibility,
        min_box_area=config.min_box_area,
        keep_empty=config.keep_empty,
        clip_boxes=config.clip_boxes,
        source_dataset=str(source_root.resolve()),
        output_dataset=str(output.resolve()),
    )
    records: list[TileRecord] = []
    for split, images_dir, labels_dir in [
        ("train", split_paths.train_images, split_paths.train_labels),
        ("val", split_paths.val_images, split_paths.val_labels),
    ]:
        split_records = process_split(
            split=split,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_root=output,
            config=config,
        )
        records.extend(split_records)
    for record in records:
        summary.output_tile_count += 1
        summary.retained_bbox_count += record.kept_bbox_count
        summary.dropped_bbox_count += record.dropped_bbox_count
        summary.empty_tile_count += int(record.empty_tile)
    summary.original_image_count = len(list_image_paths(split_paths.train_images)) + len(list_image_paths(split_paths.val_images))
    summary.missing_label_image_count = _count_missing_label_images(split_paths)
    write_reports(output, summary, records)
    return summary, records


def process_split(
    *,
    split: str,
    images_dir: Path,
    labels_dir: Path,
    output_root: Path,
    config: TileConfig,
) -> list[TileRecord]:
    output_images = output_root / "images" / split
    output_labels = output_root / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    records: list[TileRecord] = []
    for image_path in list_image_paths(images_dir):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        relative = image_path.relative_to(images_dir)
        label_path = labels_dir / relative.with_suffix(".txt")
        if label_path.exists():
            labels, bboxes = load_yolo_labels(label_path, width, height)
        else:
            labels = np.zeros((0,), dtype=np.int64)
            bboxes = np.zeros((0, 4), dtype=np.float32)

        if config.mode == "sliding_window":
            windows = sliding_windows(width, height, config.tile_width, config.tile_height, config.overlap)
        elif config.mode == "object_aware":
            windows = object_aware_windows(
                bboxes,
                width,
                height,
                config.tile_width,
                config.tile_height,
                keep_empty=config.keep_empty,
            )
        else:
            raise ValueError(f"unsupported tiling mode: {config.mode}")

        tile_index = 0
        for tile in windows:
            cropped_labels, cropped_bboxes, dropped = crop_bboxes_to_tile(
                bboxes,
                labels,
                tile,
                min_box_visibility=config.min_box_visibility,
                min_box_area=config.min_box_area,
                clip_boxes=config.clip_boxes,
            )
            if len(cropped_labels) == 0 and not config.keep_empty:
                continue
            x1, y1, x2, y2 = tile
            tile_image = image[y1:y2, x1:x2].copy()
            if tile_image.size == 0:
                continue
            flattened_stem = flatten_relative_stem(relative)
            output_stem = f"{flattened_stem}_tile_{tile_index:03d}"
            tile_index += 1
            output_image = output_images / f"{output_stem}{image_path.suffix.lower()}"
            output_label = output_labels / f"{output_stem}.txt"
            if not cv2.imwrite(str(output_image), tile_image):
                raise IOError(f"failed to write image: {output_image}")
            save_yolo_labels(output_label, cropped_labels, cropped_bboxes, x2 - x1, y2 - y1)
            records.append(
                TileRecord(
                    split=split,
                    source_image_path=str(image_path.resolve()),
                    output_image_path=str(output_image.resolve()),
                    output_label_path=str(output_label.resolve()),
                    tile_x1=x1,
                    tile_y1=y1,
                    tile_x2=x2,
                    tile_y2=y2,
                    original_bbox_count=len(bboxes),
                    kept_bbox_count=len(cropped_labels),
                    dropped_bbox_count=dropped,
                    empty_tile=len(cropped_labels) == 0,
                )
            )
    return records


def write_reports(output_root: Path, summary: TileSummary, records: list[TileRecord]) -> None:
    report = {
        "summary": asdict(summary),
        "tiles": [asdict(record) for record in records],
    }
    (output_root / "tile_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_root / "tile_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()) if records else list(TileRecord.__annotations__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    write_readme(output_root, summary)


def write_readme(output_root: Path, summary: TileSummary) -> None:
    # Extremely wide images such as 8024x800 can become about 640x64 before YOLO letterboxing.
    # Tiling keeps target pixels large enough before the train_yolo random policy search starts.
    lines = [
        "Tiled YOLO dataset",
        "",
        "This directory contains a manually tiled YOLO detection dataset.",
        "Tiling is a preprocessing step, not a random augmentation policy and not part of RandomSearch.",
        "",
        "Why tiling was used:",
        "For very wide images such as 8024x800, direct YOLO training with imgsz=640 can shrink the image to roughly 640x64 before letterboxing to 640x640.",
        "That makes objects very small and can keep mAP near zero. Tiling keeps objects larger and reduces extreme aspect-ratio effects.",
        "",
        f"Source dataset: {summary.source_dataset}",
        f"Output dataset: {summary.output_dataset}",
        f"Mode: {summary.mode}",
        f"tile_width: {summary.tile_width}",
        f"tile_height: {summary.tile_height}",
        f"overlap: {summary.overlap}",
        f"min_box_visibility: {summary.min_box_visibility}",
        f"min_box_area: {summary.min_box_area}",
        f"keep_empty: {summary.keep_empty}",
        f"clip_boxes: {summary.clip_boxes}",
        f"Original image count: {summary.original_image_count}",
        f"Output tile count: {summary.output_tile_count}",
        f"Retained bbox count: {summary.retained_bbox_count}",
        f"Dropped bbox count: {summary.dropped_bbox_count}",
        f"Empty tile count: {summary.empty_tile_count}",
        f"Missing label image count: {summary.missing_label_image_count}",
        "",
        "Output layout:",
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "",
        "Next recommended train_yolo command:",
        "python examples\\run_policy_search.py ^",
        f"  --dataset {summary.output_dataset} ^",
        "  --output outputs\\my_policy_search_train_yolo_tiled ^",
        "  --trials 50 ^",
        "  --samples 100 ^",
        "  --seed 42 ^",
        "  --evaluator train_yolo ^",
        "  --metric map50 ^",
        "  --model yolov8n.pt ^",
        "  --epochs 10 ^",
        "  --imgsz 640 ^",
        "  --batch 4 ^",
        "  --hybrid-proxy-weight 0.2",
        "",
    ]
    (output_root / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def _axis_starts(length: int, tile_length: int, overlap: int) -> list[int]:
    if length <= 0:
        raise ValueError("image dimension must be positive")
    if tile_length >= length:
        return [0]
    stride = tile_length - overlap
    starts = list(range(0, max(1, length - tile_length + 1), stride))
    last = length - tile_length
    if starts[-1] != last:
        starts.append(last)
    return starts


def _clamp_window(x1: int, y1: int, tile_width: int, tile_height: int, width: int, height: int) -> tuple[int, int, int, int]:
    crop_width = min(tile_width, width)
    crop_height = min(tile_height, height)
    x1 = int(np.clip(x1, 0, max(0, width - crop_width)))
    y1 = int(np.clip(y1, 0, max(0, height - crop_height)))
    return x1, y1, x1 + crop_width, y1 + crop_height


def _count_missing_label_images(split_paths: SplitPaths) -> int:
    count = 0
    for images_dir, labels_dir in [
        (split_paths.train_images, split_paths.train_labels),
        (split_paths.val_images, split_paths.val_labels),
    ]:
        for image_path in list_image_paths(images_dir):
            relative = image_path.relative_to(images_dir)
            if not (labels_dir / relative.with_suffix(".txt")).exists():
                count += 1
    return count


def main() -> None:
    args = parse_args()
    config = TileConfig(
        mode=args.mode,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        overlap=args.overlap,
        min_box_visibility=args.min_box_visibility,
        min_box_area=args.min_box_area,
        keep_empty=args.keep_empty,
        clip_boxes=args.clip_boxes,
    )
    print(f"Input dataset: {resolve_path(args.dataset)}")
    print(f"Output dataset: {resolve_path(args.output)}")
    print(f"Mode: {config.mode}")
    summary, records = process_dataset(
        dataset_root=args.dataset,
        output_root=args.output,
        config=config,
        train_images=args.train_images,
        train_labels=args.train_labels,
        val_images=args.val_images,
        val_labels=args.val_labels,
    )
    print("Tiling completed:")
    print(f"- Original images: {summary.original_image_count}")
    print(f"- Output tiles: {summary.output_tile_count}")
    print(f"- Retained bboxes: {summary.retained_bbox_count}")
    print(f"- Dropped bboxes: {summary.dropped_bbox_count}")
    print(f"- Empty tiles: {summary.empty_tile_count}")
    print(f"- Missing label images: {summary.missing_label_image_count}")
    print(f"- Report JSON: {resolve_path(args.output) / 'tile_report.json'}")
    print(f"- Report CSV: {resolve_path(args.output) / 'tile_report.csv'}")
    print(f"- README: {resolve_path(args.output) / 'README.txt'}")
    if not records:
        print("warning: no tiles were written; consider --keep-empty true or a lower --min-box-visibility")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
