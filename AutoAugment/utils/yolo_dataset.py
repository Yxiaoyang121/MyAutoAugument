from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import Iterable

import cv2
import numpy as np

from AutoAugment.formats.yolo import load_yolo_labels, save_yolo_labels


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class YoloImageRecord:
    image_path: Path
    label_path: Path
    relative_path: Path


@dataclass(frozen=True)
class YoloTrainValRecords:
    train_records: list[YoloImageRecord]
    val_records: list[YoloImageRecord]
    mode: str
    train_images_dir: Path
    train_labels_dir: Path
    val_images_dir: Path
    val_labels_dir: Path


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    return (base / path).resolve()


def _normalize_exts(image_exts: Iterable[str] | None) -> tuple[str, ...]:
    exts = image_exts or IMAGE_EXTENSIONS
    return tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in exts)


def find_yolo_records(
    dataset_root: str | Path,
    *,
    image_exts: Iterable[str] | None = None,
    missing_label: str = "empty",
) -> list[YoloImageRecord]:
    root = resolve_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    exts = _normalize_exts(image_exts)
    images_root = root / "images" if (root / "images").exists() else root
    labels_root = root / "labels" if (root / "labels").exists() else root
    records: list[YoloImageRecord] = []
    for image_path in sorted(path for path in images_root.rglob("*") if path.suffix.lower() in exts):
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        if not label_path.exists():
            if missing_label == "skip":
                continue
            if missing_label == "error":
                raise FileNotFoundError(f"missing label for image {image_path}: {label_path}")
            if missing_label != "empty":
                raise ValueError("missing_label must be one of: empty, skip, error")
        records.append(YoloImageRecord(image_path=image_path, label_path=label_path, relative_path=relative))
    if not records:
        raise ValueError(f"no images found under {images_root} with extensions {exts}")
    return records


def find_yolo_records_from_dirs(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    image_exts: Iterable[str] | None = None,
    missing_label: str = "empty",
) -> list[YoloImageRecord]:
    images_root = resolve_path(images_dir)
    labels_root = resolve_path(labels_dir)
    if not images_root.exists():
        raise FileNotFoundError(f"images directory does not exist: {images_root}")
    if not labels_root.exists() and missing_label == "error":
        raise FileNotFoundError(f"labels directory does not exist: {labels_root}")
    exts = _normalize_exts(image_exts)
    records: list[YoloImageRecord] = []
    for image_path in sorted(path for path in images_root.rglob("*") if path.suffix.lower() in exts):
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        if not label_path.exists():
            if missing_label == "skip":
                continue
            if missing_label == "error":
                raise FileNotFoundError(f"missing label for image {image_path}: {label_path}")
            if missing_label != "empty":
                raise ValueError("missing_label must be one of: empty, skip, error")
        records.append(YoloImageRecord(image_path=image_path, label_path=label_path, relative_path=relative))
    if not records:
        raise ValueError(f"no images found under {images_root} with extensions {exts}")
    return records


def resolve_yolo_train_val_records(
    dataset_root: str | Path,
    *,
    train_images: str | Path | None = None,
    train_labels: str | Path | None = None,
    val_images: str | Path | None = None,
    val_labels: str | Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    image_exts: Iterable[str] | None = None,
    missing_label: str = "empty",
) -> YoloTrainValRecords:
    root = resolve_path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    has_explicit = any(value is not None for value in [train_images, train_labels, val_images, val_labels])
    if has_explicit:
        if not all(value is not None for value in [train_images, train_labels, val_images, val_labels]):
            raise ValueError("--train-images, --train-labels, --val-images and --val-labels must be provided together")
        train_images_dir = resolve_path(train_images)  # type: ignore[arg-type]
        train_labels_dir = resolve_path(train_labels)  # type: ignore[arg-type]
        val_images_dir = resolve_path(val_images)  # type: ignore[arg-type]
        val_labels_dir = resolve_path(val_labels)  # type: ignore[arg-type]
        return YoloTrainValRecords(
            train_records=find_yolo_records_from_dirs(train_images_dir, train_labels_dir, image_exts=image_exts, missing_label=missing_label),
            val_records=find_yolo_records_from_dirs(val_images_dir, val_labels_dir, image_exts=image_exts, missing_label=missing_label),
            mode="explicit",
            train_images_dir=train_images_dir,
            train_labels_dir=train_labels_dir,
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
        )

    standard_train_images = root / "images" / "train"
    standard_val_images = root / "images" / "val"
    standard_train_labels = root / "labels" / "train"
    standard_val_labels = root / "labels" / "val"
    if standard_train_images.exists() and standard_val_images.exists():
        return YoloTrainValRecords(
            train_records=find_yolo_records_from_dirs(standard_train_images, standard_train_labels, image_exts=image_exts, missing_label=missing_label),
            val_records=find_yolo_records_from_dirs(standard_val_images, standard_val_labels, image_exts=image_exts, missing_label=missing_label),
            mode="standard_train_val",
            train_images_dir=standard_train_images,
            train_labels_dir=standard_train_labels,
            val_images_dir=standard_val_images,
            val_labels_dir=standard_val_labels,
        )

    records = find_yolo_records(root, image_exts=image_exts, missing_label=missing_label)
    if len(records) < 2:
        raise ValueError("simple YOLO dataset needs at least 2 images to split train/val")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    val_count = int(round(len(records) * val_ratio))
    val_count = min(max(1, val_count), len(records) - 1)
    val_indices = set(int(index) for index in indices[:val_count])
    train_records = [record for index, record in enumerate(records) if index not in val_indices]
    val_records = [record for index, record in enumerate(records) if index in val_indices]
    images_dir = root / "images" if (root / "images").exists() else root
    labels_dir = root / "labels" if (root / "labels").exists() else root
    return YoloTrainValRecords(
        train_records=train_records,
        val_records=val_records,
        mode="auto_split",
        train_images_dir=images_dir,
        train_labels_dir=labels_dir,
        val_images_dir=images_dir,
        val_labels_dir=labels_dir,
    )


def load_yolo_sample(record: YoloImageRecord) -> dict:
    image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {record.image_path}")
    height, width = image.shape[:2]
    if record.label_path.exists():
        labels, bboxes = load_yolo_labels(record.label_path, width, height)
    else:
        labels = np.zeros((0,), dtype=np.int64)
        bboxes = np.zeros((0, 4), dtype=np.float32)
    return {
        "image": image,
        "labels": labels,
        "bboxes": bboxes,
        "image_path": str(record.image_path),
        "label_path": str(record.label_path) if record.label_path.exists() else None,
        "relative_path": str(record.relative_path),
    }


def save_yolo_sample(sample: dict, image_path: str | Path, label_path: str | Path) -> None:
    image_path = Path(image_path)
    label_path = Path(label_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image = sample["image"]
    if not cv2.imwrite(str(image_path), image):
        raise IOError(f"failed to write image: {image_path}")
    height, width = image.shape[:2]
    save_yolo_labels(label_path, sample["labels"], sample["bboxes"], width, height)


def copy_yolo_records(records: list[YoloImageRecord], images_dir: str | Path, labels_dir: str | Path) -> None:
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        image_target = images_root / record.relative_path
        label_target = labels_root / record.relative_path.with_suffix(".txt")
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        copy2(record.image_path, image_target)
        if record.label_path.exists():
            copy2(record.label_path, label_target)
        else:
            label_target.write_text("", encoding="utf-8")


def flatten_relative_stem(relative_path: str | Path) -> str:
    relative = Path(relative_path)
    parts = list(relative.with_suffix("").parts)
    return "__".join(parts)


def sample_records(
    records: list[YoloImageRecord],
    num_samples: int | None,
    rng: np.random.Generator,
) -> list[YoloImageRecord]:
    if num_samples is None or num_samples <= 0 or num_samples >= len(records):
        return list(records)
    indices = rng.choice(len(records), size=int(num_samples), replace=False)
    return [records[int(index)] for index in indices]
