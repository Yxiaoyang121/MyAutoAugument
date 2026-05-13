from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

from AutoAugment.augmentations import list_augmentations
from AutoAugment.bbox.convert import xyxy_to_yolo, yolo_to_xyxy
from AutoAugment.formats.yolo import load_yolo_labels, save_yolo_labels
from AutoAugment.policies.search_space import default_detection_search_space
from tools.tile_yolo_dataset import (
    TileConfig,
    bbox_area,
    bbox_tile_intersection,
    crop_bboxes_to_tile,
    object_aware_windows,
    process_dataset,
    resolve_split_paths,
    sliding_windows,
)


def test_yolo_xyxy_round_trip() -> None:
    yolo = np.asarray([[0.5, 0.5, 0.25, 0.4]], dtype=np.float32)
    xyxy = yolo_to_xyxy(yolo, width=200, height=100)
    np.testing.assert_allclose(xyxy, np.asarray([[75, 30, 125, 70]], dtype=np.float32))
    np.testing.assert_allclose(xyxy_to_yolo(xyxy, width=200, height=100), yolo)


def test_bbox_tile_intersection_and_visibility() -> None:
    box = np.asarray([40, 10, 140, 90], dtype=np.float32)
    tile = (0, 0, 100, 100)
    intersection = bbox_tile_intersection(box, tile)
    np.testing.assert_allclose(intersection, np.asarray([40, 10, 100, 90], dtype=np.float32))
    assert bbox_area(intersection) / bbox_area(box) == pytest.approx(0.6)


def test_crop_bboxes_drops_low_visibility_box() -> None:
    labels = np.asarray([0], dtype=np.int64)
    bboxes = np.asarray([[90, 10, 190, 90]], dtype=np.float32)
    out_labels, out_boxes, dropped = crop_bboxes_to_tile(
        bboxes,
        labels,
        (0, 0, 100, 100),
        min_box_visibility=0.5,
        min_box_area=4,
    )
    assert out_labels.shape == (0,)
    assert out_boxes.shape == (0, 4)
    assert dropped == 1


def test_sliding_window_generates_multiple_tiles() -> None:
    windows = sliding_windows(width=320, height=100, tile_width=160, tile_height=100, overlap=0)
    assert windows == [(0, 0, 160, 100), (160, 0, 320, 100)]


def test_object_aware_generates_tile_containing_target() -> None:
    bboxes = np.asarray([[170, 20, 210, 60]], dtype=np.float32)
    windows = object_aware_windows(bboxes, width=320, height=100, tile_width=160, tile_height=100)
    assert windows == [(110, 0, 270, 100)]


def test_process_dataset_sliding_window_outputs_train_val_structure() -> None:
    work_dir = fresh_dir("tile_structure")
    dataset = work_dir / "source"
    output = work_dir / "tiled"
    _write_pair(dataset, "train", "wide", [[40, 20, 80, 60], [220, 20, 260, 60]], [0, 1])
    _write_pair(dataset, "val", "wide_val", [[40, 20, 80, 60]], [0])

    summary, records = process_dataset(
        dataset_root=dataset,
        output_root=output,
        config=TileConfig(mode="sliding_window", tile_width=160, tile_height=100, overlap=0),
    )

    assert summary.original_image_count == 2
    assert summary.output_tile_count == 3
    assert (output / "images" / "train").is_dir()
    assert (output / "labels" / "train").is_dir()
    assert (output / "images" / "val").is_dir()
    assert (output / "labels" / "val").is_dir()
    assert (output / "tile_report.json").exists()
    assert (output / "tile_report.csv").exists()
    assert (output / "README.txt").exists()
    assert len(records) == 3


def test_output_images_and_labels_match_and_coords_are_normalized() -> None:
    work_dir = fresh_dir("tile_coords")
    dataset = work_dir / "source"
    output = work_dir / "tiled"
    _write_pair(dataset, "train", "wide", [[40, 20, 80, 60]], [0])
    _write_pair(dataset, "val", "wide_val", [[40, 20, 80, 60]], [0])

    process_dataset(
        dataset_root=dataset,
        output_root=output,
        config=TileConfig(mode="sliding_window", tile_width=160, tile_height=100, overlap=0),
    )

    for split in ["train", "val"]:
        image_stems = {path.stem for path in (output / "images" / split).glob("*.bmp")}
        label_stems = {path.stem for path in (output / "labels" / split).glob("*.txt")}
        assert image_stems == label_stems
        for label_path in (output / "labels" / split).glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = [float(value) for value in line.split()]
                assert len(parts) == 5
                assert all(0.0 <= value <= 1.0 for value in parts[1:])
                assert parts[3] > 0
                assert parts[4] > 0


def test_empty_tiles_are_not_saved_by_default() -> None:
    work_dir = fresh_dir("tile_no_empty")
    dataset = work_dir / "source"
    output = work_dir / "tiled"
    _write_pair(dataset, "train", "wide", [[40, 20, 80, 60]], [0])
    _write_pair(dataset, "val", "wide_val", [[40, 20, 80, 60]], [0])

    summary, _ = process_dataset(
        dataset_root=dataset,
        output_root=output,
        config=TileConfig(mode="sliding_window", tile_width=160, tile_height=100, overlap=0),
    )

    assert summary.output_tile_count == 2
    assert summary.empty_tile_count == 0


def test_keep_empty_true_saves_empty_txt() -> None:
    work_dir = fresh_dir("tile_keep_empty")
    dataset = work_dir / "source"
    output = work_dir / "tiled"
    _write_pair(dataset, "train", "wide", [[40, 20, 80, 60]], [0])
    _write_pair(dataset, "val", "wide_val", [[40, 20, 80, 60]], [0])

    summary, _ = process_dataset(
        dataset_root=dataset,
        output_root=output,
        config=TileConfig(mode="sliding_window", tile_width=160, tile_height=100, overlap=0, keep_empty=True),
    )

    assert summary.output_tile_count == 4
    assert summary.empty_tile_count == 2
    empty_labels = [path for path in (output / "labels" / "train").glob("*.txt") if path.read_text(encoding="utf-8") == ""]
    assert len(empty_labels) == 1


def test_object_aware_outputs_multi_target_tile() -> None:
    work_dir = fresh_dir("tile_object_aware")
    dataset = work_dir / "source"
    output = work_dir / "tiled"
    _write_pair(dataset, "train", "objects", [[40, 20, 60, 60], [70, 20, 90, 60]], [0, 1], width=150)
    _write_pair(dataset, "val", "objects_val", [[40, 20, 60, 60]], [0], width=150)

    summary, _ = process_dataset(
        dataset_root=dataset,
        output_root=output,
        config=TileConfig(mode="object_aware", tile_width=100, tile_height=100, min_box_visibility=0.8),
    )

    assert summary.output_tile_count >= 2
    train_labels = list((output / "labels" / "train").glob("*.txt"))
    loaded = [load_yolo_labels(path, image_width=100, image_height=100) for path in train_labels]
    assert any(labels.tolist() == [0, 1] and bboxes.shape == (2, 4) for labels, bboxes in loaded)


def test_missing_labels_directory_has_clear_lables_hint() -> None:
    work_dir = fresh_dir("tile_lables_hint")
    dataset = work_dir / "source"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "lables" / "train").mkdir(parents=True)
    (dataset / "lables" / "val").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Did you mean"):
        resolve_split_paths(dataset)


def test_tile_tool_is_not_registered_as_random_augmentation() -> None:
    names = set(list_augmentations())
    search_names = {operation.name for operation in default_detection_search_space().operations}
    assert "tile_yolo_dataset" not in names
    assert "sliding_window" not in search_names
    assert "object_aware" not in search_names


def _write_pair(
    root: Path,
    split: str,
    stem: str,
    boxes: list[list[float]],
    labels: list[int],
    *,
    width: int = 320,
    height: int = 100,
) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), 80, dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (160, 160, 160), -1)
    image_path = image_dir / f"{stem}.bmp"
    label_path = label_dir / f"{stem}.txt"
    assert cv2.imwrite(str(image_path), image)
    save_yolo_labels(
        label_path,
        np.asarray(labels, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32),
        image_width=width,
        image_height=height,
    )


def fresh_dir(name: str) -> Path:
    path = Path("outputs/tests/pytest_tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path
