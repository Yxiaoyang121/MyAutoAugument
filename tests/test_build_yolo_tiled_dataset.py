from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml

from AutoAugment.formats.yolo import save_yolo_labels
from scripts.build_yolo_tiled_dataset import TileConfig, build_tiled_dataset


def test_build_yolo_tiled_dataset_writes_reports_data_yaml_and_debug() -> None:
    work_dir = Path("outputs/tests/pytest_tmp") / f"build_tiled_{uuid.uuid4().hex}"
    source = work_dir / "source"
    output = work_dir / "tiled"
    _write_pair(source, "train", "wide_train", [[40, 20, 80, 60], [220, 20, 260, 60]], [0, 1])
    _write_pair(source, "val", "wide_val", [[40, 20, 80, 60]], [0])
    (source / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {source.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 2",
                "names:",
                "  0: defect_a",
                "  1: defect_b",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = build_tiled_dataset(
        dataset_root=source,
        output_dir=output,
        class_names={0: "defect_a", 1: "defect_b"},
        config=TileConfig(tile_size=160, overlap=0.0, min_visibility=0.3, keep_empty_ratio=0.1),
        debug_limit=2,
    )

    assert report["summary"]["output_tile_count"] >= 3
    assert (output / "data.yaml").exists()
    data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == ["defect_a", "defect_b"]
    assert (output / "tiled_dataset_report.json").exists()
    assert (output / "tiled_dataset_report.md").exists()
    assert any((output / "debug_tiling").glob("*.jpg"))
    assert (output / "images" / "train").is_dir()
    assert (output / "labels" / "val").is_dir()


def test_build_yolo_tiled_dataset_preserves_unicode_class_names() -> None:
    work_dir = Path("outputs/tests/pytest_tmp") / f"build_tiled_unicode_{uuid.uuid4().hex}"
    source = work_dir / "source"
    output = work_dir / "tiled"
    _write_pair(source, "train", "wide_train", [[40, 20, 80, 60]], [0])
    _write_pair(source, "val", "wide_val", [[40, 20, 80, 60]], [0])

    build_tiled_dataset(
        dataset_root=source,
        output_dir=output,
        class_names={0: "加强筋打伤", 1: "锡膏"},
        config=TileConfig(tile_size=160, overlap=0.0, min_visibility=0.3, keep_empty_ratio=0.1),
        debug_limit=0,
    )

    data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == ["加强筋打伤", "锡膏"]


def _write_pair(root: Path, split: str, stem: str, boxes: list[list[float]], labels: list[int]) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((100, 320, 3), 70, dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (180, 180, 180), -1)
    image_path = image_dir / f"{stem}.jpg"
    label_path = label_dir / f"{stem}.txt"
    assert cv2.imwrite(str(image_path), image)
    save_yolo_labels(label_path, np.asarray(labels, dtype=np.int64), np.asarray(boxes, dtype=np.float32), 320, 100)
