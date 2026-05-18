from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml

from AutoAugment.formats.yolo import save_yolo_labels
from scripts.filter_tiled_dataset import FilterConfig, NEW_CLASS_NAMES, build_filtered_dataset


def test_filter_tiled_dataset_preserves_ok2_ok3_and_remaps_class_ids() -> None:
    work_dir = Path("outputs/tests/pytest_tmp") / f"filter_tiled_{uuid.uuid4().hex}"
    source = work_dir / "source"
    output = work_dir / "filtered"
    _write_pair(source, "train", "train_keep", [[10, 10, 30, 30], [40, 10, 60, 30]], [0, 1])
    _write_pair(source, "train", "train_empty", [[10, 10, 30, 30]], [4])
    _write_pair(source, "val", "val_keep", [[10, 10, 30, 30], [40, 10, 60, 30]], [2, 0])
    _write_pair(source, "val", "val_empty", [[10, 10, 30, 30]], [4])
    _write_data_yaml(source)

    report = build_filtered_dataset(
        source_root=source,
        output_root=output,
        source_names={
            0: "OK",
            1: "OK2",
            2: "OK3",
            3: "加强筋打伤",
            4: "定位",
            5: "开裂",
            6: "油污",
            7: "浅划伤",
            8: "漏背锡",
            9: "碰伤",
            10: "脏污",
            11: "轮廓划伤",
            12: "锡丝残留",
            13: "锡尖",
            14: "锡膏",
        },
        config=FilterConfig(),
        debug_limit=3,
    )

    assert report["summary"]["filtered_train_image_count"] == 2
    assert report["summary"]["filtered_val_image_count"] == 2
    assert report["summary"]["filtered_bbox_count"] == 2
    assert report["summary"]["deleted_class_names"] == ["OK", "定位"]
    assert (output / "data.yaml").exists()
    data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["nc"] == 13
    assert data_yaml["names"] == NEW_CLASS_NAMES
    train_label = output / "labels" / "train" / "train_keep.txt"
    val_label = output / "labels" / "val" / "val_keep.txt"
    train_lines = train_label.read_text(encoding="utf-8").splitlines()
    val_lines = val_label.read_text(encoding="utf-8").splitlines()
    assert train_lines[0].startswith("0 ")
    assert val_lines[0].startswith("1 ")
    assert train_label.exists()
    assert val_label.exists()
    assert any((output / "debug_samples").glob("*.jpg"))


def _write_pair(root: Path, split: str, stem: str, boxes: list[list[float]], labels: list[int]) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (180, 180, 180), -1)
    image_path = image_dir / f"{stem}.jpg"
    label_path = label_dir / f"{stem}.txt"
    assert cv2.imwrite(str(image_path), image)
    save_yolo_labels(label_path, np.asarray(labels, dtype=np.int64), np.asarray(boxes, dtype=np.float32), 100, 100)


def _write_data_yaml(root: Path) -> None:
    payload = {
        "path": root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "nc": 15,
        "names": [
            "OK",
            "OK2",
            "OK3",
            "加强筋打伤",
            "定位",
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
        ],
    }
    (root / "data.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
