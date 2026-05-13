from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from AutoAugment.formats.yolo import load_yolo_labels, save_yolo_labels


def fresh_dir(name: str) -> Path:
    path = Path("outputs/tests/pytest_tmp") / f"{name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_yolo_labels_read_save_round_trip() -> None:
    work_dir = fresh_dir("yolo_round_trip")
    label_path = work_dir / "sample.txt"
    labels = np.asarray([0, 2], dtype=np.int64)
    bboxes = np.asarray([[10, 20, 30, 60], [50, 10, 90, 40]], dtype=np.float32)
    save_yolo_labels(label_path, labels, bboxes, image_width=100, image_height=80)
    loaded_labels, loaded_bboxes = load_yolo_labels(label_path, image_width=100, image_height=80)
    np.testing.assert_array_equal(loaded_labels, labels)
    np.testing.assert_allclose(loaded_bboxes, bboxes, atol=1e-4)


def test_missing_yolo_label_returns_empty() -> None:
    work_dir = fresh_dir("yolo_missing")
    labels, bboxes = load_yolo_labels(work_dir / "missing.txt", image_width=100, image_height=100)
    assert labels.shape == (0,)
    assert bboxes.shape == (0, 4)
