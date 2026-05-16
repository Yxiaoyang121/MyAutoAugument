from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from AutoAugment.augmentations import apply_augmentation, get_augmentation_spec, list_augmentations
from AutoAugment.bbox.convert import xyxy_to_yolo
from AutoAugment.diagnostic_pipeline.policy_mapping import generate_candidate_policies
from AutoAugment.formats.yolo import load_yolo_labels, save_yolo_labels


DEBUG_DIR = Path("outputs/debug_copy_paste")
TEST_OUTPUT_DIR = Path("outputs/test_copy_paste")


def _make_yolo_sample(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((128, 128, 3), 45 + index * 20, dtype=np.uint8)
    labels = np.asarray([index % 2, 2], dtype=np.int64)
    bboxes = np.asarray(
        [
            [8 + index, 10, 22 + index, 25],
            [76, 78 - index, 96, 100 - index],
        ],
        dtype=np.float32,
    )
    colors = [(30, 220, 80), (30, 120, 240)]
    for box, color in zip(bboxes, colors):
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=-1)
    return image, labels, bboxes


def test_copy_paste_adds_labels_and_writes_readable_yolo_samples() -> None:
    assert "copy_paste" in list_augmentations()
    assert get_augmentation_spec("copy_paste").changes_bboxes

    images_dir = TEST_OUTPUT_DIR / "images"
    labels_dir = TEST_OUTPUT_DIR / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    for index in range(4):
        image, labels, bboxes = _make_yolo_sample(index)
        out_image, out_labels, out_bboxes = apply_augmentation(
            "copy_paste",
            image,
            labels,
            bboxes,
            params={
                "max_paste_count": 1,
                "max_overlap": 0.10,
                "max_attempts": 80,
                "class_balanced": True,
                "debug_dir": str(DEBUG_DIR),
                "debug_prefix": f"copy_paste_test_{index}",
            },
            strength=1.0,
            rng=rng,
        )

        assert out_image.shape == image.shape
        assert len(out_labels) > len(labels)
        assert len(out_labels) == len(out_bboxes)
        assert np.all(out_bboxes[:, [0, 2]] >= 0)
        assert np.all(out_bboxes[:, [0, 2]] <= image.shape[1])
        assert np.all(out_bboxes[:, [1, 3]] >= 0)
        assert np.all(out_bboxes[:, [1, 3]] <= image.shape[0])

        yolo_boxes = xyxy_to_yolo(out_bboxes, image.shape[1], image.shape[0])
        assert np.all(yolo_boxes >= 0.0)
        assert np.all(yolo_boxes <= 1.0)

        image_path = images_dir / f"sample_{index}.jpg"
        label_path = labels_dir / f"sample_{index}.txt"
        assert cv2.imwrite(str(image_path), out_image)
        save_yolo_labels(label_path, out_labels, out_bboxes, image.shape[1], image.shape[0])

        assert cv2.imread(str(image_path)) is not None
        loaded_labels, loaded_bboxes = load_yolo_labels(label_path, image.shape[1], image.shape[0])
        assert len(loaded_labels) == len(out_labels)
        assert loaded_bboxes.shape == out_bboxes.shape

    assert any(DEBUG_DIR.glob("copy_paste_test_*.jpg"))


def test_diagnostic_mapping_generates_executable_copy_paste_policies() -> None:
    diagnosis = {
        "issues": [
            {"type": "small_object_low_recall", "severity": "high", "evidence": {"small_object_recall": 0.2}},
            {"type": "class_imbalance", "severity": "medium", "evidence": {"minority_classes": [2]}},
        ]
    }
    payload = generate_candidate_policies(diagnosis, output_dir=TEST_OUTPUT_DIR / "policies", seed=11, max_policies=4)
    policies_by_issue = {tuple(policy["source_issues"]): policy for policy in payload["policies"]}

    small_policy = policies_by_issue[("small_object_low_recall",)]
    class_policy = policies_by_issue[("class_imbalance",)]
    assert "copy_paste" in {operation["name"] for operation in small_policy["operations"]}
    assert "copy_paste" in {operation["name"] for operation in class_policy["operations"]}
    assert payload["runtime_validation"]["status"] == "passed"
