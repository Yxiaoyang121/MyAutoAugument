from __future__ import annotations

import numpy as np

from AutoAugment.bbox.iou import bbox_iou
from AutoAugment.diagnostics.advisor import generate_augmentation_advice
from AutoAugment.diagnostics.yolo_error_analysis import assign_size_bucket, is_near_edge, match_detections, roi_quality


def test_iou_calculation_is_correct() -> None:
    iou = bbox_iou(
        np.asarray([[0, 0, 100, 100]], dtype=np.float32),
        np.asarray([[50, 50, 150, 150]], dtype=np.float32),
    )
    assert iou[0, 0] == np.float32(2500 / 17500)


def test_match_detections_counts_tp_fp_fn() -> None:
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    records = match_detections(
        gt_labels=np.asarray([0, 0], dtype=np.int64),
        gt_boxes=np.asarray([[10, 10, 30, 30], [70, 70, 90, 90]], dtype=np.float32),
        pred_labels=np.asarray([0, 0], dtype=np.int64),
        pred_boxes=np.asarray([[11, 11, 31, 31], [40, 40, 60, 60]], dtype=np.float32),
        pred_confs=np.asarray([0.9, 0.8], dtype=np.float32),
        image=image,
        image_path="sample.jpg",
        class_names={0: "defect"},
        match_iou=0.5,
    )
    types = sorted(record["error_type"] for record in records)
    assert types == ["FN", "FP", "TP"]


def test_size_bucket_boundaries() -> None:
    assert assign_size_bucket(0.0005) == "tiny"
    assert assign_size_bucket(0.005) == "small"
    assert assign_size_bucket(0.02) == "medium"
    assert assign_size_bucket(0.08) == "large"


def test_edge_target_detection() -> None:
    assert is_near_edge([1, 20, 20, 40], width=100, height=100)
    assert not is_near_edge([20, 20, 40, 40], width=100, height=100)


def test_low_contrast_roi_detection() -> None:
    image = np.full((50, 50, 3), 80, dtype=np.uint8)
    quality = roi_quality(image, [10, 10, 30, 30])
    assert quality["low_contrast"] is True
    assert quality["brightness_mean"] == 80.0


def test_generate_augmentation_advice_for_small_and_low_contrast_failures() -> None:
    summary = {
        "overall": {"gt_count": 10, "fn_count": 5, "fp_count": 0},
        "by_size": {
            "tiny": {"gt_count": 4, "fn_count": 3},
            "small": {"gt_count": 2, "fn_count": 2},
            "medium": {"gt_count": 4, "fn_count": 0},
        },
        "by_class": [],
        "by_position": {"edge": {"gt_count": 0, "fn_rate": 0.0}},
        "quality": {"false_negatives": {"count": 5, "low_contrast_rate": 0.8, "dark_rate": 0.0, "bright_rate": 0.0}},
    }
    advice = generate_augmentation_advice(summary)
    issues = {item["issue"] for item in advice["issues"]}
    assert "small_object_fn_high" in issues
    assert "low_contrast_fn_high" in issues
    search_space = advice["advisor_search_space"]
    assert search_space["operation_weights"]["clahe"] > 1.0
    assert search_space["operation_weights"]["motion_blur"] < 1.0


def test_generate_augmentation_advice_fallback_for_empty_issues_is_not_uniform() -> None:
    summary = {
        "overall": {
            "gt_count": 19,
            "tp_count": 19,
            "fp_count": 2,
            "fn_count": 0,
            "localization_weak_count": 0,
            "precision": 0.9047619047619048,
            "recall": 1.0,
        },
        "by_size": {},
        "by_class": [
            {"class_id": 4, "class_name": "4", "gt_count": 0, "tp_count": 0, "fp_count": 2, "fn_count": 0, "precision": 0.0},
        ],
        "by_position": {},
        "quality": {
            "false_negatives": {"count": 0},
            "false_positives": {"count": 2, "low_contrast_rate": 0.0, "dark_rate": 0.5, "bright_rate": 0.0},
        },
    }
    advice = generate_augmentation_advice(summary)
    assert advice["fallback_used"] is True
    assert advice["main_issues"]
    issue_names = {item["issue"] for item in advice["main_issues"]}
    assert "false_positive_precision_gap" in issue_names
    search_space = advice["advisor_search_space"]
    weights = search_space["operation_weights"]
    assert len({round(float(value), 4) for value in weights.values()}) > 1
    assert weights["gaussian_noise"] < weights["contrast"]
