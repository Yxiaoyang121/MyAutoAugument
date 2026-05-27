from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from AutoAugment.online_augmentation import OnlinePolicyAugmentor
from scripts.train_yolo_default_with_inloop_feedback import (
    build_train_command,
    build_train_kwargs,
    build_constraint_scoring,
    control_close_to_reference,
    initial_policy_state,
    is_native_no_feedback_mode,
    training_policy,
    update_policy_state,
    write_policy_history,
)


REFERENCE = {"precision": 0.7132, "recall": 0.7600, "map50": 0.7759, "map50_95": 0.5241}


def test_mutable_policy_update_affects_transform_probability() -> None:
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    labels = np.array([1], dtype=np.int64)
    boxes = np.array([[8, 8, 20, 20]], dtype=np.float32)
    augmentor = OnlinePolicyAugmentor({"operations": []}, seed=7)

    before = augmentor.apply(image, labels, boxes)
    assert before.audit["applied_ops"] == []

    augmentor.set_policy({"operations": [{"name": "brightness", "prob": 1.0, "strength": 0.5, "params": {"max_delta": 0.2}}]})
    after = augmentor.apply(image, labels, boxes)

    assert after.audit["applied_ops"][0]["name"] == "brightness"
    np.testing.assert_array_equal(after.labels, labels)
    np.testing.assert_allclose(after.bboxes, boxes)


def test_feedback_update_modifies_policy_state() -> None:
    policy = initial_policy_state()
    diagnosis = {
        "global": {"tp": 50, "fp": 20, "fn": 40},
        "diagnosis_vector": {"low_contrast_score": {"score": 0.5}},
        "issues": [{"type": "low_contrast_missed_defect"}],
    }
    metrics = {"precision": 0.65, "recall": 0.50, "map50": 0.70, "map50_95": 0.45}

    changes = update_policy_state(policy, diagnosis=diagnosis, metrics=metrics, reference=REFERENCE)
    active = training_policy(policy)

    assert changes
    assert any(op["name"] == "sharpen_mild" for op in active["operations"])
    assert all(op["name"] not in {"mosaic4", "randaugment_like", "copy_paste"} for op in active["operations"])


def test_policy_history_records_epoch_not_stage(tmp_path) -> None:
    history = [
        {
            "epoch": 5,
            "adjustments": [{"op": "gamma", "field": "prob", "before": 0.0, "after": 0.04, "reason": "low_contrast_fn_high"}],
            "copy_paste_status": "pending_object_bank_design",
        }
    ]

    write_policy_history(tmp_path, history, initial_policy_state())

    payload = json.loads((tmp_path / "policy_history.json").read_text(encoding="utf-8"))
    assert payload["history"][0]["epoch"] == 5
    assert "stage_index" not in payload["history"][0]
    assert "5,gamma,prob" in (tmp_path / "policy_history.csv").read_text(encoding="utf-8")


def test_industrial_augmentation_keeps_labels_legal() -> None:
    policy = {
        "operations": [
            {"name": "gamma", "prob": 1.0, "strength": 0.3},
            {"name": "cutout_safe", "prob": 1.0, "strength": 0.2, "params": {"max_holes": 2, "max_fraction": 0.12}},
        ]
    }
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    labels = np.array([0, 2], dtype=np.int64)
    boxes = np.array([[10, 10, 24, 24], [32, 32, 50, 50]], dtype=np.float32)

    result = OnlinePolicyAugmentor(policy, seed=3, num_classes=3).apply(image, labels, boxes)

    assert result.labels.tolist() == [0, 2]
    assert result.bboxes.shape == (2, 4)
    assert np.all(result.bboxes[:, 0] >= 0)
    assert np.all(result.bboxes[:, 2] <= 64)
    assert np.all(result.bboxes[:, 2] > result.bboxes[:, 0])


def test_copy_paste_pending_does_not_crash() -> None:
    policy = {"operations": [{"name": "copy_paste", "prob": 1.0, "strength": 1.0}]}
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    labels = np.array([1], dtype=np.int64)
    boxes = np.array([[8, 8, 20, 20]], dtype=np.float32)

    result = OnlinePolicyAugmentor(policy, seed=4).apply(image, labels, boxes)

    assert result.audit["skipped_ops"][0]["skip_reason"] == "copy_paste_pending"
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, boxes)


def test_industrial_disabled_returns_empty_training_policy() -> None:
    policy = initial_policy_state()
    update_policy_state(
        policy,
        diagnosis={"global": {"tp": 1, "fp": 0, "fn": 10}, "diagnosis_vector": {"low_contrast_score": {"score": 0.8}}},
        metrics={"precision": 0.8, "recall": 0.2, "map50": 0.5, "map50_95": 0.3},
        reference=REFERENCE,
    )
    active = training_policy(policy, enabled=False)
    assert active["operations"] == []


def test_constraint_scoring_flags_precision_or_map_drop() -> None:
    scoring = build_constraint_scoring(
        {"precision": 0.68, "recall": 0.8, "map50": 0.77, "map50_95": 0.50},
        {"precision": 0.70, "recall": 0.76, "map50": 0.78, "map50_95": 0.52},
        baseline_name="control",
    )
    assert scoring["constraint_failed"] is True
    assert "precision_drop_gt_0.01" in scoring["failure_reasons"]
    assert "map50_95_drop_gt_0.01" in scoring["failure_reasons"]


def test_control_close_thresholds() -> None:
    assert control_close_to_reference({"precision": 0.001, "recall": -0.002, "map50": 0.0, "map50_95": 0.01})
    assert not control_close_to_reference({"precision": 0.001, "recall": -0.05, "map50": 0.0, "map50_95": 0.01})


def test_no_feedback_no_industrial_uses_native_passthrough() -> None:
    args = Namespace(
        feedback_enabled=False,
        industrial_aug_enabled=False,
        model="yolo11n.pt",
        data="outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml",
        epochs=1,
        imgsz=1024,
        batch=2,
        workers=0,
        device="0",
        seed=42,
    )
    output_dir = Path("outputs/experiments/native_parity_test")

    assert is_native_no_feedback_mode(args)
    assert build_train_kwargs(args, output_dir)["data"].endswith("data.yaml")
    command = build_train_command(args, output_dir)

    assert "trainer=UltralyticsDefaultDetectionTrainer" in command
    assert "InLoopFeedbackDetectionTrainer" not in command
