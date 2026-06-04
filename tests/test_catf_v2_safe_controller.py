from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix
from AutoAugment.catf_v2.safe_controller import CATFSafeController, force_noop_policy
from AutoAugment.catf_v2.sample_router import ROIStats, SampleAwareAugmentationRouter
from AutoAugment.online_augmentation import OnlineAugmentationStats
from scripts.train_yolo_online_aug import OnlineTrainingContext, UltralyticsOnlinePolicyTransform, instances_to_xyxy, make_instances
from ultralytics.utils.instance import Instances


def _metrics(precision=0.70, recall=0.70, map50=0.75, map50_95=0.50):
    return {"precision": precision, "recall": recall, "map50": map50, "map50_95": map50_95}


def _active_policy() -> dict:
    policy = initial_policy_matrix({6: "漏背锡", 7: "碰伤"})
    row = policy["classes"]["6"]
    row["status"] = "active"
    row["state"] = "pending"
    row["ops"]["local_contrast"]["prob"] = 0.5
    row["ops"]["local_contrast"]["strength"] = 0.3
    return policy


def test_high_clean_recall_lag_triggers_baseline_protection() -> None:
    controller = CATFSafeController()
    event = controller.evaluate(
        epoch=20,
        policy=_active_policy(),
        metrics=_metrics(recall=0.70, map50=0.77, map50_95=0.53),
        reference_metrics=_metrics(recall=0.73, map50=0.77, map50_95=0.53),
        active_classes=[6],
        proposed_action="accept",
    )

    assert event["action"] == "no_op_freeze"
    assert "high_recall_baseline_protection" in event["reasons"]
    assert all(op["prob"] == 0.0 for row in event["policy"]["classes"].values() for op in row["ops"].values())


def test_map95_lag_triggers_noop_freeze() -> None:
    controller = CATFSafeController()
    event = controller.evaluate(
        epoch=20,
        policy=_active_policy(),
        metrics=_metrics(recall=0.73, map50=0.77, map50_95=0.505),
        reference_metrics=_metrics(recall=0.73, map50=0.77, map50_95=0.52),
        active_classes=[6],
        proposed_action="accept",
    )

    assert event["action"] == "no_op_freeze"
    assert "high_map95_baseline_protection" in event["reasons"]


def test_precision_gain_with_recall_map_drop_cannot_accept() -> None:
    controller = CATFSafeController()
    event = controller.evaluate(
        epoch=25,
        policy=_active_policy(),
        metrics=_metrics(precision=0.80, recall=0.69, map50=0.744, map50_95=0.496),
        reference_metrics=_metrics(precision=0.70, recall=0.70, map50=0.75, map50_95=0.50),
        active_classes=[6],
        proposed_action="accept",
    )

    assert event["action"] == "safe_accept_blocked"
    assert event["safe_accept_allowed"] is False


def test_early_abstention_at_epoch5() -> None:
    controller = CATFSafeController()
    event = controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(),
        reference_metrics=_metrics(),
        active_classes=[6],
        proposed_action="accept",
    )

    assert event["action"] == "no_op_freeze"
    assert "early_abstention_no_recall_or_map_gain" in event["reasons"]


def test_non_active_class_regression_triggers_negative_effect_attribution() -> None:
    controller = CATFSafeController()
    first = {
        "classes": {
            "6": {"class_id": 6, "class_name": "漏背锡", "Recall": 0.7, "AP50_95": 0.4},
            "7": {"class_id": 7, "class_name": "碰伤", "Recall": 0.8, "AP50_95": 0.5},
        }
    }
    second = {
        "classes": {
            "6": {"class_id": 6, "class_name": "漏背锡", "Recall": 0.7, "AP50_95": 0.4},
            "7": {"class_id": 7, "class_name": "碰伤", "Recall": 0.72, "AP50_95": 0.46},
        }
    }
    controller.previous_per_class = first
    event = controller.evaluate(
        epoch=10,
        policy=_active_policy(),
        metrics=_metrics(recall=0.69, map50=0.74, map50_95=0.49),
        reference_metrics=_metrics(recall=0.70, map50=0.75, map50_95=0.50),
        per_class_diagnosis=second,
        active_classes=[6],
        proposed_action="shrink",
    )

    assert event["action"] == "no_op_freeze"
    assert event["affected_classes"][0]["class_id"] == 7
    assert "non_active_class_regression_with_global_recall_or_map_drop" in event["reasons"]


def test_safe_fallback_policy_has_no_active_industrial_or_roi_aug() -> None:
    controller = CATFSafeController()
    event = controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(),
        reference_metrics=_metrics(),
        active_classes=[6],
        proposed_action="accept",
    )
    policy = event["policy"]

    assert event["industrial_aug_forced_noop"] is True
    assert event["roi_aug_forced_noop"] is True
    assert not any(float(op["prob"]) > 0.0 for row in policy["classes"].values() for op in row["ops"].values())


def test_safe_fallback_transform_does_not_rewrite_labels() -> None:
    policy = force_noop_policy(_active_policy(), reason="test")
    router = SampleAwareAugmentationRouter(
        policy,
        seed=1,
        num_classes=8,
        stats=OnlineAugmentationStats(),
        roi_stats=ROIStats(),
        roi_aware=True,
        sample_aware=True,
    )
    context = OnlineTrainingContext(augmentor=router, preview_dir=None, save_preview=False, preview_count=0, total_epochs=50)
    transform = UltralyticsOnlinePolicyTransform(context, Instances)
    image = np.full((1024, 1024, 3), 120, dtype=np.uint8)
    bbox = np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32)
    labels = {
        "img": image,
        "cls": np.array([[6]], dtype=np.float32),
        "instances": make_instances(Instances, bbox, width=1024, height=1024),
    }
    before_id = id(labels["instances"])

    out = transform(labels)

    assert out is labels
    assert id(out["instances"]) == before_id
    assert router.random_draw_count == 0
    np.testing.assert_allclose(instances_to_xyxy(out["instances"], width=1024, height=1024), bbox)


def test_safe_accept_allows_small_non_negative_profile() -> None:
    controller = CATFSafeController()
    event = controller.evaluate(
        epoch=20,
        policy=_active_policy(),
        metrics=_metrics(precision=0.699, recall=0.698, map50=0.748, map50_95=0.498),
        reference_metrics=_metrics(precision=0.700, recall=0.700, map50=0.750, map50_95=0.500),
        active_classes=[6],
        proposed_action="accept",
    )

    assert event["safe_accept_allowed"] is True
    assert event["action"] == "observe"
