from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.gated_controller import CATFGatedController, annotate_policy_history_with_gate
from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix
from AutoAugment.catf_v2.sample_router import ROIStats, SampleAwareAugmentationRouter
from AutoAugment.online_augmentation import OnlineAugmentationStats
from scripts.train_yolo_online_aug import OnlineTrainingContext, UltralyticsOnlinePolicyTransform, instances_to_xyxy, make_instances
from ultralytics.utils.instance import Instances


def _metrics(precision=0.70, recall=0.70, map50=0.75, map50_95=0.50):
    return {"precision": precision, "recall": recall, "map50": map50, "map50_95": map50_95}


def _active_policy() -> dict:
    policy = initial_policy_matrix({4: "class4", 9: "class9", 11: "class11"})
    row = policy["classes"]["4"]
    row["status"] = "active"
    row["state"] = "pending"
    row["ops"]["local_contrast"]["prob"] = 0.5
    row["ops"]["local_contrast"]["strength"] = 0.3
    return policy


def _per_class(recall=0.50, ap95=0.40) -> dict:
    return {
        "classes": {
            "4": {"class_id": 4, "class_name": "class4", "Recall": recall, "AP50_95": ap95},
            "9": {"class_id": 9, "class_name": "class9", "Recall": 0.45, "AP50_95": 0.35},
        }
    }


def test_epoch5_observes_without_no_gain_fallback() -> None:
    controller = CATFGatedController()
    event = controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(),
        reference_metrics=_metrics(),
        per_class_diagnosis=_per_class(),
        active_classes=[4],
        proposed_action="accept",
    )

    assert event["action"] == "observe"
    assert event["triggered"] is False
    assert event["positive_gain"] is False
    assert event["fallback_active_after"] is False


def test_epoch5_severe_degradation_can_fallback() -> None:
    controller = CATFGatedController()
    event = controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(map50_95=0.479),
        reference_metrics=_metrics(map50_95=0.500),
        per_class_diagnosis=_per_class(),
        active_classes=[4],
        proposed_action="accept",
    )

    assert event["action"] == "no_op_freeze"
    assert "epoch5_severe_degradation" in event["reasons"]


def test_epoch10_bad_pattern_a_fallback() -> None:
    controller = CATFGatedController()
    controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(),
        reference_metrics=_metrics(),
        per_class_diagnosis=_per_class(),
        active_classes=[4],
    )
    event = controller.evaluate(
        epoch=10,
        policy=_active_policy(),
        metrics=_metrics(recall=0.675, map50=0.740, map50_95=0.490),
        reference_metrics=_metrics(recall=0.700, map50=0.750, map50_95=0.500),
        per_class_diagnosis=_per_class(recall=0.49, ap95=0.39),
        active_classes=[4],
    )

    assert event["action"] == "no_op_freeze"
    assert "bad_pattern_A" in event["bad_patterns"]


def test_epoch10_bad_pattern_b_fallback() -> None:
    controller = CATFGatedController()
    controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(),
        reference_metrics=_metrics(),
        per_class_diagnosis=_per_class(),
        active_classes=[9],
    )
    event = controller.evaluate(
        epoch=10,
        policy=_active_policy(),
        metrics=_metrics(precision=0.736, recall=0.655, map50=0.729, map50_95=0.482),
        reference_metrics=_metrics(precision=0.700, recall=0.700, map50=0.750, map50_95=0.500),
        per_class_diagnosis=_per_class(recall=0.49, ap95=0.39),
        active_classes=[9],
    )

    assert event["action"] == "no_op_freeze"
    assert "bad_pattern_B" in event["bad_patterns"]


def test_epoch10_bad_pattern_c_fallback() -> None:
    controller = CATFGatedController()
    controller.previous_positive_gain = True
    controller.previous_per_class = _per_class(recall=0.50, ap95=0.40)
    event = controller.evaluate(
        epoch=10,
        policy=_active_policy(),
        metrics=_metrics(recall=0.695, map50=0.751, map50_95=0.499),
        reference_metrics=_metrics(recall=0.700, map50=0.750, map50_95=0.500),
        per_class_diagnosis=_per_class(recall=0.50, ap95=0.40),
        active_classes=[4],
    )

    assert event["action"] == "no_op_freeze"
    assert "bad_pattern_C" in event["bad_patterns"]


def test_seed0_like_map_gain_is_not_fallback() -> None:
    controller = CATFGatedController()
    controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(),
        reference_metrics=_metrics(),
        per_class_diagnosis=_per_class(),
        active_classes=[4],
    )
    event = controller.evaluate(
        epoch=10,
        policy=_active_policy(),
        metrics=_metrics(precision=0.694, recall=0.690, map50=0.759, map50_95=0.514),
        reference_metrics=_metrics(precision=0.700, recall=0.700, map50=0.750, map50_95=0.500),
        per_class_diagnosis=_per_class(recall=0.51, ap95=0.41),
        active_classes=[4],
    )

    assert event["action"] == "gate_continue"
    assert event["triggered"] is False
    assert event["protected_pattern"] == "seed0_like_map_gain"


def test_seed1_like_all_metric_gain_is_not_fallback() -> None:
    controller = CATFGatedController()
    controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(),
        reference_metrics=_metrics(),
        per_class_diagnosis=_per_class(),
        active_classes=[4],
    )
    event = controller.evaluate(
        epoch=10,
        policy=_active_policy(),
        metrics=_metrics(precision=0.701, recall=0.710, map50=0.755, map50_95=0.504),
        reference_metrics=_metrics(precision=0.700, recall=0.700, map50=0.750, map50_95=0.500),
        per_class_diagnosis=_per_class(recall=0.51, ap95=0.41),
        active_classes=[4],
    )

    assert event["action"] == "gate_continue"
    assert event["triggered"] is False
    assert event["protected_pattern"] == "seed1_like_all_metric_gain"


def test_seed2_like_precision_up_recall_map_down_fallback() -> None:
    controller = CATFGatedController()
    controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(),
        reference_metrics=_metrics(),
        per_class_diagnosis=_per_class(),
        active_classes=[9],
    )
    event = controller.evaluate(
        epoch=10,
        policy=_active_policy(),
        metrics=_metrics(precision=0.7368, recall=0.6548, map50=0.7292, map50_95=0.4819),
        reference_metrics=_metrics(precision=0.7000, recall=0.7000, map50=0.7500, map50_95=0.5000),
        per_class_diagnosis=_per_class(recall=0.49, ap95=0.39),
        active_classes=[9],
    )

    assert event["action"] == "no_op_freeze"
    assert event["strict_noop"] is True
    assert "bad_pattern_B" in event["bad_patterns"]


def test_gated_fallback_transform_is_strict_noop() -> None:
    controller = CATFGatedController()
    event = controller.evaluate(
        epoch=5,
        policy=_active_policy(),
        metrics=_metrics(map50=0.729),
        reference_metrics=_metrics(map50=0.750),
        per_class_diagnosis=_per_class(),
        active_classes=[4],
    )
    router = SampleAwareAugmentationRouter(
        event["policy"],
        seed=1,
        num_classes=12,
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
        "cls": np.array([[4]], dtype=np.float32),
        "instances": make_instances(Instances, bbox, width=1024, height=1024),
    }
    before_id = id(labels["instances"])

    out = transform(labels)

    assert out is labels
    assert id(out["instances"]) == before_id
    assert router.random_draw_count == 0
    np.testing.assert_allclose(instances_to_xyxy(out["instances"], width=1024, height=1024), bbox)


def test_gate_decision_is_written_to_policy_history() -> None:
    controller = CATFGatedController()
    history = [{"epoch": 10, "action": "accept", "guard_triggered": []}]
    event = controller.evaluate(
        epoch=10,
        policy=_active_policy(),
        metrics=_metrics(precision=0.7368, recall=0.6548, map50=0.7292, map50_95=0.4819),
        reference_metrics=_metrics(precision=0.7000, recall=0.7000, map50=0.7500, map50_95=0.5000),
        per_class_diagnosis=_per_class(),
        active_classes=[9],
    )

    annotate_policy_history_with_gate(history, event, event["policy"])

    assert history[0]["catf_gated_mode"] is True
    assert history[0]["gated_controller_event"]["action"] == "no_op_freeze"
    assert history[0]["action"] == "no_op_freeze"
    assert "bad_pattern_B" in history[0]["guard_triggered"]
