from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.high_risk_class_ops import (
    apply_risk_guard_to_policy,
    build_sampler_only_fallback_map,
    has_high_risk_audit_prior,
    is_high_risk_class_op,
)
from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix
from AutoAugment.catf_v2.sample_router import SampleAwareAugmentationRouter


def _sample(class_id: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[12:44, 12:44] = np.array([80, 130, 180], dtype=np.uint8)
    image[20:36, 20:36] = np.array([160, 70, 90], dtype=np.uint8)
    labels = np.array([class_id], dtype=np.int64)
    bboxes = np.array([[12, 12, 44, 44]], dtype=np.float32)
    return image, labels, bboxes


def _policy(class_id: int = 9, op_name: str = "sharpen_mild") -> dict:
    policy = initial_policy_matrix({class_id: f"class_{class_id}"})
    row = policy["classes"][str(class_id)]
    row["status"] = "active"
    row["state"] = "pending"
    row["dominant_issue"] = "texture_boundary_weak"
    row["ops"][op_name]["prob"] = 1.0
    row["ops"][op_name]["strength"] = 0.3
    return policy


def test_historical_class_op_is_audit_prior_not_default_block() -> None:
    assert has_high_risk_audit_prior(9, "roi_sharpen_mild")
    assert has_high_risk_audit_prior(9, "roi_local_contrast")
    assert not is_high_risk_class_op(9, "roi_sharpen_mild")
    assert not is_high_risk_class_op(9, "roi_local_contrast")


def test_riskguard_default_does_not_mutate_policy() -> None:
    guarded, events = apply_risk_guard_to_policy(_policy(9, "sharpen_mild"), epoch=5)

    assert events == []
    assert guarded["classes"]["9"]["ops"]["sharpen_mild"]["prob"] == 1.0
    assert guarded["classes"]["9"]["ops"]["sharpen_mild"]["strength"] == 0.3


def test_riskguard_audit_only_records_prior_without_blocking() -> None:
    guarded, events = apply_risk_guard_to_policy(_policy(9, "local_contrast"), epoch=5, audit_only=True)

    assert len(events) == 1
    assert events[0]["audit_prior_only"] is True
    assert events[0]["blocked_by_risk_guard"] is False
    assert guarded["classes"]["9"]["ops"]["local_contrast"]["prob"] == 1.0
    assert guarded["classes"]["9"]["risk_guard"]["final_decision_source"] == "causal_probe_required"


def test_direct_block_is_debug_only_and_explicit() -> None:
    guarded, events = apply_risk_guard_to_policy(_policy(9, "sharpen_mild"), epoch=5, direct_block=True)

    assert len(events) == 1
    assert events[0]["blocked_by_risk_guard"] is True
    assert events[0]["seed_specific_rule"] is False
    assert guarded["classes"]["9"]["ops"]["sharpen_mild"]["prob"] == 0.0
    assert guarded["classes"]["9"]["status"] == "observe"


def test_non_prior_class_op_has_no_audit_prior() -> None:
    guarded, events = apply_risk_guard_to_policy(_policy(8, "sharpen_mild"), epoch=5, audit_only=True)

    assert events == []
    assert not has_high_risk_audit_prior(8, "sharpen_mild")
    assert guarded["classes"]["8"]["ops"]["sharpen_mild"]["prob"] == 1.0


def test_router_no_longer_blocks_from_fixed_class_op_prior() -> None:
    router = SampleAwareAugmentationRouter(_policy(9, "sharpen_mild"), seed=2, num_classes=13, riskguard_enabled=True)
    image, labels, bboxes = _sample(9)

    result = router.apply(image, labels, bboxes)

    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_array_equal(result.bboxes, bboxes)
    assert result.audit["router"].get("riskguard_blocked") is None
    assert router.random_draw_count == 1
    assert router.roi_stats.to_dict()["roi_aug_applied"] == 1
    assert router.stats.to_dict()["samples_augmented"] == 1
    assert router.riskguard_events == []


def test_non_prior_router_op_still_executes() -> None:
    router = SampleAwareAugmentationRouter(_policy(9, "gamma"), seed=3, num_classes=13, riskguard_enabled=True)
    image, labels, bboxes = _sample(9)

    result = router.apply(image, labels, bboxes)

    assert result.audit["applied_any_aug"] is True
    assert router.random_draw_count == 1
    assert router.roi_stats.to_dict()["roi_aug_applied"] == 1
    assert router.riskguard_events == []


def test_sampler_only_fallback_generates_pending_sample_weight_map() -> None:
    payload = build_sampler_only_fallback_map(
        {"class_weights": {"9": {"weight": 1.15}}, "image_weights": {}},
        blocked_class_ids=[9],
    )

    fallback = payload["riskguard_sampler_only_fallback"]
    assert fallback["enabled"] is True
    assert fallback["target_classes"] == [9]
    assert fallback["sample_weighting_effective"] is False
    assert fallback["sample_weighting_status"] == "pending_dataloader_support"


def test_audit_prior_does_not_depend_on_seed_id() -> None:
    for seed in (0, 1, 2, 42):
        router = SampleAwareAugmentationRouter(_policy(9, "sharpen_mild"), seed=seed, num_classes=13, riskguard_enabled=True)
        image, labels, bboxes = _sample(9)
        router.apply(image, labels, bboxes)
        assert router.random_draw_count == 1
        assert router.riskguard_events == []
