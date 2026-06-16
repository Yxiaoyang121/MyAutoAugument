from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.policy_matrix import active_class_ids, initial_policy_matrix
from AutoAugment.catf_v2.sample_router import SampleAwareAugmentationRouter
from scripts.train_yolo_default_with_inloop_feedback import apply_offline_probe_decision_to_policy


def preserve_payload() -> dict:
    return {
        "selected_candidate_policy_id": "candidate_policy_preserve_original",
        "selected_candidate_action": "preserve_original",
        "preserve_original": True,
        "decision_reason": "fixed_seed_constraint_passed_preserve_safe_original_precedence",
        "original_fixed_active_class": "4;11;12",
        "original_fixed_op_list": (
            "c4:low_recall:local_contrast@p=0.005/s=0.01,sharpen_mild@p=0.005/s=0.01; "
            "c11:texture_boundary_weak:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02; "
            "c12:weak_localization:local_contrast@p=0.015/s=0.02,sharpen_mild@p=0.015/s=0.02"
        ),
        "original_fixed_prob_strength": (
            "c4:local_contrast:0.005/0.01; c4:sharpen_mild:0.005/0.01; "
            "c11:local_contrast:0.015/0.02; c11:sharpen_mild:0.015/0.02; "
            "c12:local_contrast:0.015/0.02; c12:sharpen_mild:0.015/0.02"
        ),
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_preserve_original",
            "candidate_policy": {
                "policy_id": "candidate_policy_1b_weak_roi_texture",
                "action": "weak_roi_texture",
                "op_list": ["local_contrast"],
                "weak_image_aug": True,
                "attenuation_ratio": 0.25,
            },
            "probe_set": {
                "class_id": 4,
                "active_classes": [4, 11, 12],
            },
            "decision": {
                "decision": "preserve_original",
                "image_modification_allowed": True,
                "sample_weighting_allowed": False,
                "preserve_original": True,
            },
        },
    }


def policy_with_stale_weak_class9() -> dict:
    policy = initial_policy_matrix({4: "class4", 9: "class9", 11: "class11", 12: "class12"})
    stale = policy["classes"]["9"]
    stale["status"] = "active"
    stale["state"] = "accepted"
    stale["weak_image_aug"] = {"enabled": True, "attenuation_ratio": 0.25}
    stale["ops"]["local_contrast"]["prob"] = 0.25
    stale["ops"]["local_contrast"]["strength"] = 0.20
    return policy


def test_preserve_original_installs_multiple_fixed_classes(tmp_path) -> None:
    policy = policy_with_stale_weak_class9()

    preserved, event = apply_offline_probe_decision_to_policy(
        policy,
        preserve_payload(),
        epoch_num=15,
        output_dir=tmp_path,
        disable_sampler_only=True,
    )

    assert event["action"] == "preserve_original"
    assert event["sample_weighting_allowed"] is False
    assert event["sample_weighting_effective"] is False
    assert event["sample_weighting_status"] == "not_requested"
    assert event["preserve_expected_active_classes"] == [4, 11, 12]
    assert event["preserve_installed_active_classes"] == [4, 11, 12]
    assert active_class_ids(preserved) == [4, 11, 12]

    class4 = preserved["classes"]["4"]
    class11 = preserved["classes"]["11"]
    class12 = preserved["classes"]["12"]
    assert class4["ops"]["local_contrast"]["prob"] == 0.005
    assert class4["ops"]["local_contrast"]["strength"] == 0.01
    assert class11["ops"]["sharpen_mild"]["prob"] == 0.015
    assert class11["ops"]["sharpen_mild"]["strength"] == 0.02
    assert class12["ops"]["local_contrast"]["prob"] == 0.015
    assert class12["ops"]["local_contrast"]["strength"] == 0.02


def test_preserve_original_does_not_keep_stale_weak_candidate(tmp_path) -> None:
    policy = policy_with_stale_weak_class9()

    preserved, _ = apply_offline_probe_decision_to_policy(
        policy,
        preserve_payload(),
        epoch_num=15,
        output_dir=tmp_path,
        disable_sampler_only=True,
    )

    class9 = preserved["classes"]["9"]
    assert class9["status"] != "active"
    assert class9["ops"]["local_contrast"]["prob"] == 0.0
    assert class9["ops"]["local_contrast"]["strength"] == 0.0
    assert "weak_image_aug" not in preserved["classes"]["4"]
    assert "weak_image_aug" not in preserved["classes"]["11"]
    assert "weak_image_aug" not in preserved["classes"]["12"]


def test_sample_router_sees_all_preserved_classes(tmp_path) -> None:
    policy = policy_with_stale_weak_class9()
    preserved, event = apply_offline_probe_decision_to_policy(
        policy,
        preserve_payload(),
        epoch_num=15,
        output_dir=tmp_path,
        disable_sampler_only=True,
    )

    router = SampleAwareAugmentationRouter(preserved, seed=7, num_classes=13, roi_aware=True, sample_aware=True)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    labels = np.array([4, 11, 12], dtype=np.int64)
    boxes = np.array(
        [
            [10, 10, 30, 30],
            [50, 10, 70, 30],
            [90, 10, 110, 30],
        ],
        dtype=np.float32,
    )

    result = router.apply(image, labels, boxes)
    routed_classes = {int(item["class_id"]) for item in result.audit.get("operations", [])}

    assert event["sample_router_allowed"] is True
    assert {4, 11, 12}.issubset(routed_classes)
    assert 9 not in routed_classes
