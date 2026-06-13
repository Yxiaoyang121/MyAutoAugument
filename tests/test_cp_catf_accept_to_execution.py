from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.causal_probe import candidate_policy_catalog
from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix
from AutoAugment.catf_v2.sample_router import SampleAwareAugmentationRouter
from scripts.train_yolo_default_with_inloop_feedback import apply_offline_probe_decision_to_policy


def _accept_payload(class_id: int = 4) -> dict:
    candidate = candidate_policy_catalog()["candidate_policy_1_roi_texture"]
    return {
        "epoch": 5,
        "mode": "paper",
        "decision_basis": "paper_probe_split_run_specific_benefit_risk_metrics",
        "policy_selection_source": "probe_split",
        "policy_selection_data": "probe.yaml",
        "selected_candidate_policy_id": "candidate_policy_1_roi_texture",
        "selected_candidate_action": "accept",
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_1_roi_texture",
            "candidate_policy": candidate,
            "probe_set": {
                "class_id": class_id,
                "candidate_policy_id": "candidate_policy_1_roi_texture",
                "evidence_count": 8,
                "diagnosis_confidence": 0.75,
            },
            "causal_score": 0.12,
            "decision": {
                "candidate_policy_id": "candidate_policy_1_roi_texture",
                "accepted": True,
                "decision": "accept",
                "causal_score": 0.12,
                "rejection_reasons": [],
                "image_modification_allowed": True,
                "sample_weighting_allowed": False,
                "strict_noop": False,
            },
        },
        "image_augmentation_rejected": False,
        "strict_image_noop": False,
        "paper_probe_mode": True,
        "development_probe_uses_existing_val_diagnostics": False,
        "final_val_used_for_policy_selection": False,
        "forbid_final_val_policy_selection": True,
    }


def _sampler_payload(class_id: int = 4) -> dict:
    candidate = candidate_policy_catalog()["candidate_policy_3_sampler_only"]
    return {
        "selected_candidate_policy_id": "candidate_policy_3_sampler_only",
        "selected_candidate_action": "sampler_only",
        "policy_selection_source": "probe_split",
        "final_val_used_for_policy_selection": False,
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_3_sampler_only",
            "candidate_policy": candidate,
            "probe_set": {"class_id": class_id},
            "decision": {
                "decision": "sampler_only",
                "image_modification_allowed": False,
                "sample_weighting_allowed": True,
                "sample_weighting_effective": False,
                "sample_weighting_status": "pending_dataloader_support",
            },
        },
    }


def _weak_payload(class_id: int = 4) -> dict:
    candidate = candidate_policy_catalog()["candidate_policy_1b_weak_roi_texture"]
    return {
        "selected_candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
        "selected_candidate_action": "weak_roi_texture",
        "original_candidate_policy_id": "candidate_policy_1_roi_texture",
        "downgraded_to_weak_roi_texture": True,
        "policy_selection_source": "weak_image_aug_replay",
        "final_val_used_for_policy_selection": False,
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
            "candidate_policy": candidate,
            "probe_set": {"class_id": class_id},
            "decision": {
                "decision": "accept",
                "image_modification_allowed": True,
                "sample_weighting_allowed": False,
                "precision_aware_gate_passed": True,
                "non_active_regression_gate_passed": True,
            },
        },
    }


def _sample(class_id: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((64, 64, 3), 96, dtype=np.uint8)
    labels = np.array([class_id], dtype=np.int64)
    bboxes = np.array([[14, 14, 48, 48]], dtype=np.float32)
    return image, labels, bboxes


def test_causal_probe_accept_writes_executable_policy_matrix(tmp_path) -> None:
    policy = initial_policy_matrix({4: "defect"})

    filtered, event = apply_offline_probe_decision_to_policy(policy, _accept_payload(4), epoch_num=5, output_dir=tmp_path)

    row = filtered["classes"]["4"]
    assert event["candidate_policy_injected"] is True
    assert event["sample_router_allowed"] is True
    assert event["candidate_class_id"] == 4
    assert row["status"] == "active"
    assert row["causal_probe_selected"] is True
    assert row["ops"]["sharpen_mild"]["prob"] > 0.0
    assert row["ops"]["local_contrast"]["prob"] > 0.0
    assert row["ops"]["gamma"]["prob"] == 0.0


def test_accepted_policy_enters_sample_router_and_triggers_roi_aug(tmp_path) -> None:
    policy = initial_policy_matrix({4: "defect"})
    filtered, _ = apply_offline_probe_decision_to_policy(policy, _accept_payload(4), epoch_num=5, output_dir=tmp_path)
    router = SampleAwareAugmentationRouter(filtered, seed=3, num_classes=5, roi_aware=True)
    image, labels, bboxes = _sample(4)

    result = router.apply(image, labels, bboxes)

    assert router.random_draw_count > 0
    assert router.roi_stats.to_dict()["roi_aug_applied"] > 0
    assert router.stats.to_dict()["samples_augmented"] > 0
    assert result.audit["applied_ops"]


def test_rejected_sampler_only_does_not_trigger_image_aug(tmp_path) -> None:
    policy = initial_policy_matrix({4: "defect"})
    filtered, event = apply_offline_probe_decision_to_policy(policy, _sampler_payload(4), epoch_num=5, output_dir=tmp_path)
    router = SampleAwareAugmentationRouter(filtered, seed=3, num_classes=5, roi_aware=True)
    image, labels, bboxes = _sample(4)

    result = router.apply(image, labels, bboxes)

    assert event["action"] == "sampler_only_pending"
    assert event["probe_reject_image_aug"] is True
    assert result.audit["router"]["skip_reason"] in {"stable_classes_only", "no_active_target_class"}
    assert router.random_draw_count == 0
    assert router.roi_stats.to_dict()["roi_aug_applied"] == 0
    assert router.stats.to_dict()["samples_augmented"] == 0


def test_strict_noop_bypass_does_not_block_accepted_policy(tmp_path) -> None:
    policy = initial_policy_matrix({4: "defect"})
    filtered, event = apply_offline_probe_decision_to_policy(policy, _accept_payload(4), epoch_num=5, output_dir=tmp_path)

    assert event["image_modification_allowed"] is True
    assert event["probe_reject_image_aug"] is False
    assert event["reason"] == "offline_causal_probe_candidate_accepted"
    assert filtered["global_guard"]["active"] is False


def test_paper_mode_event_keeps_final_val_out_of_policy_selection(tmp_path) -> None:
    policy = initial_policy_matrix({4: "defect"})
    _, event = apply_offline_probe_decision_to_policy(policy, _accept_payload(4), epoch_num=5, output_dir=tmp_path)

    assert event["paper_probe_mode"] is True
    assert event["policy_selection_source"] == "probe_split"
    assert event["final_val_used_for_policy_selection"] is False
    assert event["riskguard_used_as_final_rule"] is False


def test_weak_roi_texture_injects_single_attenuated_op(tmp_path) -> None:
    policy = initial_policy_matrix({4: "defect"})
    policy["classes"]["4"]["guards"]["high_fp_guarded"] = True

    filtered, event = apply_offline_probe_decision_to_policy(policy, _weak_payload(4), epoch_num=25, output_dir=tmp_path)

    row = filtered["classes"]["4"]
    assert event["weak_image_aug"] is True
    assert event["target_high_fp_guard_overridden_by_weak_gate"] is True
    assert event["downgraded_to_weak_roi_texture"] is True
    assert event["retained_op"] == "local_contrast"
    assert event["weak_prob"] == 0.045
    assert event["weak_strength"] == 0.05
    assert row["ops"]["local_contrast"]["prob"] == 0.045
    assert row["ops"]["local_contrast"]["strength"] == 0.05
    assert row["ops"]["sharpen_mild"]["prob"] == 0.0
    assert row["weak_image_aug"]["max_aug_samples_per_interval"] == 16


def test_disable_sampler_only_blocks_sample_weighting_payload(tmp_path) -> None:
    policy = initial_policy_matrix({4: "defect"})

    _, event = apply_offline_probe_decision_to_policy(
        policy,
        _sampler_payload(4),
        epoch_num=5,
        output_dir=tmp_path,
        disable_sampler_only=True,
    )

    assert event["sample_weighting_allowed"] is False
    assert event["sampler_only_disabled"] is True
    assert event["sampler_only_blocked_by_image_only_mainline"] is True
