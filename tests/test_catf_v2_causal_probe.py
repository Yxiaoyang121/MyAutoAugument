from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.causal_probe import (
    apply_candidate_view,
    build_probe_set,
    candidate_policy_catalog,
    compute_causal_score,
    decide_candidate_acceptance,
    evaluate_candidate_policy,
    select_best_candidate,
)
from AutoAugment.catf_v2.high_risk_class_ops import risk_info
from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix
from scripts.train_yolo_default_with_inloop_feedback import (
    apply_offline_probe_decision_to_policy,
    restrict_policy_to_causal_probe_ops,
)


def _benefit(**overrides: float) -> dict[str, float]:
    payload = {
        "fn_recovery_rate": 0.04,
        "localization_iou_gain": 0.01,
        "low_conf_tp_conf_gain": 0.02,
    }
    payload.update(overrides)
    return payload


def _risk(**overrides: float) -> dict[str, float]:
    payload = {
        "fp_increase_rate": 0.0,
        "high_fp_spillover_rate": 0.0,
        "non_active_regression_rate": 0.0,
        "ok_class_false_activation": 0.0,
        "bbox_instability_rate": 0.0,
    }
    payload.update(overrides)
    return payload


def test_causal_score_formula() -> None:
    score = compute_causal_score(
        _benefit(fn_recovery_rate=0.10, localization_iou_gain=0.04, low_conf_tp_conf_gain=0.02),
        _risk(fp_increase_rate=0.01, high_fp_spillover_rate=0.02, non_active_regression_rate=0.03),
    )

    expected = 0.10 + 0.5 * 0.04 + 0.3 * 0.02 - 0.01 - 1.2 * 0.02 - 0.03
    assert abs(score - expected) < 1e-9


def test_benefit_high_risk_low_accepts() -> None:
    decision = decide_candidate_acceptance(
        candidate_policy=candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        benefit_metrics=_benefit(),
        risk_metrics=_risk(),
        evidence_count=5,
        diagnosis_confidence=0.5,
    )

    assert decision["decision"] == "accept"
    assert decision["image_modification_allowed"] is True


def test_fp_risk_high_rejects() -> None:
    decision = decide_candidate_acceptance(
        candidate_policy=candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        benefit_metrics=_benefit(),
        risk_metrics=_risk(fp_increase_rate=0.03),
        evidence_count=5,
        diagnosis_confidence=0.5,
    )

    assert decision["decision"] == "reject"
    assert "fp_increase_rate_too_high" in decision["rejection_reasons"]


def test_non_active_regression_high_rejects() -> None:
    decision = decide_candidate_acceptance(
        candidate_policy=candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        benefit_metrics=_benefit(),
        risk_metrics=_risk(non_active_regression_rate=0.03),
        evidence_count=5,
        diagnosis_confidence=0.5,
    )

    assert decision["decision"] == "reject"
    assert "non_active_regression_rate_too_high" in decision["rejection_reasons"]


def test_ok_false_activation_rejects() -> None:
    decision = decide_candidate_acceptance(
        candidate_policy=candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        benefit_metrics=_benefit(),
        risk_metrics=_risk(ok_class_false_activation=1.0),
        evidence_count=5,
        diagnosis_confidence=0.5,
    )

    assert decision["decision"] == "reject"
    assert "ok_class_false_activation" in decision["rejection_reasons"]


def test_bbox_instability_high_rejects() -> None:
    decision = decide_candidate_acceptance(
        candidate_policy=candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        benefit_metrics=_benefit(),
        risk_metrics=_risk(bbox_instability_rate=0.03),
        evidence_count=5,
        diagnosis_confidence=0.5,
    )

    assert decision["decision"] == "reject"
    assert "bbox_instability_rate_too_high" in decision["rejection_reasons"]


def test_all_image_candidates_reject_selects_strict_noop() -> None:
    catalog = candidate_policy_catalog()
    evaluations = [
        evaluate_candidate_policy(
            candidate_policy=catalog["candidate_policy_1_roi_texture"],
            benefit_metrics=_benefit(fn_recovery_rate=0.0, localization_iou_gain=0.0),
            risk_metrics=_risk(non_active_regression_rate=0.05),
            evidence_count=5,
            diagnosis_confidence=0.5,
        ),
        evaluate_candidate_policy(
            candidate_policy=catalog["candidate_policy_2_roi_low_contrast"],
            benefit_metrics=_benefit(fn_recovery_rate=0.0, localization_iou_gain=0.0),
            risk_metrics=_risk(fp_increase_rate=0.05),
            evidence_count=5,
            diagnosis_confidence=0.5,
        ),
    ]

    selected = select_best_candidate(evaluations)

    assert selected["candidate_policy_id"] == "candidate_policy_0_noop"
    assert selected["decision"]["strict_noop"] is True


def test_noop_candidate_does_not_rewrite_labels_or_instances() -> None:
    labels = np.array([3], dtype=np.int64)
    bboxes = np.array([[1, 2, 9, 10]], dtype=np.float32)
    view = apply_candidate_view(
        candidate_policy=candidate_policy_catalog()["candidate_policy_0_noop"],
        sample={"labels": labels, "bboxes": bboxes},
    )

    assert view["labels"] is labels
    assert view["bboxes"] is bboxes
    assert view["labels_rewritten"] is False
    assert view["instances_rewritten"] is False
    assert view["image_modified"] is False


def test_probe_does_not_update_training_state() -> None:
    model = object()
    optimizer = object()
    ema = object()
    rng_state = {"state": 123}

    view = apply_candidate_view(
        candidate_policy=candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        model=model,
        optimizer=optimizer,
        ema=ema,
        rng_state=rng_state,
    )

    assert view["model_object_id"] == id(model)
    assert view["optimizer_object_id"] == id(optimizer)
    assert view["ema_object_id"] == id(ema)
    assert view["model_state_updated"] is False
    assert view["optimizer_state_updated"] is False
    assert view["ema_state_updated"] is False
    assert view["rng_state_updated"] is False
    assert view["rng_state_before"] == rng_state
    assert view["rng_state_after"] == rng_state


def test_decision_has_no_seed_id_input() -> None:
    kwargs = {
        "candidate_policy": candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        "benefit_metrics": _benefit(),
        "risk_metrics": _risk(),
        "evidence_count": 6,
        "diagnosis_confidence": 0.6,
    }

    assert decide_candidate_acceptance(**kwargs)["decision"] == "accept"
    assert "seed" not in decide_candidate_acceptance(**kwargs)


def test_decision_does_not_depend_on_fixed_class_id() -> None:
    catalog = candidate_policy_catalog()
    decisions = []
    for class_id in (4, 9, 12):
        probe_set = build_probe_set(
            candidate_class_id=class_id,
            candidate_policy_id="candidate_policy_1_roi_texture",
            diagnosis_record={"evidence_count": 8, "diagnosis_confidence": 0.8},
        )
        decisions.append(
            evaluate_candidate_policy(
                candidate_policy=catalog["candidate_policy_1_roi_texture"],
                probe_set=probe_set,
                benefit_metrics=_benefit(),
                risk_metrics=_risk(),
            )["decision"]["decision"]
        )

    assert decisions == ["accept", "accept", "accept"]


def test_riskguard_prior_is_audit_only_not_final_reject() -> None:
    prior = risk_info(9, "sharpen_mild")
    assert prior is not None
    decision = decide_candidate_acceptance(
        candidate_policy=candidate_policy_catalog()["candidate_policy_1_roi_texture"],
        benefit_metrics=_benefit(),
        risk_metrics=_risk(),
        evidence_count=8,
        diagnosis_confidence=0.8,
        audit_priors=[prior],
    )

    assert decision["audit_prior_only"] is True
    assert decision["decision"] == "accept"


def test_offline_probe_accept_filters_policy_to_candidate_ops(tmp_path) -> None:
    policy = initial_policy_matrix({4: "class_4"})
    row = policy["classes"]["4"]
    row["status"] = "active"
    row["state"] = "pending"
    for op_name in ("sharpen_mild", "local_contrast", "gamma"):
        row["ops"][op_name]["prob"] = 0.5
        row["ops"][op_name]["strength"] = 0.2

    filtered = restrict_policy_to_causal_probe_ops(policy, ["sharpen_mild", "local_contrast"])

    assert filtered["classes"]["4"]["ops"]["sharpen_mild"]["prob"] == 0.5
    assert filtered["classes"]["4"]["ops"]["local_contrast"]["prob"] == 0.5
    assert filtered["classes"]["4"]["ops"]["gamma"]["prob"] == 0.0
    assert filtered["classes"]["4"]["ops"]["gamma"]["strength"] == 0.0


def test_offline_probe_sampler_only_forces_image_noop(tmp_path) -> None:
    policy = initial_policy_matrix({8: "class_8"})
    row = policy["classes"]["8"]
    row["status"] = "active"
    row["state"] = "pending"
    row["ops"]["sharpen_mild"]["prob"] = 0.5
    row["ops"]["sharpen_mild"]["strength"] = 0.2
    decision_payload = {
        "selected_candidate_policy_id": "candidate_policy_3_sampler_only",
        "selected_candidate_action": "sampler_only",
        "image_augmentation_rejected": True,
        "development_probe_uses_existing_val_diagnostics": True,
        "selected_candidate": {
            "candidate_policy": candidate_policy_catalog()["candidate_policy_3_sampler_only"],
            "decision": {
                "decision": "sampler_only",
                "image_modification_allowed": False,
                "sample_weighting_allowed": True,
                "sample_weighting_effective": False,
                "sample_weighting_status": "pending_dataloader_support",
            },
        },
    }

    filtered, event = apply_offline_probe_decision_to_policy(policy, decision_payload, epoch_num=5, output_dir=tmp_path)

    assert filtered["classes"]["8"]["ops"]["sharpen_mild"]["prob"] == 0.0
    assert filtered["classes"]["8"]["ops"]["sharpen_mild"]["strength"] == 0.0
    assert event["probe_reject_image_aug"] is True
    assert event["action"] == "sampler_only_pending"
    assert event["sample_weighting_effective"] is False
    assert (tmp_path / "reports" / "cp_catf_sample_weight_map_epoch_5.json").exists()
