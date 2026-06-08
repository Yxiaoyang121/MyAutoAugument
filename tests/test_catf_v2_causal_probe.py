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
