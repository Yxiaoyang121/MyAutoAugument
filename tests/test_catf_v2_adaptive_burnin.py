from __future__ import annotations

from copy import deepcopy

from AutoAugment.catf_v2.adaptive_burnin import (
    AdaptiveBurninConfig,
    AdaptiveBurninController,
    annotate_policy_history_with_adaptive_burnin,
    evaluate_start_condition,
)
from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix


def _policy() -> dict:
    policy = initial_policy_matrix({0: "defect", 1: "OK3"})
    row = policy["classes"]["0"]
    row["status"] = "active"
    row["state"] = "pending"
    row["ops"]["sharpen_mild"]["prob"] = 0.02
    row["ops"]["sharpen_mild"]["strength"] = 0.20
    row["ops"]["gamma"]["prob"] = 0.02
    row["ops"]["gamma"]["strength"] = 0.20
    return policy


def _history(stable: bool = True) -> list[dict]:
    if stable:
        return [
            {"precision": 0.60, "recall": 0.50, "map50": 0.500, "map50_95": 0.300},
            {"precision": 0.61, "recall": 0.51, "map50": 0.510, "map50_95": 0.305},
            {"precision": 0.60, "recall": 0.515, "map50": 0.512, "map50_95": 0.307},
        ]
    return [
        {"precision": 0.60, "recall": 0.40, "map50": 0.450, "map50_95": 0.260},
        {"precision": 0.61, "recall": 0.55, "map50": 0.530, "map50_95": 0.310},
        {"precision": 0.60, "recall": 0.48, "map50": 0.490, "map50_95": 0.300},
    ]


def _diagnosis(**overrides) -> dict:
    row = {
        "class_id": 0,
        "class_name": "defect",
        "val_instances": 30,
        "evidence_count": 8,
        "diagnosis_confidence": 0.65,
        "FN": 8,
        "AP50_95": 0.35,
        "no_aug_class": False,
        "stable_class": False,
        "high_fp_guarded": False,
        "high_fp": False,
        "low_support": False,
    }
    row.update(overrides)
    return {"classes": {"0": row}}


def _metrics() -> dict:
    return {"precision": 0.60, "recall": 0.515, "map50": 0.512, "map50_95": 0.307}


def _reference() -> dict:
    return {"precision": 0.60, "recall": 0.515, "map50": 0.512, "map50_95": 0.307}


def test_epoch_before_min_burnin_cannot_trigger() -> None:
    config = AdaptiveBurninConfig(min_burnin_epoch=5)
    result = evaluate_start_condition(
        epoch=4,
        config=config,
        metrics=_metrics(),
        reference_metrics=_reference(),
        clean_reference_metrics=_reference(),
        metric_history=_history(),
        per_class_diagnosis=_diagnosis(),
        close_mosaic_start_epoch=40,
    )
    assert result["start_condition_met"] is False
    assert "before_min_burnin_epoch" in result["blocking_reasons"]


def test_metric_unstable_cannot_trigger() -> None:
    result = evaluate_start_condition(
        epoch=5,
        config=AdaptiveBurninConfig(),
        metrics=_metrics(),
        reference_metrics=_reference(),
        clean_reference_metrics=_reference(),
        metric_history=_history(stable=False),
        per_class_diagnosis=_diagnosis(),
        close_mosaic_start_epoch=40,
    )
    assert result["start_condition_met"] is False
    assert "metric_unstable" in result["blocking_reasons"]


def test_evidence_count_insufficient_cannot_trigger() -> None:
    result = evaluate_start_condition(
        epoch=5,
        config=AdaptiveBurninConfig(),
        metrics=_metrics(),
        reference_metrics=_reference(),
        clean_reference_metrics=_reference(),
        metric_history=_history(),
        per_class_diagnosis=_diagnosis(evidence_count=4),
        close_mosaic_start_epoch=40,
    )
    assert result["start_condition_met"] is False
    assert "insufficient_diagnosis_evidence" in result["blocking_reasons"]


def test_no_aug_stable_high_fp_classes_cannot_trigger() -> None:
    for overrides in [
        {"no_aug_class": True},
        {"stable_class": True},
        {"high_fp_guarded": True},
        {"low_support": True},
    ]:
        result = evaluate_start_condition(
            epoch=5,
            config=AdaptiveBurninConfig(),
            metrics=_metrics(),
            reference_metrics=_reference(),
            clean_reference_metrics=_reference(),
            metric_history=_history(),
            per_class_diagnosis=_diagnosis(**overrides),
            close_mosaic_start_epoch=40,
        )
        assert result["start_condition_met"] is False


def test_strong_clean_baseline_without_critical_issue_cannot_trigger() -> None:
    strong_reference = {"precision": 0.70, "recall": 0.73, "map50": 0.77, "map50_95": 0.53}
    result = evaluate_start_condition(
        epoch=5,
        config=AdaptiveBurninConfig(),
        metrics=_metrics(),
        reference_metrics=_reference(),
        clean_reference_metrics=strong_reference,
        metric_history=_history(),
        per_class_diagnosis=_diagnosis(diagnosis_confidence=0.70, AP50_95=0.30),
        close_mosaic_start_epoch=40,
    )
    assert result["start_condition_met"] is False
    assert "strong_clean_baseline_protection" in result["blocking_reasons"]


def test_stable_metrics_and_evidence_can_trigger() -> None:
    controller = AdaptiveBurninController(AdaptiveBurninConfig())
    event = controller.evaluate(
        epoch=5,
        metrics=_metrics(),
        reference_metrics=_reference(),
        clean_reference_metrics=_reference(),
        metric_history=_history(),
        per_class_diagnosis=_diagnosis(),
        old_policy=initial_policy_matrix({0: "defect"}),
        proposed_policy=_policy(),
        close_mosaic_start_epoch=40,
    )
    assert event["action"] == "start_candidate"
    assert event["candidate_branch_started"] is True
    assert controller.start_epoch == 5


def test_max_burnin_not_ready_enters_noop_fallback() -> None:
    controller = AdaptiveBurninController(AdaptiveBurninConfig(max_burnin_epoch=15, allow_force_start_at_max_burnin=False))
    event = controller.evaluate(
        epoch=15,
        metrics=_metrics(),
        reference_metrics=_reference(),
        clean_reference_metrics=_reference(),
        metric_history=_history(stable=False),
        per_class_diagnosis=_diagnosis(),
        old_policy=initial_policy_matrix({0: "defect"}),
        proposed_policy=_policy(),
        close_mosaic_start_epoch=40,
    )
    assert event["action"] == "no_op_fallback"
    assert event["strict_noop"] is True
    probs = [
        op["prob"]
        for row in event["policy"]["classes"].values()
        for op in row["ops"].values()
    ]
    assert max(probs) == 0.0


def test_trigger_epoch_replaces_fixed_epoch5() -> None:
    controller = AdaptiveBurninController(AdaptiveBurninConfig(min_burnin_epoch=10, max_burnin_epoch=15, burnin_check_interval=5))
    event5 = controller.evaluate(
        epoch=5,
        metrics=_metrics(),
        reference_metrics=_reference(),
        clean_reference_metrics=_reference(),
        metric_history=_history(),
        per_class_diagnosis=_diagnosis(),
        old_policy=initial_policy_matrix({0: "defect"}),
        proposed_policy=_policy(),
        close_mosaic_start_epoch=40,
    )
    event10 = controller.evaluate(
        epoch=10,
        metrics=_metrics(),
        reference_metrics=_reference(),
        clean_reference_metrics=_reference(),
        metric_history=_history(),
        per_class_diagnosis=_diagnosis(),
        old_policy=initial_policy_matrix({0: "defect"}),
        proposed_policy=_policy(),
        close_mosaic_start_epoch=40,
    )
    assert event5["action"] == "burnin_observe"
    assert event10["action"] == "start_candidate"
    assert controller.start_epoch == 10


def test_adaptive_trigger_event_can_be_written_to_policy_history() -> None:
    history = [{"epoch": 10, "action": "accept", "accepted_policy": _policy()}]
    old_policy = initial_policy_matrix({0: "defect"})
    event = {
        "action": "burnin_observe",
        "state_after": "burnin_observe",
        "start_condition_met": False,
        "candidate_branch_started": False,
        "strict_noop": False,
        "reasons": ["metric_unstable"],
        "policy": deepcopy(old_policy),
    }
    annotate_policy_history_with_adaptive_burnin(history, event, old_policy, old_policy=old_policy)
    assert history[0]["adaptive_burnin"] is True
    assert history[0]["policy_update_applied"] is False
    assert history[0]["random_draws_allowed"] is False
