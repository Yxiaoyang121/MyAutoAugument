from __future__ import annotations

import json

from AutoAugment.feedback_policy_controller import FeedbackPolicyController, default_catf_policy


REF = {"precision": 0.80, "recall": 0.70, "map50": 0.80, "map50_95": 0.55}


def op(policy: dict, name: str) -> dict:
    return next(item for item in policy["operations"] if item["name"] == name)


def test_trust_region_limits_single_step_prob_and_strength(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)
    before = controller.policy

    updated = controller.update(
        {"low_contrast_fn_high": True},
        epoch=5,
        metrics={"precision": 0.81, "recall": 0.65, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )

    for name in ["clahe", "gamma", "sharpen_mild", "local_contrast"]:
        assert abs(op(updated, name)["prob"] - op(before, name)["prob"]) <= 0.0200001
        assert abs(op(updated, name)["strength"] - op(before, name)["strength"]) <= 0.0300001


def test_photometric_group_budget_is_enforced(tmp_path) -> None:
    policy = default_catf_policy()
    for name in ["clahe", "gamma", "brightness", "contrast"]:
        op(policy, name)["prob"] = 0.20

    controller = FeedbackPolicyController(policy, history_dir=tmp_path)

    total = sum(op(controller.policy, name)["prob"] for name in ["clahe", "gamma", "brightness", "contrast"])
    assert total <= 0.3000001


def test_cutout_safe_budget_is_enforced(tmp_path) -> None:
    policy = default_catf_policy()
    op(policy, "cutout_safe")["prob"] = 0.50

    controller = FeedbackPolicyController(policy, history_dir=tmp_path)

    assert op(controller.policy, "cutout_safe")["prob"] <= 0.0800001


def test_precision_guard_shrinks_risky_ops(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)

    updated = controller.update(
        {},
        epoch=5,
        metrics={"precision": 0.789, "recall": 0.72, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )

    assert op(updated, "brightness")["prob"] < 0.03
    assert op(updated, "clahe")["prob"] < 0.05
    assert "precision_guard" in controller.history[-1]["guard_triggered"]


def test_map50_95_guard_shrinks_cutout_and_boosts_texture(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)

    updated = controller.update(
        {},
        epoch=5,
        metrics={"precision": 0.80, "recall": 0.70, "map50": 0.80, "map50_95": 0.539},
        reference_metrics=REF,
    )

    assert op(updated, "cutout_safe")["prob"] < 0.03
    assert op(updated, "sharpen_mild")["prob"] > 0.08
    assert "map50_95_guard" in controller.history[-1]["guard_triggered"]


def test_delayed_acceptance_marks_next_safe_update(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)
    controller.update(
        {"low_contrast_fn_high": True},
        epoch=5,
        metrics={"precision": 0.80, "recall": 0.69, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )
    pending = controller.pending_policy_id

    controller.update(
        {},
        epoch=10,
        metrics={"precision": 0.81, "recall": 0.72, "map50": 0.81, "map50_95": 0.56},
        reference_metrics=REF,
    )

    assert controller.history[-1]["action"] == "accept"
    assert controller.history[-1]["accepted_previous_policy"] is True
    assert controller.last_safe_policy_id == pending


def test_rollback_restores_last_safe_policy(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)
    safe_policy = json.loads(json.dumps(controller.last_safe_policy))
    controller.update(
        {"low_contrast_fn_high": True},
        epoch=5,
        metrics={"precision": 0.80, "recall": 0.69, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )

    updated = controller.update(
        {},
        epoch=10,
        metrics={"precision": 0.78, "recall": 0.73, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )

    assert controller.history[-1]["action"] == "rollback"
    assert controller.history[-1]["rollback_reason"]
    assert op(updated, "clahe")["prob"] == op(safe_policy, "clahe")["prob"]


def test_cooldown_after_rollback_only_shrinks_risk_ops(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)
    controller.update(
        {"low_contrast_fn_high": True},
        epoch=5,
        metrics={"precision": 0.80, "recall": 0.69, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )
    controller.update(
        {},
        epoch=10,
        metrics={"precision": 0.78, "recall": 0.73, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )
    before = controller.policy

    updated = controller.update(
        {"low_contrast_fn_high": True},
        epoch=15,
        metrics={"precision": 0.80, "recall": 0.73, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )

    assert controller.history[-1]["action"] == "cooldown"
    assert op(updated, "clahe")["prob"] <= op(before, "clahe")["prob"]
    assert op(updated, "brightness")["prob"] <= op(before, "brightness")["prob"]


def test_freeze_after_two_constraint_warnings(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)

    controller.update({}, epoch=5, metrics={"precision": 0.789, "recall": 0.70, "map50": 0.80, "map50_95": 0.55}, reference_metrics=REF)
    controller.update({}, epoch=10, metrics={"precision": 0.789, "recall": 0.70, "map50": 0.80, "map50_95": 0.55}, reference_metrics=REF)

    assert controller.frozen is True
    assert controller.history[-1]["frozen"] is True


def test_epoch_40_auto_freeze(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)

    controller.update({}, epoch=40, metrics={"precision": 0.81, "recall": 0.71, "map50": 0.81, "map50_95": 0.56}, reference_metrics=REF)

    assert controller.frozen is True
    assert controller.history[-1]["action"] == "freeze"


def test_policy_history_fields_are_complete(tmp_path) -> None:
    controller = FeedbackPolicyController(default_catf_policy(), history_dir=tmp_path)
    controller.update(
        {"low_contrast_fn_high": True},
        epoch=5,
        metrics={"precision": 0.80, "recall": 0.69, "map50": 0.80, "map50_95": 0.55},
        reference_metrics=REF,
    )

    payload = json.loads((tmp_path / "policy_history.json").read_text(encoding="utf-8"))
    record = payload["history"][0]
    for key in [
        "epoch",
        "metrics",
        "reference_metrics",
        "delta_metrics",
        "diagnosis_summary",
        "old_policy",
        "proposed_policy",
        "accepted_policy",
        "last_safe_policy_id",
        "action",
        "guard_triggered",
        "rollback_reason",
        "frozen",
        "group_budget_before",
        "group_budget_after",
        "trust_region_clipping",
    ]:
        assert key in record
