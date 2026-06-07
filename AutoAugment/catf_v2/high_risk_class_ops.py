from __future__ import annotations

from copy import deepcopy
from typing import Any


RISK_OP_ALIASES = {
    "roi_sharpen_mild": "sharpen_mild",
    "roi_local_contrast": "local_contrast",
}

HIGH_RISK_CLASS_OPS: dict[int, dict[str, dict[str, Any]]] = {
    9: {
        "sharpen_mild": {
            "risk_reasons": [
                "active_class_regression",
                "non_active_class_regression",
                "seed2_root_cause_audit",
                "texture_boundary_weak_intervention_failed",
            ],
            "causal_probe_required": True,
            "default_action": "block",
        },
        "local_contrast": {
            "risk_reasons": [
                "active_class_regression",
                "non_active_class_regression",
                "seed2_root_cause_audit",
                "texture_boundary_weak_intervention_failed",
            ],
            "causal_probe_required": True,
            "default_action": "block",
        },
    }
}


def canonical_op_name(op_name: str) -> str:
    return RISK_OP_ALIASES.get(str(op_name), str(op_name))


def risk_info(class_id: int | str, op_name: str) -> dict[str, Any] | None:
    class_key = int(class_id)
    canonical = canonical_op_name(op_name)
    info = HIGH_RISK_CLASS_OPS.get(class_key, {}).get(canonical)
    return deepcopy(info) if info else None


def is_high_risk_class_op(class_id: int | str, op_name: str, *, causal_probe_passed: bool = False) -> bool:
    info = risk_info(class_id, op_name)
    if not info:
        return False
    if info.get("causal_probe_required", True) and not causal_probe_passed:
        return True
    return bool(info.get("default_action") == "block")


def build_riskguard_event(
    *,
    class_id: int | str,
    op_name: str,
    epoch: int | None = None,
    source: str = "policy_matrix",
    sampler_only_fallback: bool = True,
    sample_weight_map_path: str | None = None,
    causal_probe_passed: bool = False,
) -> dict[str, Any]:
    info = risk_info(class_id, op_name) or {}
    canonical = canonical_op_name(op_name)
    reasons = list(info.get("risk_reasons") or [])
    return {
        "epoch": None if epoch is None else int(epoch),
        "source": str(source),
        "blocked_by_risk_guard": True,
        "class_id": int(class_id),
        "op_name": str(op_name),
        "canonical_op_name": canonical,
        "risk_reason": ",".join(reasons),
        "risk_reasons": reasons,
        "causal_probe_required": bool(info.get("causal_probe_required", True)),
        "causal_probe_passed": bool(causal_probe_passed),
        "action": "block_roi_op",
        "fallback_action": "sampler_only" if sampler_only_fallback else "no_op",
        "sampler_only_fallback": bool(sampler_only_fallback),
        "sample_weight_map_path": sample_weight_map_path,
        "sample_weighting_effective": False,
        "sample_weighting_status": "pending_dataloader_support" if sampler_only_fallback else "not_requested",
        "seed_specific_rule": False,
    }


def apply_risk_guard_to_policy(
    policy_matrix: dict[str, Any],
    *,
    epoch: int | None = None,
    sampler_only_fallback: bool = True,
    sample_weight_map_path: str | None = None,
    causal_probe_passed: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a policy with high-risk class-op combinations zeroed.

    The guard is intentionally class-op based, not seed based. It blocks known
    risky ROI texture interventions unless a future causal probe explicitly
    clears the pair.
    """

    guarded = deepcopy(policy_matrix)
    events: list[dict[str, Any]] = []
    for raw_class_id, class_policy in (guarded.get("classes") or {}).items():
        class_id = int(raw_class_id)
        blocked_ops: list[dict[str, Any]] = []
        for op_name, op in sorted((class_policy.get("ops") or {}).items()):
            prob = float(op.get("prob", 0.0) or 0.0)
            strength = float(op.get("strength", 0.0) or 0.0)
            if prob <= 0.0 or strength <= 0.0:
                continue
            if not is_high_risk_class_op(class_id, op_name, causal_probe_passed=causal_probe_passed):
                continue
            event = build_riskguard_event(
                class_id=class_id,
                op_name=op_name,
                epoch=epoch,
                source="policy_matrix",
                sampler_only_fallback=sampler_only_fallback,
                sample_weight_map_path=sample_weight_map_path,
                causal_probe_passed=causal_probe_passed,
            )
            event["blocked_prob"] = prob
            event["blocked_strength"] = strength
            events.append(event)
            blocked_ops.append(
                {
                    "op_name": str(op_name),
                    "canonical_op_name": canonical_op_name(op_name),
                    "prob": prob,
                    "strength": strength,
                    "risk_reasons": event["risk_reasons"],
                }
            )
            op["prob"] = 0.0
            op["strength"] = 0.0

        if not blocked_ops:
            continue
        class_policy["risk_guard"] = {
            "blocked_by_risk_guard": True,
            "blocked_ops": blocked_ops,
            "fallback_action": "sampler_only" if sampler_only_fallback else "no_op",
            "sampler_only_fallback": bool(sampler_only_fallback),
            "sample_weight_map_path": sample_weight_map_path,
            "sample_weighting_effective": False,
            "sample_weighting_status": "pending_dataloader_support" if sampler_only_fallback else "not_requested",
            "seed_specific_rule": False,
        }
        if not active_ops(class_policy):
            class_policy["status"] = "observe"
            class_policy["state"] = "accepted"
            class_policy["pending_since_epoch"] = None
            class_policy["risk_guard"]["class_action"] = "class_no_op"
            for event in events:
                if int(event.get("class_id", -1)) == class_id:
                    event["class_action"] = "class_no_op"
    return guarded, events


def active_ops(class_policy: dict[str, Any]) -> list[str]:
    out = []
    for op_name, op in sorted((class_policy.get("ops") or {}).items()):
        if float(op.get("prob", 0.0) or 0.0) > 0.0 and float(op.get("strength", 0.0) or 0.0) > 0.0:
            out.append(str(op_name))
    return out


def build_sampler_only_fallback_map(
    sample_weight_map: dict[str, Any] | None,
    *,
    blocked_class_ids: list[int],
) -> dict[str, Any]:
    payload = deepcopy(sample_weight_map or {"class_weights": {}, "image_weights": {}})
    payload["riskguard_sampler_only_fallback"] = {
        "enabled": True,
        "target_classes": [int(item) for item in sorted(set(blocked_class_ids))],
        "sample_weighting_effective": False,
        "sample_weighting_status": "pending_dataloader_support",
        "image_modification": False,
    }
    return payload
