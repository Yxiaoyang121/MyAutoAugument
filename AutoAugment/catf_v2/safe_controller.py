from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


@dataclass
class CATFSafeController:
    """Safety layer for CATF-v2 policy updates.

    The safe controller does not generate augmentations itself. It observes
    feedback metrics against the clean reference curve and can force the
    class-aware policy matrix into a strict no-op state when CATF-v2 starts
    damaging a strong baseline.
    """

    early_abstention_epochs: set[int] = field(default_factory=lambda: {5, 10, 15})
    fallback_active: bool = False
    fallback_reason: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    previous_per_class: dict[str, Any] | None = None

    def evaluate(
        self,
        *,
        epoch: int,
        policy: dict[str, Any],
        metrics: dict[str, Any],
        reference_metrics: dict[str, Any],
        per_class_diagnosis: dict[str, Any] | None = None,
        active_classes: list[int] | None = None,
        proposed_action: str | None = None,
    ) -> dict[str, Any]:
        delta = metric_delta(metrics, reference_metrics)
        event = {
            "epoch": int(epoch),
            "catf_safe_mode": True,
            "metrics": compact_metrics(metrics),
            "reference_metrics": compact_metrics(reference_metrics),
            "delta_metrics": delta,
            "proposed_action": proposed_action,
            "active_classes": [int(value) for value in (active_classes or [])],
            "triggered": False,
            "action": "observe",
            "reasons": [],
            "affected_classes": [],
            "safe_accept_allowed": safe_accept_allowed(delta),
            "fallback_active_before": bool(self.fallback_active),
            "fallback_active_after": bool(self.fallback_active),
            "industrial_aug_forced_noop": False,
            "roi_aug_forced_noop": False,
        }

        if self.fallback_active:
            event["triggered"] = True
            event["action"] = "no_op_freeze"
            event["reasons"] = [self.fallback_reason or "safe_fallback_already_active"]
            event["policy"] = force_noop_policy(policy, reason=event["reasons"][0])
            event["fallback_active_after"] = True
            event["industrial_aug_forced_noop"] = True
            event["roi_aug_forced_noop"] = True
            self._remember(event, per_class_diagnosis)
            return event

        reasons = baseline_protection_reasons(metrics, reference_metrics, delta)
        if not reasons and int(epoch) in self.early_abstention_epochs:
            if not any((delta.get(key) or 0.0) > 0.0 for key in ("recall", "map50", "map50_95")):
                reasons.append("early_abstention_no_recall_or_map_gain")

        negative = self.negative_effect_attribution(per_class_diagnosis, active_classes or [])
        if negative["affected_classes"]:
            event["affected_classes"] = negative["affected_classes"]
            event["negative_effect_attribution"] = negative
            if any((delta.get(key) or 0.0) < 0.0 for key in ("recall", "map50", "map50_95")):
                reasons.append("non_active_class_regression_with_global_recall_or_map_drop")

        if reasons:
            reason = reasons[0]
            self.fallback_active = True
            self.fallback_reason = reason
            event["triggered"] = True
            event["action"] = "no_op_freeze"
            event["reasons"] = reasons
            event["policy"] = force_noop_policy(policy, reason=reason)
            event["fallback_active_after"] = True
            event["industrial_aug_forced_noop"] = True
            event["roi_aug_forced_noop"] = True
            self._remember(event, per_class_diagnosis)
            return event

        if proposed_action == "accept" and not event["safe_accept_allowed"]:
            event["triggered"] = True
            event["action"] = "safe_accept_blocked"
            event["reasons"] = ["safe_accept_metric_drop_guard"]
            event["policy"] = shrink_active_policy(policy, reason="safe_accept_metric_drop_guard")
            self._remember(event, per_class_diagnosis)
            return event

        event["policy"] = deepcopy(policy)
        self._remember(event, per_class_diagnosis)
        return event

    def negative_effect_attribution(self, per_class_diagnosis: dict[str, Any] | None, active_classes: list[int]) -> dict[str, Any]:
        if not per_class_diagnosis or not self.previous_per_class:
            return {"affected_classes": [], "reason": "insufficient_history"}
        current = per_class_diagnosis.get("classes") or {}
        previous = self.previous_per_class.get("classes") or {}
        active = {int(value) for value in active_classes}
        affected = []
        for class_id, row in current.items():
            cid = int(class_id)
            if cid in active:
                continue
            prev = previous.get(str(cid), {})
            recall_drop = float(row.get("Recall", 0.0) or 0.0) - float(prev.get("Recall", row.get("Recall", 0.0)) or 0.0)
            ap95_drop = float(row.get("AP50_95", 0.0) or 0.0) - float(prev.get("AP50_95", row.get("AP50_95", 0.0)) or 0.0)
            if recall_drop < -0.05 or ap95_drop < -0.03:
                affected.append(
                    {
                        "class_id": cid,
                        "class_name": row.get("class_name", str(cid)),
                        "delta_recall_vs_previous": round(recall_drop, 6),
                        "delta_ap50_95_vs_previous": round(ap95_drop, 6),
                    }
                )
        return {"affected_classes": affected, "reason": "non_active_class_regression"}

    def _remember(self, event: dict[str, Any], per_class_diagnosis: dict[str, Any] | None) -> None:
        stored = deepcopy(event)
        stored.pop("policy", None)
        self.events.append(stored)
        if per_class_diagnosis:
            self.previous_per_class = deepcopy(per_class_diagnosis)


def baseline_protection_reasons(metrics: dict[str, Any], reference_metrics: dict[str, Any], delta: dict[str, float | None]) -> list[str]:
    reasons: list[str] = []
    ref_recall = reference_metrics.get("recall")
    ref_map95 = reference_metrics.get("map50_95")
    ref_map50 = reference_metrics.get("map50")
    if ref_recall is not None and float(ref_recall) >= 0.72 and (delta.get("recall") or 0.0) < -0.015:
        reasons.append("high_recall_baseline_protection")
    if ref_map95 is not None and float(ref_map95) >= 0.52 and (delta.get("map50_95") or 0.0) < -0.008:
        reasons.append("high_map95_baseline_protection")
    if ref_map50 is not None and float(ref_map50) >= 0.76 and (delta.get("map50") or 0.0) < -0.008:
        reasons.append("high_map50_baseline_protection")
    return reasons


def safe_accept_allowed(delta: dict[str, float | None]) -> bool:
    return all((delta.get(key) is None or float(delta[key]) >= -0.005) for key in METRIC_KEYS)


def force_noop_policy(policy: dict[str, Any], *, reason: str) -> dict[str, Any]:
    matrix = deepcopy(policy)
    matrix.setdefault("safe_controller", {})["fallback_active"] = True
    matrix["safe_controller"]["fallback_reason"] = reason
    matrix["global_guard"] = {"active": True, "reason": reason}
    for item in (matrix.get("classes") or {}).values():
        item["status"] = "frozen"
        item["state"] = "frozen"
        item["frozen_reason"] = reason
        for op in (item.get("ops") or {}).values():
            op["prob"] = 0.0
            op["strength"] = 0.0
    return matrix


def shrink_active_policy(policy: dict[str, Any], *, reason: str) -> dict[str, Any]:
    matrix = deepcopy(policy)
    matrix.setdefault("safe_controller", {})["last_action"] = "safe_accept_blocked"
    matrix["safe_controller"]["last_reason"] = reason
    for item in (matrix.get("classes") or {}).values():
        if item.get("status") not in {"active", "pending", "accepted"}:
            continue
        for op in (item.get("ops") or {}).values():
            op["prob"] = round(float(op.get("prob", 0.0) or 0.0) * 0.5, 6)
            op["strength"] = round(float(op.get("strength", 0.0) or 0.0) * 0.8, 6)
    return matrix


def metric_delta(metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        reference = reference_metrics.get(key)
        out[key] = None if value is None or reference is None else float(value) - float(reference)
    return out


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}
