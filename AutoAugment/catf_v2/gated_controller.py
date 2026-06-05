from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from AutoAugment.catf_v2.safe_controller import compact_metrics, force_noop_policy, metric_delta


METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


@dataclass
class CATFGatedController:
    """Gate CATF-v2 updates after a short observation window.

    The gated controller keeps the CATF-v2 proposal path live at epoch 5,
    delays the first formal fallback decision until epoch 10, and only enters
    strict no-op mode for clear degradation patterns.
    """

    fallback_active: bool = False
    fallback_reason: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    previous_per_class: dict[str, Any] | None = None
    previous_positive_gain: bool | None = None
    previous_active_classes: list[int] = field(default_factory=list)

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
        epoch = int(epoch)
        active = [int(value) for value in (active_classes or [])]
        delta = metric_delta(metrics, reference_metrics)
        positive = positive_gain(delta)
        severe = severe_degradation(delta)
        event = {
            "epoch": epoch,
            "catf_gated_mode": True,
            "gate_phase": gate_phase(epoch),
            "metrics": compact_metrics(metrics),
            "reference_metrics": compact_metrics(reference_metrics),
            "delta_metrics": delta,
            "proposed_action": proposed_action,
            "active_classes": active,
            "triggered": False,
            "action": "observe" if epoch <= 5 else "gate_continue",
            "reasons": [],
            "bad_patterns": [],
            "positive_gain": bool(positive),
            "previous_positive_gain": self.previous_positive_gain,
            "no_positive_gain_streak": no_positive_gain_streak(self.previous_positive_gain, positive),
            "severe_degradation": bool(severe),
            "fallback_active_before": bool(self.fallback_active),
            "fallback_active_after": bool(self.fallback_active),
            "industrial_aug_forced_noop": False,
            "roi_aug_forced_noop": False,
            "strict_noop": False,
        }

        if self.fallback_active:
            reason = self.fallback_reason or "gated_fallback_already_active"
            return self._fallback_event(event, policy, reason, per_class_diagnosis, active)

        reasons: list[str] = []
        bad_patterns: list[str] = []

        if epoch <= 5:
            if severe:
                reasons.append("epoch5_severe_degradation")
        elif epoch == 10:
            bad_patterns = gate_bad_patterns(
                delta,
                per_class_diagnosis=per_class_diagnosis,
                previous_per_class=self.previous_per_class,
                active_classes=active or self.previous_active_classes,
            )
            event["bad_patterns"] = bad_patterns
            if seed1_all_metric_gain_keep(delta):
                event["protected_pattern"] = "seed1_like_all_metric_gain"
            elif seed0_map_gain_keep(delta):
                event["protected_pattern"] = "seed0_like_map_gain"
            else:
                if bad_patterns:
                    reasons.append("epoch10_" + bad_patterns[0])
                elif self.previous_positive_gain is False and not positive:
                    reasons.append("two_feedback_points_without_positive_gain")
        else:
            # After the first gate, avoid reacting to a single noisy epoch.
            # Fallback only if two consecutive feedback points have no positive
            # gain and the current point is also below clean on recall or mAP.
            if self.previous_positive_gain is False and not positive and global_recall_or_map_drop(delta):
                reasons.append("two_consecutive_monitor_points_without_positive_gain")

        if reasons:
            return self._fallback_event(event, policy, reasons[0], per_class_diagnosis, active, reasons=reasons)

        event["policy"] = deepcopy(policy)
        self._remember(event, per_class_diagnosis, positive, active)
        return event

    def _fallback_event(
        self,
        event: dict[str, Any],
        policy: dict[str, Any],
        reason: str,
        per_class_diagnosis: dict[str, Any] | None,
        active_classes: list[int],
        *,
        reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        self.fallback_active = True
        self.fallback_reason = reason
        event["triggered"] = True
        event["action"] = "no_op_freeze"
        event["reasons"] = list(reasons or [reason])
        event["policy"] = force_noop_policy(policy, reason=reason)
        event["fallback_active_after"] = True
        event["industrial_aug_forced_noop"] = True
        event["roi_aug_forced_noop"] = True
        event["strict_noop"] = True
        self._remember(event, per_class_diagnosis, bool(event.get("positive_gain")), active_classes)
        return event

    def _remember(
        self,
        event: dict[str, Any],
        per_class_diagnosis: dict[str, Any] | None,
        positive_gain_value: bool,
        active_classes: list[int],
    ) -> None:
        stored = deepcopy(event)
        stored.pop("policy", None)
        self.events.append(stored)
        self.previous_positive_gain = bool(positive_gain_value)
        if per_class_diagnosis:
            self.previous_per_class = deepcopy(per_class_diagnosis)
        if active_classes:
            self.previous_active_classes = [int(value) for value in active_classes]


def gate_phase(epoch: int) -> str:
    if int(epoch) <= 5:
        return "epoch5_observe"
    if int(epoch) == 10:
        return "first_formal_gate"
    return "monitor"


def positive_gain(delta: dict[str, float | None]) -> bool:
    precision = value(delta, "precision")
    recall = value(delta, "recall")
    map50 = value(delta, "map50")
    map95 = value(delta, "map50_95")
    return map95 >= 0.003 or map50 >= 0.005 or (recall >= 0.010 and precision >= -0.005)


def severe_degradation(delta: dict[str, float | None]) -> bool:
    recall = value(delta, "recall")
    map50 = value(delta, "map50")
    map95 = value(delta, "map50_95")
    return map95 < -0.020 or map50 < -0.020 or (recall < -0.050 and map95 < -0.010)


def gate_bad_patterns(
    delta: dict[str, float | None],
    *,
    per_class_diagnosis: dict[str, Any] | None,
    previous_per_class: dict[str, Any] | None,
    active_classes: list[int],
) -> list[str]:
    precision = value(delta, "precision")
    recall = value(delta, "recall")
    map50 = value(delta, "map50")
    map95 = value(delta, "map50_95")
    patterns: list[str] = []
    if recall < -0.020 and map50 < -0.008 and map95 < -0.008:
        patterns.append("bad_pattern_A")
    if precision > 0.020 and recall < -0.030 and map95 < -0.010:
        patterns.append("bad_pattern_B")
    if active_class_no_improvement(per_class_diagnosis, previous_per_class, active_classes) and global_recall_or_map_drop(delta):
        patterns.append("bad_pattern_C")
    return patterns


def seed0_map_gain_keep(delta: dict[str, float | None]) -> bool:
    return value(delta, "map50") > 0.0 and value(delta, "map50_95") > 0.0 and value(delta, "precision") >= -0.010


def seed1_all_metric_gain_keep(delta: dict[str, float | None]) -> bool:
    return all(value(delta, key) >= 0.0 for key in METRIC_KEYS)


def global_recall_or_map_drop(delta: dict[str, float | None]) -> bool:
    return value(delta, "recall") < 0.0 or value(delta, "map50_95") < 0.0


def no_positive_gain_streak(previous: bool | None, current: bool) -> int:
    if current:
        return 0
    if previous is False:
        return 2
    return 1


def active_class_no_improvement(
    per_class_diagnosis: dict[str, Any] | None,
    previous_per_class: dict[str, Any] | None,
    active_classes: list[int],
) -> bool:
    if not per_class_diagnosis or not previous_per_class or not active_classes:
        return False
    current = per_class_diagnosis.get("classes") or {}
    previous = previous_per_class.get("classes") or {}
    checked = 0
    for class_id in active_classes:
        row = current.get(str(int(class_id)))
        prev = previous.get(str(int(class_id)))
        if not isinstance(row, dict) or not isinstance(prev, dict):
            continue
        checked += 1
        recall_delta = float(row.get("Recall", 0.0) or 0.0) - float(prev.get("Recall", 0.0) or 0.0)
        ap_delta = float(row.get("AP50_95", 0.0) or 0.0) - float(prev.get("AP50_95", 0.0) or 0.0)
        if recall_delta > 0.0 or ap_delta > 0.0:
            return False
    return checked > 0


def annotate_policy_history_with_gate(
    history: list[dict[str, Any]],
    event: dict[str, Any],
    new_policy: dict[str, Any],
) -> None:
    if not history:
        return
    event_for_history = deepcopy(event)
    event_for_history.pop("policy", None)
    latest = history[-1]
    latest["gated_controller_event"] = event_for_history
    latest["catf_gated_mode"] = True
    latest["gated_fallback_active"] = bool(event_for_history.get("fallback_active_after"))
    latest["gated_positive_gain"] = bool(event_for_history.get("positive_gain"))
    latest["gated_bad_patterns"] = list(event_for_history.get("bad_patterns") or [])
    if event_for_history.get("triggered"):
        latest["action"] = event_for_history.get("action", latest.get("action"))
        latest["accepted_policy"] = deepcopy(new_policy)
        latest["new_policy"] = deepcopy(new_policy)
        latest["guard_triggered"] = list(
            dict.fromkeys(
                list(latest.get("guard_triggered", []) or [])
                + list(event_for_history.get("reasons", []) or [])
                + list(event_for_history.get("bad_patterns", []) or [])
            )
        )
        latest["frozen"] = event_for_history.get("action") == "no_op_freeze"


def value(delta: dict[str, float | None], key: str) -> float:
    raw = delta.get(key)
    return 0.0 if raw is None else float(raw)
