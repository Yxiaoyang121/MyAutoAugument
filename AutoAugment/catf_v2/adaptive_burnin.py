from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from AutoAugment.catf_v2.safe_controller import force_noop_policy


METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
LOW_RISK_ROI_OPS = {"sharpen_mild", "local_contrast"}


@dataclass
class AdaptiveBurninConfig:
    min_burnin_epoch: int = 5
    max_burnin_epoch: int = 15
    burnin_check_interval: int = 5
    metric_stability_window: int = 3
    map50_stability_threshold: float = 0.02
    recall_stability_threshold: float = 0.03
    min_diagnosis_evidence: int = 5
    min_active_class_evidence: int = 5
    allow_force_start_at_max_burnin: bool = True
    probe_window: int | None = None


@dataclass
class AdaptiveBurninController:
    """Adaptive burn-in trigger for CATF-v2 candidate branches.

    The controller keeps CATF-v2 in strict observe mode until the validation
    curve is stable enough and diagnosis evidence is strong enough to justify a
    candidate branch. Until then, the returned policy must remain no-op.
    """

    config: AdaptiveBurninConfig = field(default_factory=AdaptiveBurninConfig)
    state: str = "burnin_observe"
    triggered: bool = False
    start_epoch: int | None = None
    fallback_active: bool = False
    fallback_reason: str | None = None
    low_risk_forced_start: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def evaluate(
        self,
        *,
        epoch: int,
        metrics: dict[str, Any],
        reference_metrics: dict[str, Any],
        clean_reference_metrics: dict[str, Any] | None,
        metric_history: list[dict[str, Any]],
        per_class_diagnosis: dict[str, Any] | None,
        old_policy: dict[str, Any],
        proposed_policy: dict[str, Any],
        close_mosaic_start_epoch: int | None,
    ) -> dict[str, Any]:
        epoch = int(epoch)
        state_before = self.state
        condition = evaluate_start_condition(
            epoch=epoch,
            config=self.config,
            metrics=metrics,
            reference_metrics=reference_metrics,
            clean_reference_metrics=clean_reference_metrics,
            metric_history=metric_history,
            per_class_diagnosis=per_class_diagnosis,
            close_mosaic_start_epoch=close_mosaic_start_epoch,
        )
        event = {
            "epoch": epoch,
            "adaptive_burnin": True,
            "state_before": state_before,
            "state_after": self.state,
            "checked": should_check_burnin(epoch, self.config),
            "action": "burnin_observe",
            "triggered": False,
            "candidate_branch_started": False,
            "strict_noop": False,
            "start_condition_met": bool(condition["start_condition_met"]),
            "start_condition": condition,
            "reasons": list(condition["blocking_reasons"]),
            "probe_window": probe_window(self.config),
            "fallback_active_before": bool(self.fallback_active),
            "fallback_active_after": bool(self.fallback_active),
            "low_risk_policy": False,
        }

        if self.triggered:
            event["action"] = "candidate_branch"
            event["state_after"] = self.state
            event["policy"] = deepcopy(proposed_policy)
            self._remember(event)
            return event

        if self.fallback_active:
            return self._noop_event(event, old_policy, self.fallback_reason or "adaptive_burnin_fallback_active")

        if not event["checked"]:
            event["reasons"] = ["not_burnin_check_epoch"]
            event["policy"] = deepcopy(old_policy)
            self._remember(event)
            return event

        if condition["start_condition_met"]:
            return self._start_candidate(event, proposed_policy, epoch, low_risk=False, reasons=["adaptive_burnin_ready"])

        if epoch >= int(self.config.max_burnin_epoch):
            if (
                bool(self.config.allow_force_start_at_max_burnin)
                and condition["medium_confidence_issue_classes"]
                and not condition["strong_clean_baseline_protection"]
            ):
                low_risk_policy = build_low_risk_roi_policy(
                    proposed_policy,
                    eligible_class_ids=condition["medium_confidence_issue_classes"],
                    reason="adaptive_burnin_force_start_low_risk_at_max",
                )
                return self._start_candidate(
                    event,
                    low_risk_policy,
                    epoch,
                    low_risk=True,
                    reasons=["force_start_at_max_burnin_low_risk_issue"],
                )
            return self._noop_event(event, old_policy, "adaptive_burnin_not_ready")

        event["policy"] = deepcopy(old_policy)
        self._remember(event)
        return event

    def _start_candidate(
        self,
        event: dict[str, Any],
        policy: dict[str, Any],
        epoch: int,
        *,
        low_risk: bool,
        reasons: list[str],
    ) -> dict[str, Any]:
        self.triggered = True
        self.start_epoch = int(epoch)
        self.state = "candidate_branch"
        self.low_risk_forced_start = bool(low_risk)
        event["triggered"] = True
        event["candidate_branch_started"] = True
        event["action"] = "start_candidate_low_risk" if low_risk else "start_candidate"
        event["state_after"] = self.state
        event["reasons"] = reasons
        event["policy"] = deepcopy(policy)
        event["low_risk_policy"] = bool(low_risk)
        self._remember(event)
        return event

    def _noop_event(self, event: dict[str, Any], policy: dict[str, Any], reason: str) -> dict[str, Any]:
        self.fallback_active = True
        self.fallback_reason = reason
        self.state = "no_op_fallback"
        event["triggered"] = True
        event["action"] = "no_op_fallback"
        event["state_after"] = self.state
        event["reasons"] = [reason]
        event["policy"] = force_noop_policy(policy, reason=reason)
        event["strict_noop"] = True
        event["fallback_active_after"] = True
        self._remember(event)
        return event

    def _remember(self, event: dict[str, Any]) -> None:
        stored = deepcopy(event)
        stored.pop("policy", None)
        self.events.append(stored)


def evaluate_start_condition(
    *,
    epoch: int,
    config: AdaptiveBurninConfig,
    metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    clean_reference_metrics: dict[str, Any] | None,
    metric_history: list[dict[str, Any]],
    per_class_diagnosis: dict[str, Any] | None,
    close_mosaic_start_epoch: int | None,
) -> dict[str, Any]:
    epoch = int(epoch)
    reasons: list[str] = []
    condition_results: dict[str, bool] = {}

    condition_results["min_burnin_epoch"] = epoch >= int(config.min_burnin_epoch)
    if not condition_results["min_burnin_epoch"]:
        reasons.append("before_min_burnin_epoch")

    close_ok = close_mosaic_start_epoch is None or epoch < int(close_mosaic_start_epoch)
    condition_results["before_close_mosaic_start"] = close_ok
    if not close_ok:
        reasons.append("after_close_mosaic_start")

    stability = metric_stability(
        metric_history,
        window=int(config.metric_stability_window),
        map50_threshold=float(config.map50_stability_threshold),
        recall_threshold=float(config.recall_stability_threshold),
    )
    condition_results["metric_stable"] = bool(stability["stable"])
    if not stability["stable"]:
        reasons.append(str(stability["reason"]))

    evidence = diagnosis_evidence(per_class_diagnosis, config=config)
    condition_results["diagnosis_evidence_sufficient"] = bool(evidence["eligible_active_classes"])
    if not condition_results["diagnosis_evidence_sufficient"]:
        reasons.append("insufficient_diagnosis_evidence")

    condition_results["not_random_small_sample"] = not bool(evidence["small_sample_only"])
    if not condition_results["not_random_small_sample"]:
        reasons.append("random_small_sample_only")

    strong_baseline = strong_clean_baseline_protection(
        clean_reference_metrics or reference_metrics,
        evidence=evidence,
        metric_stable=bool(stability["stable"]),
    )
    condition_results["not_strong_clean_baseline_protected"] = not strong_baseline
    if strong_baseline:
        reasons.append("strong_clean_baseline_protection")

    met = all(condition_results.values())
    return {
        "start_condition_met": bool(met),
        "condition_results": condition_results,
        "blocking_reasons": list(dict.fromkeys(reasons)),
        "metric_stability": stability,
        "eligible_active_classes": evidence["eligible_active_classes"],
        "medium_confidence_issue_classes": evidence["medium_confidence_issue_classes"],
        "rejected_class_reasons": evidence["rejected_class_reasons"],
        "small_sample_only": bool(evidence["small_sample_only"]),
        "high_confidence_active_issue": bool(evidence["high_confidence_active_issue"]),
        "strong_clean_baseline_protection": bool(strong_baseline),
        "metrics": compact_metrics(metrics),
        "reference_metrics": compact_metrics(reference_metrics),
        "clean_reference_metrics": compact_metrics(clean_reference_metrics or reference_metrics),
    }


def should_check_burnin(epoch: int, config: AdaptiveBurninConfig) -> bool:
    epoch = int(epoch)
    if epoch < int(config.min_burnin_epoch):
        return False
    interval = max(1, int(config.burnin_check_interval))
    return epoch % interval == 0 or epoch >= int(config.max_burnin_epoch)


def metric_stability(
    metric_history: list[dict[str, Any]],
    *,
    window: int,
    map50_threshold: float,
    recall_threshold: float,
) -> dict[str, Any]:
    window = max(1, int(window))
    history = [compact_metrics(row) for row in metric_history if row]
    if len(history) < window:
        return {
            "stable": False,
            "reason": "insufficient_history",
            "window": window,
            "history_count": len(history),
            "map50_range": None,
            "recall_range": None,
        }
    recent = history[-window:]
    map_values = [float(row["map50"]) for row in recent if row.get("map50") is not None]
    recall_values = [float(row["recall"]) for row in recent if row.get("recall") is not None]
    if len(map_values) < window or len(recall_values) < window:
        return {
            "stable": False,
            "reason": "missing_metric_history",
            "window": window,
            "history_count": len(history),
            "map50_range": None,
            "recall_range": None,
        }
    map_range = max(map_values) - min(map_values)
    recall_range = max(recall_values) - min(recall_values)
    stable = map_range <= float(map50_threshold) and recall_range <= float(recall_threshold)
    reason = "stable" if stable else "metric_unstable"
    return {
        "stable": bool(stable),
        "reason": reason,
        "window": window,
        "history_count": len(history),
        "map50_range": round(map_range, 6),
        "recall_range": round(recall_range, 6),
        "map50_threshold": float(map50_threshold),
        "recall_threshold": float(recall_threshold),
        "recent": recent,
    }


def diagnosis_evidence(per_class_diagnosis: dict[str, Any] | None, *, config: AdaptiveBurninConfig) -> dict[str, Any]:
    eligible: list[int] = []
    medium: list[int] = []
    rejected: dict[str, list[str]] = {}
    small_sample_only = True
    high_confidence_active_issue = False
    rows = (per_class_diagnosis or {}).get("classes") or {}
    for raw_id, row in rows.items():
        class_id = int(raw_id)
        reasons = class_rejection_reasons(row, config=config)
        if int(row.get("val_instances", 0) or 0) >= 10 and int(row.get("evidence_count", 0) or 0) >= int(config.min_diagnosis_evidence):
            small_sample_only = False
        if not reasons:
            eligible.append(class_id)
        else:
            rejected[str(class_id)] = reasons
        if medium_confidence_issue(row, config=config):
            medium.append(class_id)
        if critical_high_confidence_issue(row, config=config):
            high_confidence_active_issue = True
    return {
        "eligible_active_classes": eligible,
        "medium_confidence_issue_classes": medium,
        "rejected_class_reasons": rejected,
        "small_sample_only": bool(rows) and small_sample_only,
        "high_confidence_active_issue": high_confidence_active_issue,
    }


def class_rejection_reasons(row: dict[str, Any], *, config: AdaptiveBurninConfig) -> list[str]:
    reasons: list[str] = []
    evidence_count = int(row.get("evidence_count", 0) or 0)
    val_instances = int(row.get("val_instances", 0) or 0)
    confidence = float(row.get("diagnosis_confidence", 0.0) or 0.0)
    fn_count = int(row.get("FN", 0) or 0)
    ap95 = float(row.get("AP50_95", 1.0) or 0.0)
    if evidence_count < int(config.min_active_class_evidence):
        reasons.append("evidence_count_below_min_active_class_evidence")
    if confidence < 0.50:
        reasons.append("diagnosis_confidence_below_0.50")
    if fn_count < 5 and not bool(row.get("low_ap50_95", False)) and ap95 > 0.45:
        reasons.append("no_fn_or_ap95_shortfall")
    if bool(row.get("no_aug_class", False)) and not bool(row.get("no_aug_exception_allowed", False)):
        reasons.append("no_aug_class")
    if bool(row.get("stable_class", False)):
        reasons.append("stable_class")
    if bool(row.get("high_fp_guarded", False)) or bool(row.get("high_fp", False)):
        reasons.append("high_fp_guarded")
    if bool(row.get("low_support", False)) or val_instances < 10:
        reasons.append("low_support_or_val_instances_lt_10")
    return reasons


def medium_confidence_issue(row: dict[str, Any], *, config: AdaptiveBurninConfig) -> bool:
    if bool(row.get("no_aug_class", False)) and not bool(row.get("no_aug_exception_allowed", False)):
        return False
    if bool(row.get("stable_class", False)):
        return False
    if bool(row.get("high_fp_guarded", False)) or bool(row.get("high_fp", False)):
        return False
    if bool(row.get("low_support", False)) or int(row.get("val_instances", 0) or 0) < 10:
        return False
    if int(row.get("evidence_count", 0) or 0) < int(config.min_diagnosis_evidence):
        return False
    if float(row.get("diagnosis_confidence", 0.0) or 0.0) < 0.45:
        return False
    return int(row.get("FN", 0) or 0) >= 5 or float(row.get("AP50_95", 1.0) or 0.0) <= 0.50


def critical_high_confidence_issue(row: dict[str, Any], *, config: AdaptiveBurninConfig) -> bool:
    if not medium_confidence_issue(row, config=config):
        return False
    if float(row.get("diagnosis_confidence", 0.0) or 0.0) < 0.85:
        return False
    if int(row.get("evidence_count", 0) or 0) < max(10, 2 * int(config.min_diagnosis_evidence)):
        return False
    return int(row.get("FN", 0) or 0) >= 10 and float(row.get("AP50_95", 1.0) or 0.0) <= 0.20


def strong_clean_baseline_protection(reference_metrics: dict[str, Any], *, evidence: dict[str, Any], metric_stable: bool = True) -> bool:
    strong = (
        float(reference_metrics.get("recall", 0.0) or 0.0) >= 0.72
        and float(reference_metrics.get("map50", 0.0) or 0.0) >= 0.76
        and float(reference_metrics.get("map50_95", 0.0) or 0.0) >= 0.52
    )
    return bool(strong and (not metric_stable or not evidence.get("high_confidence_active_issue")))


def build_low_risk_roi_policy(policy: dict[str, Any], *, eligible_class_ids: list[int], reason: str) -> dict[str, Any]:
    matrix = deepcopy(policy)
    allowed = {int(class_id) for class_id in eligible_class_ids}
    matrix.setdefault("adaptive_burnin", {})["low_risk_policy"] = True
    matrix["adaptive_burnin"]["reason"] = reason
    for raw_id, row in (matrix.get("classes") or {}).items():
        class_id = int(raw_id)
        row["state"] = "pending" if class_id in allowed else "accepted"
        if class_id in allowed:
            row["status"] = "active"
        for op_name, op in (row.get("ops") or {}).items():
            if class_id in allowed and op_name in LOW_RISK_ROI_OPS:
                op["prob"] = round(min(float(op.get("prob", 0.0) or 0.0), 0.01), 6)
                op["strength"] = round(min(float(op.get("strength", 0.0) or 0.0), 0.20), 6)
            else:
                op["prob"] = 0.0
                op["strength"] = 0.0
    return matrix


def annotate_policy_history_with_adaptive_burnin(
    history: list[dict[str, Any]],
    event: dict[str, Any],
    new_policy: dict[str, Any],
    *,
    old_policy: dict[str, Any] | None = None,
) -> None:
    if not history:
        return
    latest = history[-1]
    event_for_history = deepcopy(event)
    event_for_history.pop("policy", None)
    latest["adaptive_burnin"] = True
    latest["adaptive_burnin_event"] = event_for_history
    latest["adaptive_burnin_state"] = event_for_history.get("state_after")
    latest["adaptive_start_condition_met"] = bool(event_for_history.get("start_condition_met"))
    latest["adaptive_candidate_branch_started"] = bool(event_for_history.get("candidate_branch_started"))
    latest["adaptive_strict_noop"] = bool(event_for_history.get("strict_noop"))
    latest["accepted_policy"] = deepcopy(new_policy)
    latest["new_policy"] = deepcopy(new_policy)
    if old_policy is not None:
        latest["old_policy_before_adaptive_burnin"] = deepcopy(old_policy)
    if event_for_history.get("action") in {"burnin_observe", "no_op_fallback"}:
        latest["policy_update_applied"] = False
        latest["industrial_aug_applied"] = False
        latest["roi_aug_applied"] = False
        latest["random_draws_allowed"] = False
        latest["action"] = event_for_history.get("action")
    if event_for_history.get("strict_noop"):
        latest["frozen"] = True
        latest["guard_triggered"] = list(
            dict.fromkeys(list(latest.get("guard_triggered", []) or []) + list(event_for_history.get("reasons", []) or []))
        )


def probe_window(config: AdaptiveBurninConfig) -> int:
    if config.probe_window is not None:
        return max(1, int(config.probe_window))
    return max(1, int(config.burnin_check_interval))


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {key: metrics.get(key) for key in METRIC_KEYS}
