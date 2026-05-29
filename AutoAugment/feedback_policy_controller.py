from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
PHOTOMETRIC_OPS = {"clahe", "gamma", "brightness", "contrast"}
TEXTURE_OPS = {"sharpen_mild", "local_contrast"}
OCCLUSION_OPS = {"cutout_safe"}
COPY_PASTE_OPS = {"copy_paste", "online_copy_paste", "class_balanced_copy_paste"}
INDUSTRIAL_OPS = PHOTOMETRIC_OPS | TEXTURE_OPS | OCCLUSION_OPS

GROUPS = {
    "photometric": PHOTOMETRIC_OPS,
    "texture": TEXTURE_OPS,
    "occlusion": OCCLUSION_OPS,
}
GROUP_BUDGETS = {"photometric": 0.30, "texture": 0.35, "occlusion": 0.08}
TRUST_REGION_PROB = 0.02
TRUST_REGION_STRENGTH = 0.03
WARNING_DROP = -0.01
ROLLBACK_DROP = -0.015

DEFAULT_LIMITS = {
    "clahe": {"min_prob": 0.0, "max_prob": 0.15, "min_strength": 0.0, "max_strength": 0.45},
    "gamma": {"min_prob": 0.0, "max_prob": 0.15, "min_strength": 0.0, "max_strength": 0.45},
    "brightness": {"min_prob": 0.0, "max_prob": 0.10, "min_strength": 0.0, "max_strength": 0.25},
    "contrast": {"min_prob": 0.0, "max_prob": 0.10, "min_strength": 0.0, "max_strength": 0.25},
    "sharpen_mild": {"min_prob": 0.0, "max_prob": 0.20, "min_strength": 0.0, "max_strength": 0.45},
    "local_contrast": {"min_prob": 0.0, "max_prob": 0.18, "min_strength": 0.0, "max_strength": 0.40},
    "cutout_safe": {"min_prob": 0.0, "max_prob": 0.08, "min_strength": 0.0, "max_strength": 0.20},
}

CATF_INITIAL_POLICY = {
    "clahe": {"prob": 0.05, "strength": 0.20},
    "gamma": {"prob": 0.05, "strength": 0.20},
    "brightness": {"prob": 0.03, "strength": 0.15},
    "contrast": {"prob": 0.03, "strength": 0.15},
    "sharpen_mild": {"prob": 0.08, "strength": 0.25},
    "local_contrast": {"prob": 0.06, "strength": 0.20},
    "cutout_safe": {"prob": 0.03, "strength": 0.10},
}


@dataclass
class PolicyAdjustment:
    op: str
    field: str
    before: float
    after: float
    reason: str


@dataclass
class FeedbackPolicyController:
    """Constraint-aware trust-region feedback controller for industrial online aug.

    The controller compares each feedback point with the clean native YOLO default
    reference at the same epoch when available. Proposed policy changes are small,
    budgeted, and only become the new safe policy after the next feedback point
    confirms that Precision and mAP constraints have not been violated.
    """

    policy: dict[str, Any]
    history_dir: Path
    policy_state_path: Path | None = None
    profile: str = "industrial"
    reference_curve: dict[int, dict[str, Any]] = field(default_factory=dict)
    freeze_epoch: int = 40
    feedback_interval: int = 5
    history: list[dict[str, Any]] = field(default_factory=list)
    last_safe_policy: dict[str, Any] = field(default_factory=dict)
    last_safe_policy_id: str = "initial"
    pending_policy_id: str | None = None
    cooldown_remaining: int = 0
    frozen: bool = False
    warning_streak: int = 0
    no_improvement_streak: int = 0
    last_balanced_score: float | None = None
    policy_counter: int = 0

    def __post_init__(self) -> None:
        self.policy = enforce_policy_limits(deepcopy(self.policy))
        self.policy, _ = enforce_group_budgets(self.policy)
        self.last_safe_policy = deepcopy(self.policy)
        self.history_dir = Path(self.history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if self.policy_state_path is not None:
            self.policy_state_path = Path(self.policy_state_path)

    def update(
        self,
        diagnostics: dict[str, Any],
        *,
        stage_index: int | None = None,
        epoch: int | None = None,
        metrics: dict[str, Any] | None = None,
        reference_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        epoch_num = int(epoch if epoch is not None else (stage_index if stage_index is not None else len(self.history) + 1))
        metrics = compact_metrics(metrics or {})
        reference = compact_metrics(reference_metrics or self.reference_curve.get(epoch_num, {}))
        delta_metrics = metric_delta(metrics, reference)
        flags = normalize_diagnostics(diagnostics, metrics=metrics, delta_metrics=delta_metrics)
        diagnosis_summary = summarize_diagnostics(diagnostics, flags)
        old_policy = deepcopy(self.policy)
        budget_before = group_budget_usage(old_policy)

        constraint_warning = has_constraint_warning(delta_metrics)
        rollback_required, rollback_reason = has_rollback_violation(delta_metrics)
        guard_triggered = triggered_guards(flags, delta_metrics, constraint_warning, rollback_required)

        action = "accept"
        accepted_previous = False
        base_policy = deepcopy(self.policy)

        if self.pending_policy_id is not None:
            if rollback_required:
                action = "rollback"
                base_policy = deepcopy(self.last_safe_policy)
                self.policy = deepcopy(base_policy)
                self.cooldown_remaining = 1
            elif constraint_warning:
                action = "shrink"
            else:
                action = "accept"
                accepted_previous = True
                self.last_safe_policy = strip_internal_fields(base_policy)
                self.last_safe_policy_id = self.pending_policy_id
            self.pending_policy_id = None
        elif rollback_required:
            action = "rollback"
            base_policy = deepcopy(self.last_safe_policy)
            self.policy = deepcopy(base_policy)
            self.cooldown_remaining = 1
        elif constraint_warning:
            action = "shrink"

        balanced_score = compute_balanced_score(metrics, delta_metrics)
        if self.last_balanced_score is not None and balanced_score <= self.last_balanced_score + 1e-9:
            self.no_improvement_streak += 1
        else:
            self.no_improvement_streak = 0
        self.last_balanced_score = balanced_score
        self.warning_streak = self.warning_streak + 1 if constraint_warning else 0

        freeze_reason = None
        if epoch_num >= self.freeze_epoch:
            freeze_reason = "epoch_ge_freeze_epoch"
        elif self.warning_streak >= 2:
            freeze_reason = "two_consecutive_constraint_warnings"
        elif self.no_improvement_streak >= 2:
            freeze_reason = "two_consecutive_no_balanced_score_improvement"
        if freeze_reason:
            self.frozen = True
            if action not in {"rollback", "shrink"}:
                action = "freeze"

        trust_region_clipping: list[dict[str, Any]] = []
        budget_clipping: list[dict[str, Any]] = []
        proposed_policy = deepcopy(base_policy)

        if self.frozen:
            accepted_policy = enforce_policy_limits(base_policy)
            accepted_policy, budget_clipping = enforce_group_budgets(accepted_policy)
            self.policy = deepcopy(accepted_policy)
            proposed_policy = deepcopy(accepted_policy)
        elif self.cooldown_remaining > 0 and action != "rollback":
            action = "cooldown"
            desired = cooldown_policy(base_policy)
            proposed_policy, trust_region_clipping = apply_trust_region(base_policy, desired)
            proposed_policy, budget_clipping = enforce_group_budgets(proposed_policy)
            self.policy = deepcopy(proposed_policy)
            self.cooldown_remaining -= 1
            self.pending_policy_id = self._next_policy_id(epoch_num)
            accepted_policy = deepcopy(proposed_policy)
        elif action == "rollback":
            accepted_policy = deepcopy(base_policy)
        else:
            desired = propose_policy_update(base_policy, flags=flags, delta_metrics=delta_metrics)
            proposed_policy, trust_region_clipping = apply_trust_region(base_policy, desired)
            proposed_policy, budget_clipping = enforce_group_budgets(proposed_policy)
            self.policy = deepcopy(proposed_policy)
            self.pending_policy_id = self._next_policy_id(epoch_num)
            accepted_policy = deepcopy(proposed_policy)

        adjustments = policy_diff(old_policy, accepted_policy)
        proposed_policy = strip_internal_fields(proposed_policy)
        accepted_policy = strip_internal_fields(accepted_policy)
        self.policy = strip_internal_fields(self.policy)
        budget_after = group_budget_usage(accepted_policy)
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch_num,
            "stage_index": stage_index,
            "profile": self.profile,
            "metrics": metrics,
            "reference_metrics": reference,
            "delta_metrics": delta_metrics,
            "diagnosis_summary": diagnosis_summary,
            "normalized_flags": flags,
            "old_policy": old_policy,
            "proposed_policy": proposed_policy,
            "accepted_policy": accepted_policy,
            "last_safe_policy_id": self.last_safe_policy_id,
            "pending_policy_id": self.pending_policy_id,
            "accepted_previous_policy": accepted_previous,
            "action": action,
            "guard_triggered": guard_triggered,
            "constraint_warning": constraint_warning,
            "rollback_reason": rollback_reason if action == "rollback" else None,
            "frozen": self.frozen,
            "freeze_reason": freeze_reason,
            "cooldown_remaining": self.cooldown_remaining,
            "group_budget_before": budget_before,
            "group_budget_after": budget_after,
            "trust_region_clipping": trust_region_clipping,
            "group_budget_clipping": budget_clipping,
            "adjustments": [item.__dict__ for item in adjustments],
            "copy_paste_status": "pending_object_bank_design",
        }
        self.history.append(record)
        self._write_outputs()
        return deepcopy(self.policy)

    def _next_policy_id(self, epoch_num: int) -> str:
        self.policy_counter += 1
        return f"catf_epoch_{epoch_num:03d}_{self.policy_counter:03d}"

    def _write_outputs(self) -> None:
        write_json(self.history_dir / "policy_history.json", {"history": self.history, "latest_policy": self.policy})
        write_markdown(self.history_dir / "policy_history.md", render_history_markdown(self.history))
        write_history_csv(self.history_dir / "policy_history.csv", self.history)
        if self.policy_state_path is not None:
            write_json(self.policy_state_path, self.policy)


def default_catf_policy() -> dict[str, Any]:
    return {
        "policy_id": "catf_yolo_default_industrial_feedback",
        "controller": "CATF",
        "copy_paste_status": "pending_object_bank_design",
        "notes": [
            "Ultralytics YOLO default augmentation remains enabled.",
            "CATF controls only additional low-strength industrial online operations.",
            "Policy updates are constrained by reference curve, trust region, group budgets, delayed acceptance, rollback, cooldown, and freeze.",
        ],
        "operations": [
            op_payload("clahe", params={"clip_limit": 2.0, "tile_grid_size": [8, 8]}),
            op_payload("gamma"),
            op_payload("brightness", params={"max_delta": 0.12}),
            op_payload("contrast", params={"alpha": 0.15}),
            op_payload("sharpen_mild", params={"amount": 0.6}),
            op_payload("local_contrast"),
            op_payload("cutout_safe", params={"max_holes": 2, "max_fraction": 0.12, "max_bbox_overlap": 0.05}),
        ],
    }


def op_payload(name: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    base = CATF_INITIAL_POLICY[name]
    limits = DEFAULT_LIMITS[name]
    return {
        "name": name,
        "prob": float(base["prob"]),
        "base_prob": float(base["prob"]),
        "min_prob": float(limits["min_prob"]),
        "max_prob": float(limits["max_prob"]),
        "strength": float(base["strength"]),
        "base_strength": float(base["strength"]),
        "min_strength": float(limits["min_strength"]),
        "max_strength": float(limits["max_strength"]),
        "params": params or {},
    }


def propose_policy_update(policy: dict[str, Any], *, flags: dict[str, bool], delta_metrics: dict[str, float | None]) -> dict[str, Any]:
    proposed = deepcopy(policy)
    precision_guard = (delta_metrics.get("precision") is not None and float(delta_metrics["precision"]) < -0.01) or flags.get("fp_high", False)
    map95_guard = delta_metrics.get("map50_95") is not None and float(delta_metrics["map50_95"]) < -0.01

    if precision_guard:
        for name in ["brightness", "contrast", "clahe", "gamma", "cutout_safe"]:
            add_delta(proposed, name, prob_delta=-0.02, strength_delta=-0.03, reason="precision_guard")
        return proposed

    if map95_guard:
        for name in ["cutout_safe", "brightness", "contrast"]:
            add_delta(proposed, name, prob_delta=-0.02, strength_delta=-0.03, reason="map50_95_guard")
        for name in ["sharpen_mild", "local_contrast"]:
            add_delta(proposed, name, prob_delta=0.01, strength_delta=0.015, reason="map50_95_guard_texture")

    delta_p = delta_metrics.get("precision")
    if flags.get("low_contrast_fn_high", False) and (delta_p is None or float(delta_p) >= -0.005):
        for name in ["sharpen_mild", "local_contrast"]:
            add_delta(proposed, name, prob_delta=0.02, strength_delta=0.03, reason="low_contrast_fn_high_texture_first")
        for name in ["gamma", "clahe"]:
            add_delta(proposed, name, prob_delta=0.01, strength_delta=0.015, reason="low_contrast_fn_high_photometric_safe")

    delta_r = delta_metrics.get("recall")
    if delta_r is not None and float(delta_r) < -0.01:
        factor = 0.5 if flags.get("fp_high", False) else 1.0
        for name in ["gamma", "clahe", "sharpen_mild", "local_contrast"]:
            add_delta(proposed, name, prob_delta=0.01 * factor, strength_delta=0.015 * factor, reason="recall_low_reference_curve")

    return proposed


def shrink_risk_policy(policy: dict[str, Any], *, reason: str) -> dict[str, Any]:
    shrunk = deepcopy(policy)
    for name in ["brightness", "contrast", "clahe", "gamma", "cutout_safe"]:
        add_delta(shrunk, name, prob_delta=-0.02, strength_delta=-0.03, reason=reason)
    return shrunk


def cooldown_policy(policy: dict[str, Any]) -> dict[str, Any]:
    cooled = shrink_risk_policy(policy, reason="cooldown_shrink_risk_ops")
    for name in ["sharpen_mild", "local_contrast"]:
        add_delta(cooled, name, prob_delta=0.005, strength_delta=0.005, reason="cooldown_texture_only")
    return cooled


def add_delta(policy: dict[str, Any], name: str, *, prob_delta: float, strength_delta: float, reason: str) -> None:
    op = find_or_create_operation(policy, name)
    op["_proposal_reasons"] = list(op.get("_proposal_reasons", [])) + [reason]
    op["prob"] = float(op.get("prob", 0.0) or 0.0) + float(prob_delta)
    op["strength"] = float(op.get("strength", 0.0) or 0.0) + float(strength_delta)


def apply_trust_region(old_policy: dict[str, Any], desired_policy: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clipped_policy = deepcopy(desired_policy)
    clipping: list[dict[str, Any]] = []
    old_by_name = operations_by_name(old_policy)
    for op in clipped_policy.get("operations", []):
        name = str(op.get("name", ""))
        old_op = old_by_name.get(name, {})
        for field_name, max_step in [("prob", TRUST_REGION_PROB), ("strength", TRUST_REGION_STRENGTH)]:
            before = float(old_op.get(field_name, op.get(field_name, 0.0)) or 0.0)
            desired = float(op.get(field_name, before) or 0.0)
            limited = _clip(desired, before - max_step, before + max_step)
            if limited != desired:
                clipping.append({"op": name, "field": field_name, "before": before, "desired": desired, "after": limited, "max_step": max_step})
            op[field_name] = limited
    return enforce_policy_limits(clipped_policy), clipping


def enforce_policy_limits(policy: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(policy)
    for op in out.setdefault("operations", []):
        name = str(op.get("name", ""))
        if name in DEFAULT_LIMITS:
            for key, value in DEFAULT_LIMITS[name].items():
                op.setdefault(key, value)
            op["prob"] = _clip(float(op.get("prob", 0.0) or 0.0), float(op["min_prob"]), float(op["max_prob"]))
            op["strength"] = _clip(float(op.get("strength", 0.0) or 0.0), float(op["min_strength"]), float(op["max_strength"]))
        elif name in COPY_PASTE_OPS:
            op["prob"] = 0.0
            op["strength"] = 0.0
            op["status"] = "pending_object_bank_design"
    return out


def enforce_group_budgets(policy: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    out = enforce_policy_limits(policy)
    clipping: list[dict[str, Any]] = []
    by_name = operations_by_name(out)
    for group_name, names in GROUPS.items():
        budget = GROUP_BUDGETS[group_name]
        total = sum(float(by_name[name].get("prob", 0.0) or 0.0) for name in names if name in by_name)
        if total <= budget or total <= 0.0:
            continue
        scale = budget / total
        for name in names:
            if name not in by_name:
                continue
            before = float(by_name[name].get("prob", 0.0) or 0.0)
            after = before * scale
            by_name[name]["prob"] = after
            clipping.append({"group": group_name, "op": name, "field": "prob", "before": before, "after": after, "budget": budget, "total_before": total})
    return enforce_policy_limits(out), clipping


def find_or_create_operation(policy: dict[str, Any], name: str) -> dict[str, Any]:
    for op in policy.setdefault("operations", []):
        if str(op.get("name", "")) == name:
            return op
    if name not in DEFAULT_LIMITS:
        op = {"name": name, "prob": 0.0, "strength": 0.0, "status": "pending_object_bank_design"}
    else:
        op = op_payload(name)
        op["prob"] = 0.0
        op["strength"] = 0.0
    policy["operations"].append(op)
    return op


def operations_by_name(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(op.get("name", "")): op for op in policy.get("operations", []) if isinstance(op, dict)}


def group_budget_usage(policy: dict[str, Any]) -> dict[str, Any]:
    by_name = operations_by_name(policy)
    usage: dict[str, Any] = {}
    for group_name, names in GROUPS.items():
        total = sum(float(by_name[name].get("prob", 0.0) or 0.0) for name in names if name in by_name)
        usage[group_name] = {"used": total, "budget": GROUP_BUDGETS[group_name], "within_budget": total <= GROUP_BUDGETS[group_name] + 1e-12}
    return usage


def policy_diff(old_policy: dict[str, Any], new_policy: dict[str, Any]) -> list[PolicyAdjustment]:
    old_by_name = operations_by_name(old_policy)
    changes: list[PolicyAdjustment] = []
    for name, new_op in operations_by_name(new_policy).items():
        old_op = old_by_name.get(name, {})
        reasons = ",".join(new_op.get("_proposal_reasons", [])) or "catf_update"
        for field_name in ("prob", "strength"):
            before = float(old_op.get(field_name, 0.0) or 0.0)
            after = float(new_op.get(field_name, 0.0) or 0.0)
            if abs(after - before) > 1e-12:
                changes.append(PolicyAdjustment(name, field_name, before, after, reasons))
    return changes


def strip_internal_fields(policy: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(policy)
    for op in clean.get("operations", []):
        if isinstance(op, dict):
            op.pop("_proposal_reasons", None)
    return clean


def normalize_diagnostics(
    diagnostics: dict[str, Any],
    *,
    metrics: dict[str, Any] | None = None,
    delta_metrics: dict[str, float | None] | None = None,
) -> dict[str, bool]:
    metrics = metrics or {}
    delta_metrics = delta_metrics or {}
    precision = _optional_float(metrics.get("precision"))
    recall = _optional_float(metrics.get("recall"))
    map50 = _optional_float(metrics.get("map50"))
    map50_95 = _optional_float(metrics.get("map50_95"))
    global_diag = diagnostics.get("global", {}) or {}
    vector = diagnostics.get("diagnosis_vector", {}) or {}
    issues = {str(item.get("type")) for item in diagnostics.get("issues", []) if isinstance(item, dict)}
    tp = int(global_diag.get("tp", 0) or 0)
    fp = int(global_diag.get("fp", 0) or 0)
    fn = int(global_diag.get("fn", 0) or 0)
    low_contrast_score = float((vector.get("low_contrast_score") or {}).get("score", 0.0) or 0.0)
    flags = {
        "low_contrast_fn_high": bool(
            diagnostics.get("low_contrast_fn_high")
            or diagnostics.get("low_contrast_missed_defect")
            or low_contrast_score > 0.15
            or "low_contrast_missed_defect" in issues
        ),
        "recall_low": bool(diagnostics.get("recall_low") or (delta_metrics.get("recall") is not None and float(delta_metrics["recall"]) < -0.01)),
        "fn_high": bool(diagnostics.get("fn_high") or (tp + fn > 0 and fn / max(1, tp + fn) > 0.18)),
        "precision_low": bool(diagnostics.get("precision_low") or (delta_metrics.get("precision") is not None and float(delta_metrics["precision"]) < -0.01)),
        "fp_high": bool(diagnostics.get("fp_high") or diagnostics.get("false_positive_high") or (tp + fp > 0 and fp / max(1, tp + fp) > 0.25)),
        "map50_high_map95_low": bool(
            diagnostics.get("map50_high_map95_low")
            or (map50 is not None and map50_95 is not None and map50 >= 0.60 and (map50 - map50_95) >= 0.18)
        ),
        "localization_weak": bool(diagnostics.get("localization_weak") or diagnostics.get("localization_bias")),
        "class_imbalance": bool(diagnostics.get("class_imbalance")),
        "low_support": bool(diagnostics.get("low_support")),
    }
    if recall is not None and recall < float(diagnostics.get("recall_threshold", 0.70)):
        flags["recall_low"] = True
    if precision is not None and precision < float(diagnostics.get("precision_threshold", 0.72)):
        flags["precision_low"] = True
    return flags


def infer_feedback_diagnostics(metrics: dict[str, Any], *, profile: str = "industrial") -> dict[str, Any]:
    precision = _optional_float(metrics.get("precision")) or 0.0
    recall = _optional_float(metrics.get("recall")) or 0.0
    map50 = _optional_float(metrics.get("map50")) or 0.0
    map50_95 = _optional_float(metrics.get("map50_95")) or 0.0
    return {
        "profile": profile,
        "recall_low": recall < 0.70,
        "precision_low": precision < 0.72,
        "fn_high": recall < 0.68,
        "fp_high": precision < 0.68,
        "map50_high_map95_low": map50 >= 0.60 and (map50 - map50_95) >= 0.18,
        "localization_weak": map50 >= 0.60 and (map50 - map50_95) >= 0.20,
        "low_contrast_fn_high": profile in {"industrial", "low_contrast"} and recall < 0.72,
        "class_imbalance": any((row.get("instances") or 0) < 20 for row in metrics.get("per_class", []) if isinstance(row, dict)),
        "low_support": any((row.get("recall") or 1.0) < 0.50 for row in metrics.get("per_class", []) if isinstance(row, dict)),
    }


def summarize_diagnostics(diagnostics: dict[str, Any], flags: dict[str, bool]) -> dict[str, Any]:
    return {
        "global": diagnostics.get("global", {}),
        "active_flags": [name for name, value in flags.items() if value],
        "issue_types": [str(item.get("type")) for item in diagnostics.get("issues", []) if isinstance(item, dict)],
    }


def triggered_guards(
    flags: dict[str, bool],
    delta_metrics: dict[str, float | None],
    constraint_warning: bool,
    rollback_required: bool,
) -> list[str]:
    guards: list[str] = []
    if (delta_metrics.get("precision") is not None and float(delta_metrics["precision"]) < -0.01) or flags.get("fp_high", False):
        guards.append("precision_guard")
    if delta_metrics.get("map50_95") is not None and float(delta_metrics["map50_95"]) < -0.01:
        guards.append("map50_95_guard")
    if flags.get("low_contrast_fn_high", False):
        guards.append("low_contrast_fn")
    if delta_metrics.get("recall") is not None and float(delta_metrics["recall"]) < -0.01:
        guards.append("recall_low")
    if constraint_warning:
        guards.append("constraint_warning")
    if rollback_required:
        guards.append("rollback")
    return guards


def has_constraint_warning(delta_metrics: dict[str, float | None]) -> bool:
    return any(delta_metrics.get(key) is not None and float(delta_metrics[key]) < WARNING_DROP for key in ("precision", "map50", "map50_95"))


def has_rollback_violation(delta_metrics: dict[str, float | None]) -> tuple[bool, str | None]:
    reasons = []
    for key in ("precision", "map50", "map50_95"):
        value = delta_metrics.get(key)
        if value is not None and float(value) < ROLLBACK_DROP:
            reasons.append(f"{key}_drop_lt_{abs(ROLLBACK_DROP):.3f}")
    return bool(reasons), ",".join(reasons) if reasons else None


def compute_balanced_score(metrics: dict[str, Any], delta_metrics: dict[str, float | None]) -> float:
    score = 0.0
    for key in METRIC_KEYS:
        score += float(metrics.get(key, 0.0) or 0.0)
    for key in ("precision", "map50", "map50_95"):
        value = delta_metrics.get(key)
        if value is not None and float(value) < 0.0:
            score += 2.0 * float(value)
    return score


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics and metrics.get(key) is not None}


def metric_delta(metrics: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | None]:
    delta: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        ref = reference.get(key)
        delta[key] = None if value is None or ref is None else float(value) - float(ref)
    return delta


def render_history_markdown(history: list[dict[str, Any]]) -> str:
    lines = [
        "# CATF Policy History",
        "",
        "| epoch | action | frozen | guards | adjustments | rollback_reason |",
        "|---:|---|---|---|---:|---|",
    ]
    for record in history:
        lines.append(
            f"| {record.get('epoch')} | {record.get('action')} | {str(record.get('frozen')).lower()} | "
            f"{','.join(record.get('guard_triggered', [])) or 'none'} | {len(record.get('adjustments', []))} | "
            f"{record.get('rollback_reason') or ''} |"
        )
    lines.extend(["", "## Adjustment Details", ""])
    for record in history:
        lines.append(f"### Epoch {record.get('epoch')} - {record.get('action')}")
        lines.append(f"- Delta metrics: `{record.get('delta_metrics')}`")
        lines.append(f"- Group budget before: `{record.get('group_budget_before')}`")
        lines.append(f"- Group budget after: `{record.get('group_budget_after')}`")
        if record.get("trust_region_clipping"):
            lines.append(f"- Trust-region clipping: `{record.get('trust_region_clipping')}`")
        if record.get("group_budget_clipping"):
            lines.append(f"- Group budget clipping: `{record.get('group_budget_clipping')}`")
        if not record.get("adjustments"):
            lines.append("- No policy changes.")
        for adjustment in record.get("adjustments", []):
            lines.append(
                f"- `{adjustment['op']}` {adjustment['field']}: "
                f"{float(adjustment['before']):.4f} -> {float(adjustment['after']):.4f} ({adjustment['reason']})"
            )
    return "\n".join(lines) + "\n"


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "action",
        "op",
        "field",
        "before",
        "after",
        "reason",
        "delta_precision",
        "delta_recall",
        "delta_map50",
        "delta_map50_95",
        "guard_triggered",
        "rollback_reason",
        "frozen",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            adjustments = record.get("adjustments") or [{}]
            for adjustment in adjustments:
                writer.writerow(
                    {
                        "epoch": record.get("epoch"),
                        "action": record.get("action"),
                        "op": adjustment.get("op"),
                        "field": adjustment.get("field"),
                        "before": adjustment.get("before"),
                        "after": adjustment.get("after"),
                        "reason": adjustment.get("reason"),
                        "delta_precision": (record.get("delta_metrics") or {}).get("precision"),
                        "delta_recall": (record.get("delta_metrics") or {}).get("recall"),
                        "delta_map50": (record.get("delta_metrics") or {}).get("map50"),
                        "delta_map50_95": (record.get("delta_metrics") or {}).get("map50_95"),
                        "guard_triggered": ",".join(record.get("guard_triggered", [])),
                        "rollback_reason": record.get("rollback_reason"),
                        "frozen": record.get("frozen"),
                    }
                )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
