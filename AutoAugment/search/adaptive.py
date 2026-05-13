from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from AutoAugment.diagnostics.advisor import generate_augmentation_advice
from AutoAugment.policies.policy import Policy
from AutoAugment.policies.search_space import OperationSpace, SearchSpace


@dataclass
class PolicyAdjustment:
    diagnosis: dict[str, Any]
    adjust_reason: str
    before_search_space: SearchSpace
    after_search_space: SearchSpace
    after_policy: Policy
    policy_diff: dict[str, Any]


class DiagnosticPolicyUpdater:
    """Update the next search space from trial diagnostics and trial outcome."""

    def __init__(
        self,
        search_space: SearchSpace,
        *,
        learning_rate: float = 0.65,
        reinforce_rate: float = 0.18,
        min_weight: float = 0.05,
        max_weight: float = 4.0,
    ) -> None:
        self.current_search_space = clone_search_space(search_space)
        self.learning_rate = float(np.clip(learning_rate, 0.0, 1.0))
        self.reinforce_rate = float(np.clip(reinforce_rate, 0.0, 1.0))
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def sample_policy(self, rng: np.random.Generator, *, name: str) -> Policy:
        return self.current_search_space.sample_policy(rng, name=name)

    def update(
        self,
        *,
        evaluated_policy: Policy,
        metrics: dict[str, Any],
        diagnosis: dict[str, Any],
        accepted: bool,
        rng: np.random.Generator,
        next_policy_name: str,
    ) -> PolicyAdjustment:
        before_space = clone_search_space(self.current_search_space)
        after_space = update_search_space_from_diagnosis(
            before_space,
            diagnosis,
            evaluated_policy=evaluated_policy,
            accepted=accepted,
            learning_rate=self.learning_rate,
            reinforce_rate=self.reinforce_rate,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
        )
        self.current_search_space = after_space
        after_policy = self.current_search_space.sample_policy(rng, name=next_policy_name)
        diff = diff_policies_and_space(
            before_policy=evaluated_policy,
            after_policy=after_policy,
            before_space=before_space,
            after_space=after_space,
            diagnosis=diagnosis,
            accepted=accepted,
        )
        return PolicyAdjustment(
            diagnosis=diagnosis,
            adjust_reason=adjust_reason_from_diagnosis(diagnosis, accepted=accepted, metrics=metrics),
            before_search_space=before_space,
            after_search_space=after_space,
            after_policy=after_policy,
            policy_diff=diff,
        )


def diagnose_trial_result(
    metrics: dict[str, Any],
    *,
    context: dict[str, Any] | None = None,
    previous_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    metrics = metrics or {}
    raw_diagnosis = metrics.get("diagnosis")
    if isinstance(raw_diagnosis, dict):
        return normalize_diagnosis(raw_diagnosis, metrics=metrics, previous_metrics=previous_metrics)

    summary = _first_dict(
        metrics.get("val_analysis"),
        metrics.get("error_summary"),
        metrics.get("diagnosis_summary"),
        context.get("trial_error_summary"),
    )
    if summary is None:
        summary = {}
    advice = generate_augmentation_advice(summary, metrics=metrics)
    advice["previous_metrics"] = _metric_delta(previous_metrics, metrics)
    advice["metrics_used"] = _compact_metrics(metrics)
    advice.setdefault("source", "trial_metrics_fallback" if not summary else "trial_error_summary")
    return normalize_diagnosis(advice, metrics=metrics, previous_metrics=previous_metrics)


def normalize_diagnosis(
    diagnosis: dict[str, Any],
    *,
    metrics: dict[str, Any] | None = None,
    previous_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = dict(diagnosis)
    issues = list(normalized.get("main_issues") or normalized.get("issues") or [])
    if not issues:
        fallback = generate_augmentation_advice({}, metrics=metrics or {})
        issues = list(fallback.get("main_issues") or fallback.get("issues") or [])
        normalized["advisor_search_space"] = fallback.get("advisor_search_space")
        normalized["fallback_used"] = True
    normalized["issues"] = issues
    normalized["main_issues"] = issues
    normalized.setdefault("fallback_used", not bool(issues))
    normalized.setdefault("advisor_search_space", generate_augmentation_advice({}, metrics=metrics or {})["advisor_search_space"])
    normalized.setdefault("metrics_used", _compact_metrics(metrics or {}))
    normalized.setdefault("previous_metrics", _metric_delta(previous_metrics, metrics or {}))
    normalized.setdefault("source", "normalized_diagnosis")
    return normalized


def update_search_space_from_diagnosis(
    search_space: SearchSpace,
    diagnosis: dict[str, Any],
    *,
    evaluated_policy: Policy,
    accepted: bool,
    learning_rate: float,
    reinforce_rate: float,
    min_weight: float,
    max_weight: float,
) -> SearchSpace:
    config = diagnosis.get("advisor_search_space") or {}
    target_weights = {
        str(key).strip().lower(): float(value)
        for key, value in (config.get("operation_weights") or {}).items()
        if _is_number(value)
    }
    target_strength = config.get("strength_ranges") or {}
    target_prob = config.get("prob_ranges") or {}
    current_weights = dict(search_space.operation_weights or {})
    operation_names = [operation.name for operation in search_space.operations]
    new_weights: dict[str, float] = {}
    for name in operation_names:
        current = float(current_weights.get(name, 1.0))
        target = float(target_weights.get(name, current))
        blended = _blend(current, target, learning_rate)
        if any(operation.name == name for operation in evaluated_policy.operations):
            multiplier = 1.0 + reinforce_rate if accepted else max(0.1, 1.0 - reinforce_rate)
            blended *= multiplier
        new_weights[name] = float(np.clip(blended, min_weight, max_weight))

    operations: list[OperationSpace] = []
    for operation in search_space.operations:
        operations.append(
            OperationSpace(
                name=operation.name,
                prob_range=_blend_range(operation.prob_range, target_prob.get(operation.name), learning_rate),
                strength_range=_blend_range(operation.strength_range, target_strength.get(operation.name), learning_rate),
                params=dict(operation.params),
            )
        )
    forbidden = list(search_space.forbidden_combinations)
    for combination in config.get("forbidden_combinations") or []:
        normalized = tuple(str(item).strip().lower() for item in combination if str(item).strip())
        if normalized and normalized not in forbidden:
            forbidden.append(normalized)
    return SearchSpace(
        operations=operations,
        operation_count_range=search_space.operation_count_range,
        allow_repeated_operations=search_space.allow_repeated_operations,
        name_prefix=search_space.name_prefix,
        operation_weights=new_weights,
        forbidden_combinations=forbidden,
    )


def adjust_reason_from_diagnosis(diagnosis: dict[str, Any], *, accepted: bool, metrics: dict[str, Any]) -> str:
    issue_names = [str(item.get("issue")) for item in diagnosis.get("main_issues", []) if item.get("issue")]
    if not issue_names:
        issue_names = ["no_main_issue"]
    score = metrics.get("final_score", metrics.get("yolo_metric_score", metrics.get("score")))
    outcome = "accepted" if accepted else "rejected"
    return f"{outcome}; issues={','.join(issue_names)}; score={score}"


def diff_policies_and_space(
    *,
    before_policy: Policy,
    after_policy: Policy | None,
    before_space: SearchSpace | None = None,
    after_space: SearchSpace | None = None,
    diagnosis: dict[str, Any] | None = None,
    accepted: bool | None = None,
) -> dict[str, Any]:
    after_policy = after_policy or Policy(name="", operations=[])
    before_names = [operation.name for operation in before_policy.operations]
    after_names = [operation.name for operation in after_policy.operations]
    before_counts = _counts(before_names)
    after_counts = _counts(after_names)
    names = sorted(set(before_counts) | set(after_counts))
    op_delta = {name: after_counts.get(name, 0) - before_counts.get(name, 0) for name in names if after_counts.get(name, 0) != before_counts.get(name, 0)}
    weight_delta: dict[str, dict[str, Any]] = {}
    unchanged_weights: list[str] = []
    prob_range_changes: dict[str, dict[str, Any]] = {}
    strength_range_changes: dict[str, dict[str, Any]] = {}
    unchanged_prob_ranges: list[str] = []
    unchanged_strength_ranges: list[str] = []
    if before_space is not None and after_space is not None:
        before_weights = {operation.name: float(before_space.operation_weights.get(operation.name, 1.0)) for operation in before_space.operations}
        after_weights = {operation.name: float(after_space.operation_weights.get(operation.name, 1.0)) for operation in after_space.operations}
        for name in sorted(set(before_weights) | set(after_weights)):
            before_value = before_weights.get(name, 1.0)
            after_value = after_weights.get(name, 1.0)
            if abs(after_value - before_value) > 1e-9:
                weight_delta[name] = {
                    "before": before_value,
                    "after": after_value,
                    "delta": after_value - before_value,
                    "reason": _change_reason(
                        name,
                        "operation_weight",
                        diagnosis=diagnosis,
                        before_policy=before_policy,
                        accepted=accepted,
                    ),
                }
            else:
                unchanged_weights.append(name)
        before_operations = {operation.name: operation for operation in before_space.operations}
        after_operations = {operation.name: operation for operation in after_space.operations}
        for name in sorted(set(before_operations) | set(after_operations)):
            before_operation = before_operations.get(name)
            after_operation = after_operations.get(name)
            if before_operation is None or after_operation is None:
                continue
            before_prob = tuple(float(value) for value in before_operation.prob_range)
            after_prob = tuple(float(value) for value in after_operation.prob_range)
            if _range_changed(before_prob, after_prob):
                prob_range_changes[name] = {
                    "before": list(before_prob),
                    "after": list(after_prob),
                    "delta": [after_prob[0] - before_prob[0], after_prob[1] - before_prob[1]],
                    "reason": _change_reason(name, "prob_range", diagnosis=diagnosis),
                }
            else:
                unchanged_prob_ranges.append(name)
            before_strength = tuple(float(value) for value in before_operation.strength_range)
            after_strength = tuple(float(value) for value in after_operation.strength_range)
            if _range_changed(before_strength, after_strength):
                strength_range_changes[name] = {
                    "before": list(before_strength),
                    "after": list(after_strength),
                    "delta": [after_strength[0] - before_strength[0], after_strength[1] - before_strength[1]],
                    "reason": _change_reason(name, "strength_range", diagnosis=diagnosis),
                }
            else:
                unchanged_strength_ranges.append(name)
    return {
        "diff_schema_version": 2,
        "before_policy_name": before_policy.name,
        "after_policy_name": after_policy.name,
        "ops_added": sorted([name for name, delta in op_delta.items() if delta > 0]),
        "ops_removed": sorted([name for name, delta in op_delta.items() if delta < 0]),
        "operation_count_delta": op_delta,
        "operation_weight_changes": weight_delta,
        "operation_weight_unchanged": unchanged_weights,
        "prob_range_changes": prob_range_changes,
        "prob_range_unchanged": unchanged_prob_ranges,
        "strength_range_changes": strength_range_changes,
        "strength_range_unchanged": unchanged_strength_ranges,
        "search_space_weight_delta": weight_delta,
    }


def clone_search_space(search_space: SearchSpace) -> SearchSpace:
    return SearchSpace(
        operations=[
            OperationSpace(
                name=operation.name,
                prob_range=tuple(operation.prob_range),
                strength_range=tuple(operation.strength_range),
                params=dict(operation.params),
            )
            for operation in search_space.operations
        ],
        operation_count_range=tuple(search_space.operation_count_range),
        allow_repeated_operations=search_space.allow_repeated_operations,
        name_prefix=search_space.name_prefix,
        operation_weights=dict(search_space.operation_weights),
        forbidden_combinations=list(search_space.forbidden_combinations),
    )


def _blend(current: float, target: float, rate: float) -> float:
    return current * (1.0 - rate) + target * rate


def _blend_range(current: tuple[float, float], target: Any, rate: float) -> tuple[float, float]:
    if target is None:
        return tuple(current)
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        return tuple(current)
    low = float(np.clip(_blend(float(current[0]), float(target[0]), rate), 0.0, 1.0))
    high = float(np.clip(_blend(float(current[1]), float(target[1]), rate), low, 1.0))
    return (low, high)


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _first_dict(*items: Any) -> dict[str, Any] | None:
    for item in items:
        if isinstance(item, dict):
            return item
    return None


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "yolo_map50",
        "yolo_map50_95",
        "yolo_metric_name",
        "yolo_metric_score",
        "final_score",
        "proxy_score",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def _metric_delta(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {}
    out: dict[str, Any] = {}
    for key in ["yolo_map50", "yolo_map50_95", "yolo_metric_score", "final_score", "proxy_score", "precision", "recall"]:
        before = previous.get(key)
        after = current.get(key)
        if _is_number(before) and _is_number(after):
            out[key] = {"before": float(before), "after": float(after), "delta": float(after) - float(before)}
    return out


def _counts(names: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in names:
        out[name] = out.get(name, 0) + 1
    return out


def _range_changed(before: tuple[float, float], after: tuple[float, float]) -> bool:
    return abs(after[0] - before[0]) > 1e-9 or abs(after[1] - before[1]) > 1e-9


_ISSUE_OPERATION_REASONS: dict[str, dict[str, set[str]]] = {
    "operation_weight": {
        "small_object_fn_high": {"sharpen", "clahe", "scale", "translate", "motion_blur", "gaussian_blur", "gaussian_noise"},
        "edge_object_fn_high": {"translate", "affine", "scale"},
        "low_contrast_fn_high": {"contrast", "gamma", "clahe", "sharpen"},
        "exposure_fn_high": {"brightness", "gamma"},
        "false_positive_high": {"gaussian_noise", "cutout"},
        "false_positive_precision_gap": {"gaussian_noise", "cutout", "motion_blur", "contrast", "gamma", "clahe"},
        "false_negative_recall_gap": {"brightness", "contrast", "gamma", "clahe", "sharpen", "scale"},
        "metric_map50_gap": {"brightness", "contrast", "gamma", "clahe", "sharpen", "scale"},
        "weak_localization_gap": {"translate", "scale", "affine", "rotate", "motion_blur", "gaussian_blur"},
        "position_fn_gap": {"translate", "scale", "affine", "rotate", "motion_blur", "gaussian_blur"},
        "metric_map50_95_gap": {"translate", "scale", "affine", "rotate", "motion_blur", "gaussian_blur"},
        "class_precision_imbalance": {"contrast", "gamma", "clahe"},
        "false_negatives_low_contrast_hint": {"contrast", "clahe", "gamma", "sharpen"},
        "false_positives_low_contrast_hint": {"contrast", "clahe", "gamma", "sharpen"},
        "false_negatives_exposure_hint": {"brightness", "gamma", "clahe"},
        "false_positives_exposure_hint": {"brightness", "gamma", "clahe"},
        "metric_precision_gap": {"gaussian_noise", "cutout", "motion_blur", "contrast", "gamma", "clahe"},
        "metric_recall_gap": {"brightness", "contrast", "gamma", "clahe", "sharpen", "scale"},
        "stable_validation_keep_light_policy": {
            "brightness",
            "contrast",
            "gamma",
            "clahe",
            "translate",
            "scale",
            "rotate",
            "gaussian_blur",
            "motion_blur",
            "gaussian_noise",
            "cutout",
        },
    },
    "prob_range": {
        "edge_object_fn_high": {"translate"},
        "false_positive_precision_gap": {"gaussian_noise", "cutout"},
        "weak_localization_gap": {"translate", "scale", "affine"},
        "position_fn_gap": {"translate", "scale", "affine"},
        "metric_map50_95_gap": {"translate", "scale", "affine"},
        "stable_validation_keep_light_policy": {"brightness", "contrast", "gamma", "translate", "scale"},
    },
    "strength_range": {
        "small_object_fn_high": {"motion_blur", "gaussian_blur", "gaussian_noise"},
        "low_contrast_fn_high": {"contrast", "gamma", "clahe"},
        "exposure_fn_high": {"brightness"},
        "false_positive_precision_gap": {"gaussian_noise", "motion_blur"},
        "weak_localization_gap": {"translate", "scale", "affine", "rotate"},
        "position_fn_gap": {"translate", "scale", "affine", "rotate"},
        "metric_map50_95_gap": {"translate", "scale", "affine", "rotate"},
        "false_negatives_low_contrast_hint": {"contrast", "clahe", "gamma"},
        "false_positives_low_contrast_hint": {"contrast", "clahe", "gamma"},
        "false_negatives_exposure_hint": {"brightness", "gamma"},
        "false_positives_exposure_hint": {"brightness", "gamma"},
        "stable_validation_keep_light_policy": {"brightness", "contrast", "gamma", "translate", "scale"},
    },
}


def _change_reason(
    operation_name: str,
    change_kind: str,
    *,
    diagnosis: dict[str, Any] | None = None,
    before_policy: Policy | None = None,
    accepted: bool | None = None,
) -> str:
    issue_names = [
        str(item.get("issue"))
        for item in (diagnosis or {}).get("main_issues", [])
        if isinstance(item, dict) and item.get("issue")
    ]
    matched = [
        issue
        for issue in issue_names
        if operation_name in _ISSUE_OPERATION_REASONS.get(change_kind, {}).get(issue, set())
    ]
    reasons = matched or (issue_names if issue_names else ["diagnostic_search_space_update"])
    if change_kind == "operation_weight" and before_policy is not None and accepted is not None:
        if any(operation.name == operation_name for operation in before_policy.operations):
            reasons.append("accepted_policy_reinforcement" if accepted else "rejected_policy_penalty")
    return ",".join(dict.fromkeys(reasons))
