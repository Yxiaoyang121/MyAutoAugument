from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


CATF_V2_OPS = ("clahe", "gamma", "brightness", "contrast", "sharpen_mild", "local_contrast", "cutout_safe")
PHOTOMETRIC_OPS = ("clahe", "gamma", "brightness", "contrast")
TEXTURE_OPS = ("sharpen_mild", "local_contrast")
OCCLUSION_OPS = ("cutout_safe",)
TRUST_REGION_PROB = 0.015
TRUST_REGION_STRENGTH = 0.025
DEFAULT_STRENGTHS = {
    "clahe": 0.18,
    "gamma": 0.18,
    "brightness": 0.12,
    "contrast": 0.12,
    "sharpen_mild": 0.22,
    "local_contrast": 0.20,
    "cutout_safe": 0.05,
}
MAX_PROBS = {
    "clahe": 0.15,
    "gamma": 0.15,
    "brightness": 0.10,
    "contrast": 0.10,
    "sharpen_mild": 0.20,
    "local_contrast": 0.18,
    "cutout_safe": 0.08,
}
MAX_GLOBAL_PROB = {
    "clahe": 0.08,
    "gamma": 0.08,
    "brightness": 0.06,
    "contrast": 0.06,
    "sharpen_mild": 0.12,
    "local_contrast": 0.12,
    "cutout_safe": 0.04,
}


def initial_policy_matrix(class_names: dict[int, str]) -> dict[str, Any]:
    """Return a zero-probability class-aware policy matrix."""

    classes: dict[str, Any] = {}
    for class_id in sorted(class_names):
        classes[str(class_id)] = {
            "class_id": int(class_id),
            "class_name": class_names[class_id],
            "status": "inactive",
            "state": "accepted",
            "version": 0,
            "dominant_issue": None,
            "secondary_issues": [],
            "ops": {
                op: {
                    "prob": 0.0,
                    "strength": 0.0,
                    "max_prob": MAX_PROBS[op],
                    "max_strength": 0.45 if op not in {"brightness", "contrast", "cutout_safe"} else 0.25,
                }
                for op in CATF_V2_OPS
            },
            "guards": {
                "precision_guard": False,
                "map95_guard": False,
                "high_fp_guarded": False,
            },
            "threshold_calibration_candidate": False,
            "oversampling_candidate": False,
            "copy_paste_candidate": False,
            "copy_paste_status": "pending_object_bank_design",
            "last_safe_policy": None,
            "failed_update_count": 0,
            "pending_since_epoch": None,
            "frozen_reason": None,
        }
    return {
        "policy_id": "catf_v2_class_aware",
        "version": 0,
        "catf_version": "v2",
        "copy_paste_status": "pending_object_bank_design",
        "classes": classes,
        "global_guard": {"active": False, "reason": None},
    }


@dataclass
class ClassAwarePolicyMatrix:
    matrix: dict[str, Any]
    top_k: int = 3
    top_m: int = 2
    freeze_epoch: int = 40
    last_safe_matrix: dict[str, Any] = field(init=False)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.matrix = deepcopy(self.matrix)
        self.last_safe_matrix = deepcopy(self.matrix)

    def update(
        self,
        *,
        epoch: int,
        per_class_diagnosis: dict[str, Any],
        issue_attribution: dict[str, Any],
        metrics: dict[str, Any],
        reference_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        before = deepcopy(self.matrix)
        if epoch >= self.freeze_epoch:
            self._freeze_all("epoch_ge_40")
            record = self._record(epoch, before, deepcopy(self.matrix), [], "freeze", ["epoch_ge_40"], metrics, reference_metrics)
            self.history.append(record)
            return deepcopy(self.matrix)

        global_guards = global_guard_names(metrics, reference_metrics)
        if global_guards:
            self._apply_global_shrink()

        candidates = self._rank_candidates(per_class_diagnosis, issue_attribution)
        adjustments: list[dict[str, Any]] = []
        class_actions: list[dict[str, Any]] = []
        for class_id in candidates[: max(0, int(self.top_k))]:
            row = (per_class_diagnosis.get("classes") or {}).get(str(class_id), {})
            attr = (issue_attribution.get("classes") or {}).get(str(class_id), {})
            action = self._update_class(class_id, row, attr, epoch=epoch, global_guards=global_guards)
            adjustments.extend(action["adjustments"])
            class_actions.append(action)

        self._enforce_global_average_budget(adjustments)
        self.matrix["version"] = int(self.matrix.get("version", 0) or 0) + 1
        action_name = "shrink" if global_guards else ("accept" if adjustments else "observe")
        if any(item["action"] == "rollback" for item in class_actions):
            action_name = "rollback"
        record = self._record(epoch, before, deepcopy(self.matrix), adjustments, action_name, global_guards, metrics, reference_metrics)
        record["class_actions"] = class_actions
        record["active_classes"] = active_class_ids(self.matrix)
        record["frozen_classes"] = frozen_class_ids(self.matrix)
        self.history.append(record)
        self.last_safe_matrix = deepcopy(self.matrix)
        return deepcopy(self.matrix)

    def _rank_candidates(self, per_class_diagnosis: dict[str, Any], issue_attribution: dict[str, Any]) -> list[int]:
        rows = per_class_diagnosis.get("classes") or {}
        attrs = issue_attribution.get("classes") or {}
        scored: list[tuple[float, int]] = []
        for raw_id, row in rows.items():
            class_id = int(raw_id)
            matrix_row = self.matrix["classes"].setdefault(str(class_id), _blank_class(class_id, row.get("class_name", str(class_id))))
            if matrix_row.get("status") == "frozen" or row.get("stable_class"):
                matrix_row["status"] = "frozen"
                matrix_row["state"] = "frozen"
                matrix_row["frozen_reason"] = matrix_row.get("frozen_reason") or "stable_class"
                continue
            attr = attrs.get(str(class_id), {})
            scores = attr.get("issue_scores", {}) or {}
            if row.get("low_support"):
                scored.append((0.85, class_id))
                continue
            dominant_score = float(scores.get(attr.get("dominant_issue", ""), 0.0) or 0.0)
            evidence = min(1.0, float(row.get("evidence_count", 0) or 0) / 20.0)
            scored.append((dominant_score + 0.15 * evidence, class_id))
        scored.sort(reverse=True)
        return [class_id for _, class_id in scored]

    def _update_class(
        self,
        class_id: int,
        row: dict[str, Any],
        attr: dict[str, Any],
        *,
        epoch: int,
        global_guards: list[str],
    ) -> dict[str, Any]:
        class_key = str(class_id)
        policy = self.matrix["classes"].setdefault(class_key, _blank_class(class_id, row.get("class_name", str(class_id))))
        before = deepcopy(policy)
        adjustments: list[dict[str, Any]] = []
        pending_action = self._resolve_pending_policy(policy, row)
        if pending_action == "rollback":
            return {"class_id": class_id, "action": "rollback", "adjustments": [], "before": before, "after": deepcopy(policy)}
        if pending_action == "accept" and row.get("stable_class"):
            return {"class_id": class_id, "action": "accept", "adjustments": [], "before": before, "after": deepcopy(policy)}
        dominant = str(attr.get("dominant_issue") or "stable_class")
        policy["dominant_issue"] = dominant
        policy["secondary_issues"] = list(attr.get("secondary_issues") or [])
        policy["threshold_calibration_candidate"] = bool(attr.get("threshold_calibration_candidate", False))
        policy["oversampling_candidate"] = bool(attr.get("oversampling_candidate", False))
        policy["copy_paste_candidate"] = bool(attr.get("copy_paste_candidate", False))

        if row.get("stable_class"):
            policy["status"] = "frozen"
            policy["state"] = "frozen"
            policy["frozen_reason"] = "stable_class"
            return {"class_id": class_id, "action": "freeze", "adjustments": [], "before": before, "after": deepcopy(policy)}

        if row.get("high_fp") or dominant == "high_fp":
            policy["guards"]["precision_guard"] = True
            policy["guards"]["high_fp_guarded"] = True
            adjustments.extend(self._adjust_ops(policy, PHOTOMETRIC_OPS, -TRUST_REGION_PROB, -TRUST_REGION_STRENGTH, "high_fp_guard"))
            adjustments.extend(self._adjust_ops(policy, OCCLUSION_OPS, -TRUST_REGION_PROB, -TRUST_REGION_STRENGTH, "high_fp_cutout_guard"))
            policy["status"] = "active"
            policy["state"] = "pending"
            policy["pending_since_epoch"] = epoch
            return {"class_id": class_id, "action": "guard", "adjustments": adjustments, "before": before, "after": deepcopy(policy)}

        if row.get("low_support"):
            policy["status"] = "observe"
            policy["state"] = "accepted"
            policy["oversampling_candidate"] = True
            policy["copy_paste_candidate"] = True
            policy["copy_paste_status"] = "pending_object_bank_design"
            return {"class_id": class_id, "action": "observe_low_support", "adjustments": [], "before": before, "after": deepcopy(policy)}

        op_plan = op_plan_for_issue(dominant, row)
        if global_guards:
            op_plan = [(op, min(delta, 0.0), min(strength_delta, 0.0), reason + "_global_guard") for op, delta, strength_delta, reason in op_plan]
        for op_name, prob_delta, strength_delta, reason in op_plan[: max(0, int(self.top_m))]:
            adjustments.extend(self._adjust_ops(policy, (op_name,), prob_delta, strength_delta, reason))

        if adjustments:
            policy["status"] = "active"
            policy["state"] = "pending"
            policy["version"] = int(policy.get("version", 0) or 0) + 1
            policy["pending_since_epoch"] = epoch
            policy["last_safe_policy"] = before
            policy["last_safe_metrics"] = class_metric_snapshot(row)
        return {"class_id": class_id, "action": "propose" if adjustments else "observe", "adjustments": adjustments, "before": before, "after": deepcopy(policy)}

    def _adjust_ops(
        self,
        policy: dict[str, Any],
        op_names: tuple[str, ...],
        prob_delta: float,
        strength_delta: float,
        reason: str,
    ) -> list[dict[str, Any]]:
        changes = []
        for op_name in op_names:
            op = policy["ops"].setdefault(op_name, _blank_op(op_name))
            before_prob = float(op.get("prob", 0.0) or 0.0)
            before_strength = float(op.get("strength", 0.0) or 0.0)
            clipped_prob_delta = max(-TRUST_REGION_PROB, min(TRUST_REGION_PROB, float(prob_delta)))
            clipped_strength_delta = max(-TRUST_REGION_STRENGTH, min(TRUST_REGION_STRENGTH, float(strength_delta)))
            target_prob = max(0.0, min(float(op.get("max_prob", MAX_PROBS[op_name])), before_prob + clipped_prob_delta))
            target_strength = max(0.0, min(float(op.get("max_strength", 0.45)), before_strength + clipped_strength_delta))
            if target_prob > 0 and target_strength == 0.0:
                target_strength = min(float(op.get("max_strength", 0.45)), DEFAULT_STRENGTHS[op_name])
            if target_prob != before_prob:
                op["prob"] = round(target_prob, 6)
                changes.append(_adjustment(policy["class_id"], op_name, "prob", before_prob, target_prob, reason, clipped_prob_delta != prob_delta))
            if target_strength != before_strength:
                op["strength"] = round(target_strength, 6)
                changes.append(
                    _adjustment(policy["class_id"], op_name, "strength", before_strength, target_strength, reason, clipped_strength_delta != strength_delta)
                )
        return changes

    def _apply_global_shrink(self) -> None:
        self.matrix["global_guard"] = {"active": True, "reason": "global_constraint_warning"}
        for policy in self.matrix.get("classes", {}).values():
            if policy.get("status") not in {"active", "pending"}:
                continue
            for op_name in PHOTOMETRIC_OPS:
                op = policy.get("ops", {}).get(op_name, {})
                op["prob"] = round(float(op.get("prob", 0.0) or 0.0) * 0.8, 6)
            op = policy.get("ops", {}).get("cutout_safe", {})
            op["prob"] = round(float(op.get("prob", 0.0) or 0.0) * 0.5, 6)

    def _enforce_global_average_budget(self, adjustments: list[dict[str, Any]]) -> None:
        classes = list(self.matrix.get("classes", {}).values())
        if not classes:
            return
        for op_name, max_avg in MAX_GLOBAL_PROB.items():
            values = [float(item.get("ops", {}).get(op_name, {}).get("prob", 0.0) or 0.0) for item in classes]
            avg = sum(values) / max(1, len(values))
            if avg <= max_avg or avg <= 0:
                continue
            factor = max_avg / avg
            for item in classes:
                op = item.get("ops", {}).get(op_name, {})
                before = float(op.get("prob", 0.0) or 0.0)
                after = round(before * factor, 6)
                op["prob"] = after
                if before != after:
                    adjustments.append(_adjustment(item["class_id"], op_name, "prob", before, after, "max_global_prob", False))

    def _resolve_pending_policy(self, policy: dict[str, Any], row: dict[str, Any]) -> str | None:
        if policy.get("state") != "pending":
            return None
        last_safe = policy.get("last_safe_policy")
        last_metrics = policy.get("last_safe_metrics") or {}
        current_precision = float(row.get("Precision", 0.0) or 0.0)
        current_ap95 = float(row.get("AP50_95", 0.0) or 0.0)
        safe_precision = float(last_metrics.get("Precision", current_precision) or current_precision)
        safe_ap95 = float(last_metrics.get("AP50_95", current_ap95) or current_ap95)
        precision_drop = current_precision < safe_precision - 0.02
        ap95_drop = current_ap95 < safe_ap95 - 0.02
        if row.get("high_fp") or precision_drop or ap95_drop:
            if isinstance(last_safe, dict):
                keep_version = int(policy.get("version", 0) or 0)
                policy.clear()
                policy.update(deepcopy(last_safe))
                policy["version"] = keep_version
            policy["state"] = "rollback"
            policy["guards"]["precision_guard"] = bool(row.get("high_fp") or precision_drop)
            policy["guards"]["map95_guard"] = bool(ap95_drop)
            policy["guards"]["high_fp_guarded"] = bool(row.get("high_fp") or precision_drop)
            policy["failed_update_count"] = int(policy.get("failed_update_count", 0) or 0) + 1
            if int(policy.get("failed_update_count", 0) or 0) >= 2:
                policy["state"] = "frozen"
                policy["status"] = "frozen"
                policy["frozen_reason"] = "repeated_class_guard"
            return "rollback"
        policy["state"] = "accepted"
        policy["status"] = "active"
        policy["failed_update_count"] = 0
        return "accept"

    def _freeze_all(self, reason: str) -> None:
        for policy in self.matrix.get("classes", {}).values():
            policy["state"] = "frozen"
            policy["status"] = "frozen"
            policy["frozen_reason"] = reason

    def _record(
        self,
        epoch: int,
        before: dict[str, Any],
        after: dict[str, Any],
        adjustments: list[dict[str, Any]],
        action: str,
        guards: list[str],
        metrics: dict[str, Any],
        reference_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "metrics": deepcopy(metrics),
            "reference_metrics": deepcopy(reference_metrics),
            "delta_metrics": delta_metrics(metrics, reference_metrics),
            "diagnosis_summary": {},
            "old_policy": before,
            "proposed_policy": after,
            "accepted_policy": after,
            "last_safe_policy_id": f"catf_v2_epoch_{epoch:03d}",
            "action": action,
            "guard_triggered": guards,
            "rollback_reason": None,
            "frozen": action == "freeze",
            "group_budget_before": class_group_budget(before),
            "group_budget_after": class_group_budget(after),
            "trust_region_clipping": [item for item in adjustments if item.get("trust_region_clipped")],
            "adjustments": adjustments,
            "copy_paste_status": "pending_object_bank_design",
        }


def op_plan_for_issue(issue: str, row: dict[str, Any]) -> list[tuple[str, float, float, str]]:
    if issue == "low_contrast_fn":
        return [
            ("local_contrast", 0.020, 0.020, "low_contrast_fn"),
            ("gamma", 0.010, 0.015, "low_contrast_fn"),
            ("clahe", 0.010, 0.015, "low_contrast_fn"),
            ("sharpen_mild", 0.010, 0.015, "low_contrast_fn"),
        ]
    if issue in {"texture_boundary_weak", "weak_localization"}:
        return [
            ("sharpen_mild", 0.020, 0.020, issue),
            ("local_contrast", 0.020, 0.020, issue),
            ("gamma", 0.005, 0.010, issue),
        ]
    if issue == "low_recall":
        return [
            ("sharpen_mild", 0.010, 0.015, "low_recall"),
            ("local_contrast", 0.010, 0.015, "low_recall"),
            ("gamma", 0.005, 0.010, "low_recall"),
        ]
    return []


def active_class_ids(matrix: dict[str, Any]) -> list[int]:
    return [
        int(class_id)
        for class_id, item in sorted((matrix.get("classes") or {}).items(), key=lambda pair: int(pair[0]))
        if item.get("status") == "active"
    ]


def frozen_class_ids(matrix: dict[str, Any]) -> list[int]:
    return [
        int(class_id)
        for class_id, item in sorted((matrix.get("classes") or {}).items(), key=lambda pair: int(pair[0]))
        if item.get("status") == "frozen" or item.get("state") == "frozen"
    ]


def class_group_budget(matrix: dict[str, Any]) -> dict[str, Any]:
    totals = {"photometric": 0.0, "texture": 0.0, "occlusion": 0.0}
    for item in (matrix.get("classes") or {}).values():
        ops = item.get("ops", {})
        totals["photometric"] += sum(float(ops.get(op, {}).get("prob", 0.0) or 0.0) for op in PHOTOMETRIC_OPS)
        totals["texture"] += sum(float(ops.get(op, {}).get("prob", 0.0) or 0.0) for op in TEXTURE_OPS)
        totals["occlusion"] += sum(float(ops.get(op, {}).get("prob", 0.0) or 0.0) for op in OCCLUSION_OPS)
    return {
        "photometric": {"value": round(totals["photometric"], 6), "limit": None, "within_budget": True},
        "texture": {"value": round(totals["texture"], 6), "limit": None, "within_budget": True},
        "occlusion": {"value": round(totals["occlusion"], 6), "limit": None, "within_budget": True},
    }


def global_guard_names(metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> list[str]:
    guards = []
    deltas = delta_metrics(metrics, reference_metrics)
    if deltas.get("precision") is not None and deltas["precision"] < -0.01:
        guards.append("global_precision_guard")
    if deltas.get("map50") is not None and deltas["map50"] < -0.01:
        guards.append("global_map50_guard")
    if deltas.get("map50_95") is not None and deltas["map50_95"] < -0.01:
        guards.append("global_map95_guard")
    return guards


def delta_metrics(metrics: dict[str, Any], reference_metrics: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("precision", "recall", "map50", "map50_95"):
        value = metrics.get(key)
        ref = reference_metrics.get(key)
        out[key] = None if value is None or ref is None else float(value) - float(ref)
    return out


def class_metric_snapshot(row: dict[str, Any]) -> dict[str, float]:
    return {
        "Precision": float(row.get("Precision", 0.0) or 0.0),
        "Recall": float(row.get("Recall", 0.0) or 0.0),
        "AP50": float(row.get("AP50", 0.0) or 0.0),
        "AP50_95": float(row.get("AP50_95", 0.0) or 0.0),
    }


def _adjustment(class_id: int, op: str, field: str, before: float, after: float, reason: str, clipped: bool) -> dict[str, Any]:
    return {
        "class_id": int(class_id),
        "op": op,
        "field": field,
        "before": round(float(before), 6),
        "after": round(float(after), 6),
        "reason": reason,
        "trust_region_clipped": bool(clipped),
    }


def _blank_op(op_name: str) -> dict[str, float]:
    return {"prob": 0.0, "strength": 0.0, "max_prob": MAX_PROBS[op_name], "max_strength": 0.45}


def _blank_class(class_id: int, class_name: str) -> dict[str, Any]:
    return initial_policy_matrix({class_id: class_name})["classes"][str(class_id)]
