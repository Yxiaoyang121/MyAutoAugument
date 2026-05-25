from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PHOTOMETRIC_VISIBILITY_OPS = {"clahe", "gamma", "brightness", "contrast", "sharpen", "sharpen_mild", "local_contrast"}
AGGRESSIVE_PHOTOMETRIC_OPS = {"brightness", "contrast", "gamma", "gaussian_noise", "blur_mild"}
LOCALIZATION_OPS = {"mild_scale", "mild_translate", "random_scale_translate"}
CUTOUT_OPS = {"cutout_safe", "random_erasing"}
COPY_PASTE_OPS = {"copy_paste", "online_copy_paste", "class_balanced_copy_paste"}


@dataclass
class PolicyAdjustment:
    op: str
    field: str
    before: float
    after: float
    reason: str


@dataclass
class FeedbackPolicyController:
    policy: dict[str, Any]
    history_dir: Path
    policy_state_path: Path | None = None
    profile: str = "industrial"
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.policy = deepcopy(self.policy)
        self.history_dir = Path(self.history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        if self.policy_state_path is not None:
            self.policy_state_path = Path(self.policy_state_path)

    def update(
        self,
        diagnostics: dict[str, Any],
        *,
        stage_index: int,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        adjustments: list[PolicyAdjustment] = []
        flags = normalize_diagnostics(diagnostics, metrics=metrics)

        if flags["low_contrast_fn_high"]:
            for name in ["clahe", "gamma", "brightness", "contrast", "sharpen_mild", "sharpen"]:
                adjustments.extend(self._adjust(name, prob_delta=0.08, strength_delta=0.04, reason="low_contrast_fn_high"))

        if flags["recall_low"] or flags["fn_high"]:
            for name in ["clahe", "gamma", "brightness", "contrast", "mild_scale", "mosaic4"]:
                adjustments.extend(self._adjust(name, prob_delta=0.05, strength_delta=0.03, reason="recall_low_or_fn_high"))
            adjustments.extend(self._adjust("copy_paste", prob_delta=0.04, strength_delta=0.0, reason="recall_low_copy_paste_pending"))

        if flags["precision_low"] or flags["fp_high"]:
            for name in ["brightness", "contrast", "gamma", "gaussian_noise", "blur_mild", "copy_paste"]:
                adjustments.extend(self._adjust(name, prob_delta=-0.06, strength_delta=-0.03, reason="precision_low_or_fp_high"))
            for name in ["cutout_safe", "random_erasing"]:
                adjustments.extend(self._adjust(name, prob_delta=0.06, strength_delta=0.02, reason="precision_low_hard_negative_like"))

        if flags["map50_high_map95_low"]:
            for name in ["sharpen_mild", "sharpen", "local_contrast"]:
                adjustments.extend(self._adjust(name, prob_delta=0.06, strength_delta=0.03, reason="map50_high_map95_low"))
            for name in ["mosaic4", "random_scale_translate", "cutout_safe", "random_erasing"]:
                adjustments.extend(self._adjust(name, prob_delta=-0.04, strength_delta=-0.03, reason="map50_high_map95_low_reduce_destructive_ops"))

        if flags["localization_weak"]:
            for name in ["mild_scale", "mild_translate", "random_scale_translate"]:
                adjustments.extend(self._adjust(name, prob_delta=0.06, strength_delta=0.03, reason="localization_weak"))
            for name in ["mosaic4"]:
                adjustments.extend(self._adjust(name, prob_delta=-0.03, strength_delta=-0.02, reason="localization_weak_reduce_strong_mosaic"))

        if flags["class_imbalance"] or flags["low_support"]:
            adjustments.extend(self._adjust("class_balanced_copy_paste", prob_delta=0.06, strength_delta=0.0, reason="class_imbalance_copy_paste_pending"))

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage_index": int(stage_index),
            "profile": self.profile,
            "diagnostics": diagnostics,
            "normalized_flags": flags,
            "metrics": metrics or {},
            "adjustments": [adjustment.__dict__ for adjustment in adjustments],
            "copy_paste_status": "pending_object_bank_design",
            "policy": deepcopy(self.policy),
        }
        self.history.append(record)
        self._write_outputs()
        return deepcopy(self.policy)

    def _adjust(self, name: str, *, prob_delta: float, strength_delta: float, reason: str) -> list[PolicyAdjustment]:
        op = self._find_or_create_op(name)
        changes: list[PolicyAdjustment] = []
        before_prob = _op_prob(op)
        min_prob = float(op.get("min_prob", 0.0))
        max_prob = float(op.get("max_prob", 1.0))
        after_prob = _clip(before_prob + prob_delta, min_prob, max_prob)
        if after_prob != before_prob:
            op["prob"] = after_prob
            changes.append(PolicyAdjustment(name, "prob", before_prob, after_prob, reason))

        before_strength = _op_strength(op)
        min_strength = float(op.get("min_strength", 0.0))
        max_strength = float(op.get("max_strength", 1.0))
        after_strength = _clip(before_strength + strength_delta, min_strength, max_strength)
        if after_strength != before_strength:
            op["strength"] = after_strength
            changes.append(PolicyAdjustment(name, "strength", before_strength, after_strength, reason))
        return changes

    def _find_or_create_op(self, name: str) -> dict[str, Any]:
        operations = self.policy.setdefault("operations", [])
        for op in operations:
            if str(op.get("name", "")).strip().lower() == name:
                return op
        op = {
            "name": name,
            "prob": 0.0,
            "base_prob": 0.0,
            "min_prob": 0.0,
            "max_prob": 0.4 if name in COPY_PASTE_OPS else 1.0,
            "strength": 0.0,
            "min_strength": 0.0,
            "max_strength": 1.0,
            "params": {},
            "update_rule": "created_by_feedback_controller",
        }
        if name in COPY_PASTE_OPS:
            op["status"] = "pending_object_bank_design"
        operations.append(op)
        return op

    def _write_outputs(self) -> None:
        write_json(self.history_dir / "policy_history.json", {"history": self.history, "latest_policy": self.policy})
        write_markdown(self.history_dir / "policy_history.md", render_history_markdown(self.history))
        write_history_csv(self.history_dir / "policy_history.csv", self.history)
        if self.policy_state_path is not None:
            write_json(self.policy_state_path, self.policy)


def normalize_diagnostics(diagnostics: dict[str, Any], *, metrics: dict[str, Any] | None = None) -> dict[str, bool]:
    metrics = metrics or {}
    precision = _optional_float(metrics.get("precision"))
    recall = _optional_float(metrics.get("recall"))
    map50 = _optional_float(metrics.get("map50"))
    map50_95 = _optional_float(metrics.get("map50_95"))
    flags = {
        "low_contrast_fn_high": bool(diagnostics.get("low_contrast_fn_high") or diagnostics.get("low_contrast_missed_defect")),
        "recall_low": bool(diagnostics.get("recall_low")),
        "fn_high": bool(diagnostics.get("fn_high")),
        "precision_low": bool(diagnostics.get("precision_low")),
        "fp_high": bool(diagnostics.get("fp_high") or diagnostics.get("false_positive_high")),
        "map50_high_map95_low": bool(diagnostics.get("map50_high_map95_low")),
        "localization_weak": bool(diagnostics.get("localization_weak") or diagnostics.get("localization_bias")),
        "class_imbalance": bool(diagnostics.get("class_imbalance")),
        "low_support": bool(diagnostics.get("low_support")),
    }
    if recall is not None and recall < float(diagnostics.get("recall_threshold", 0.70)):
        flags["recall_low"] = True
    if precision is not None and precision < float(diagnostics.get("precision_threshold", 0.72)):
        flags["precision_low"] = True
    if map50 is not None and map50_95 is not None and map50 >= 0.60 and (map50 - map50_95) >= 0.18:
        flags["map50_high_map95_low"] = True
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


def render_history_markdown(history: list[dict[str, Any]]) -> str:
    lines = [
        "# Feedback Policy History",
        "",
        "| stage | profile | adjustments | copy_paste_status |",
        "|---:|---|---:|---|",
    ]
    for record in history:
        lines.append(
            f"| {record.get('stage_index')} | {record.get('profile')} | "
            f"{len(record.get('adjustments', []))} | {record.get('copy_paste_status')} |"
        )
    lines.extend(["", "## Adjustment Details", ""])
    for record in history:
        lines.append(f"### Stage {record.get('stage_index')}")
        if not record.get("adjustments"):
            lines.append("- No policy changes.")
            continue
        for adjustment in record["adjustments"]:
            lines.append(
                f"- `{adjustment['op']}` {adjustment['field']}: "
                f"{adjustment['before']:.4f} -> {adjustment['after']:.4f} ({adjustment['reason']})"
            )
    return "\n".join(lines) + "\n"


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage_index", "op", "field", "before", "after", "reason"])
        writer.writeheader()
        for record in history:
            for adjustment in record.get("adjustments", []):
                writer.writerow(
                    {
                        "stage_index": record.get("stage_index"),
                        "op": adjustment.get("op"),
                        "field": adjustment.get("field"),
                        "before": adjustment.get("before"),
                        "after": adjustment.get("after"),
                        "reason": adjustment.get("reason"),
                    }
                )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _op_prob(op: dict[str, Any]) -> float:
    for key in ("prob", "current_prob", "base_prob"):
        if key in op:
            return float(op[key])
    return 0.0


def _op_strength(op: dict[str, Any]) -> float:
    for key in ("strength", "current_strength", "base_strength"):
        if key in op:
            return float(op[key])
    return 0.0


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
