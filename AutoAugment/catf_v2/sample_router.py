from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from AutoAugment.online_augmentation import (
    OnlineAugmentationStats,
    OnlineAugmentResult,
    cutout_safe,
    validate_detection_sample,
)
from AutoAugment.catf_v2.high_risk_class_ops import build_riskguard_event, is_high_risk_class_op


PHOTOMETRIC_OPS = {"clahe", "gamma", "brightness", "contrast"}
ROI_OPS = {"sharpen_mild", "local_contrast", "gamma", "clahe"}
DOMAIN_PRIOR_BLOCKED_ROI_OPS = {"gamma", "clahe"}


@dataclass
class ROIStats:
    roi_aug_applied: int = 0
    roi_aug_skipped_small_roi: int = 0
    roi_aug_skipped_conflict: int = 0
    affected_classes: dict[str, int] = field(default_factory=dict)

    def record_applied(self, class_id: int) -> None:
        self.roi_aug_applied += 1
        key = str(int(class_id))
        self.affected_classes[key] = int(self.affected_classes.get(key, 0)) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_aug_applied": int(self.roi_aug_applied),
            "roi_aug_skipped_small_roi": int(self.roi_aug_skipped_small_roi),
            "roi_aug_skipped_conflict": int(self.roi_aug_skipped_conflict),
            "affected_classes": dict(sorted(self.affected_classes.items(), key=lambda item: int(item[0]))),
        }


class SampleAwareAugmentationRouter:
    """Route industrial augmentation by classes present in the current image."""

    def __init__(
        self,
        policy_matrix: dict[str, Any],
        *,
        seed: int = 42,
        num_classes: int | None = None,
        stats: OnlineAugmentationStats | None = None,
        roi_stats: ROIStats | None = None,
        roi_aware: bool = True,
        sample_aware: bool = True,
        total_epochs: int | None = None,
        riskguard_enabled: bool = False,
        riskguard_sampler_only: bool = True,
    ) -> None:
        self.policy_matrix = deepcopy(policy_matrix)
        self.rng = np.random.default_rng(int(seed))
        self.num_classes = num_classes
        self.stats = stats if stats is not None else OnlineAugmentationStats()
        self.roi_stats = roi_stats if roi_stats is not None else ROIStats()
        self.roi_aware = bool(roi_aware)
        self.sample_aware = bool(sample_aware)
        self.sample_provider = None
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.random_draw_count = 0
        self.riskguard_enabled = bool(riskguard_enabled)
        self.riskguard_sampler_only = bool(riskguard_sampler_only)
        self.riskguard_events: list[dict[str, Any]] = []
        self.weak_image_aug_counts: dict[str, int] = {}

    def set_policy(self, policy_matrix: dict[str, Any]) -> None:
        self.policy_matrix = deepcopy(policy_matrix)

    def set_sample_provider(self, sample_provider: Any | None) -> None:
        self.sample_provider = sample_provider

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        self.current_epoch = max(0, int(epoch))
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)

    def apply(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> OnlineAugmentResult:
        generator = rng if rng is not None else self.rng
        original_image = image
        original_labels = labels
        original_bboxes = bboxes
        labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
        bboxes_arr = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        input_bbox_count = int(len(bboxes_arr))
        audit: dict[str, Any] = {
            "operations": [],
            "applied_ops": [],
            "skipped_ops": [],
            "router": {},
            "applied_any_aug": False,
        }
        present = sorted({int(value) for value in labels_arr.tolist()})
        class_policies = self.policy_matrix.get("classes", {}) or {}
        if not present:
            audit["router"]["skip_reason"] = "empty_labels"
            return self._bypass(original_image, original_labels, original_bboxes, input_bbox_count, audit)

        present_policies = [class_policies.get(str(class_id), {}) for class_id in present]
        all_stable = bool(present_policies) and all(_is_stable(item) for item in present_policies)
        if all_stable:
            audit["router"]["skip_reason"] = "stable_classes_only"
            return self._bypass(original_image, original_labels, original_bboxes, input_bbox_count, audit)

        high_fp_present = any(_is_high_fp_guarded(item) for item in present_policies)
        guarded_or_no_aug_classes = {
            class_id
            for class_id in present
            if _is_high_fp_guarded(class_policies.get(str(class_id), {})) or _is_no_aug(class_policies.get(str(class_id), {}))
        }
        active_targets = [class_id for class_id in present if _has_active_ops(class_policies.get(str(class_id), {}))]
        if not active_targets:
            audit["router"]["skip_reason"] = "no_active_target_class"
            return self._bypass(original_image, original_labels, original_bboxes, input_bbox_count, audit)

        current_image = np.asarray(image).copy()
        current_labels = labels_arr.copy()
        current_bboxes = bboxes_arr.copy()
        for class_id in active_targets:
            policy = class_policies.get(str(class_id), {})
            if _is_high_fp_guarded(policy) or _is_no_aug(policy):
                self.roi_stats.roi_aug_skipped_conflict += 1
                continue
            if self._weak_interval_limit_reached(policy, class_id):
                audit["skipped_ops"].append(
                    {
                        "name": "weak_image_aug_interval_cap",
                        "class_id": int(class_id),
                        "applied": False,
                        "skip_reason": "weak_image_aug_interval_cap",
                    }
                )
                continue
            conflict_bboxes = current_bboxes[np.isin(current_labels, list(guarded_or_no_aug_classes - {class_id}))]
            for op_name, op in sorted((policy.get("ops") or {}).items()):
                prob = float(op.get("prob", 0.0) or 0.0)
                strength = float(op.get("strength", 0.0) or 0.0)
                if prob <= 0.0 or strength <= 0.0:
                    continue
                routed_prob = prob
                if high_fp_present and op_name in PHOTOMETRIC_OPS:
                    routed_prob *= 0.5
                op_audit = {
                    "name": op_name,
                    "class_id": int(class_id),
                    "prob": prob,
                    "routed_prob": routed_prob,
                    "strength": strength,
                    "draw": None,
                    "applied": False,
                    "skip_reason": None,
                }
                if self.riskguard_enabled and is_high_risk_class_op(class_id, op_name):
                    event = build_riskguard_event(
                        class_id=class_id,
                        op_name=op_name,
                        epoch=self.current_epoch,
                        source="sample_router",
                        sampler_only_fallback=self.riskguard_sampler_only,
                    )
                    self.riskguard_events.append(event)
                    op_audit["blocked_by_risk_guard"] = True
                    op_audit["risk_reasons"] = event.get("risk_reasons", [])
                    op_audit["skip_reason"] = "risk_guard"
                    audit["router"]["riskguard_blocked"] = True
                    audit["router"]["riskguard_block_count"] = int(audit["router"].get("riskguard_block_count", 0) or 0) + 1
                    audit["skipped_ops"].append(op_audit)
                    audit["operations"].append(op_audit)
                    continue
                self.stats.record_op(op_name, "seen")
                if _is_domain_prior(policy) and op_name in DOMAIN_PRIOR_BLOCKED_ROI_OPS:
                    op_audit["skip_reason"] = "domain_prior_blocks_roi_photometric"
                    audit["skipped_ops"].append(op_audit)
                    self.stats.record_op(op_name, "skipped_domain_prior")
                    audit["operations"].append(op_audit)
                    continue
                if self.roi_aware and op_name in ROI_OPS and not self._roi_op_available(
                    current_labels,
                    current_bboxes,
                    class_id,
                    conflict_bboxes=conflict_bboxes,
                    image_shape=current_image.shape,
                ):
                    op_audit["skip_reason"] = "roi_unavailable"
                    audit["skipped_ops"].append(op_audit)
                    self.stats.record_op(op_name, "skipped_roi_unavailable")
                    audit["operations"].append(op_audit)
                    continue
                self.random_draw_count += 1
                draw = float(generator.random())
                op_audit["draw"] = draw
                if draw > routed_prob:
                    op_audit["skip_reason"] = "probability"
                    audit["skipped_ops"].append(op_audit)
                    self.stats.record_op(op_name, "skipped_probability")
                    audit["operations"].append(op_audit)
                    continue
                if self.roi_aware and op_name in ROI_OPS:
                    applied = self._apply_roi_op(
                        current_image,
                        current_labels,
                        current_bboxes,
                        class_id,
                        op_name,
                        strength,
                        conflict_bboxes=conflict_bboxes,
                    )
                    if applied:
                        op_audit["applied"] = True
                        audit["applied_ops"].append(op_audit)
                        self.stats.record_op(op_name, "applied")
                        self._record_weak_interval_apply(policy, class_id)
                    else:
                        op_audit["skip_reason"] = "roi_unavailable"
                        audit["skipped_ops"].append(op_audit)
                        self.stats.record_op(op_name, "skipped_roi_unavailable")
                    audit["operations"].append(op_audit)
                    continue
                if op_name == "cutout_safe":
                    result = cutout_safe(
                        current_image,
                        current_bboxes,
                        params={"max_holes": 1, "max_fraction": 0.08, "max_overlap_ratio": 0.10},
                        strength=strength,
                        rng=generator,
                    )
                    current_image = result.image
                    for key, value in result.stats.items():
                        self.stats.cutout[key] += int(value)
                    applied = bool(result.stats.get("holes_applied", 0))
                    op_audit["applied"] = applied
                    if applied:
                        audit["applied_ops"].append(op_audit)
                        self.stats.record_op(op_name, "applied")
                        self._record_weak_interval_apply(policy, class_id)
                    else:
                        op_audit["skip_reason"] = "cutout_safety"
                        audit["skipped_ops"].append(op_audit)
                        self.stats.record_op(op_name, "skipped_safety")
                    audit["operations"].append(op_audit)
                    continue
                op_audit["skip_reason"] = "unsupported_router_op"
                audit["skipped_ops"].append(op_audit)
                self.stats.record_op(op_name, "skipped_unsupported_router_op")
                audit["operations"].append(op_audit)
        if not audit.get("applied_ops"):
            audit["router"]["skip_reason"] = audit["router"].get("skip_reason") or "no_operation_applied"
            return self._bypass(original_image, original_labels, original_bboxes, input_bbox_count, audit)
        return self._finish(current_image, current_labels, current_bboxes, input_bbox_count, audit)

    def _weak_interval_key(self, policy: dict[str, Any], class_id: int) -> str | None:
        weak = policy.get("weak_image_aug") or {}
        if not bool(weak.get("enabled", False)):
            return None
        start_epoch = int(weak.get("interval_start_epoch", self.current_epoch) or self.current_epoch)
        return f"{int(class_id)}:{start_epoch}"

    def _weak_interval_limit_reached(self, policy: dict[str, Any], class_id: int) -> bool:
        weak = policy.get("weak_image_aug") or {}
        if not bool(weak.get("enabled", False)):
            return False
        cap = int(weak.get("max_aug_samples_per_interval", 0) or 0)
        if cap <= 0:
            return False
        key = self._weak_interval_key(policy, class_id)
        if key is None:
            return False
        return int(self.weak_image_aug_counts.get(key, 0) or 0) >= cap

    def _record_weak_interval_apply(self, policy: dict[str, Any], class_id: int) -> None:
        key = self._weak_interval_key(policy, class_id)
        if key is None:
            return
        self.weak_image_aug_counts[key] = int(self.weak_image_aug_counts.get(key, 0) or 0) + 1

    def _apply_roi_op(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        class_id: int,
        op_name: str,
        strength: float,
        conflict_bboxes: np.ndarray | None = None,
    ) -> bool:
        applied = False
        height, width = image.shape[:2]
        for bbox in bboxes[labels == int(class_id)]:
            x1, y1, x2, y2 = expand_box(bbox, width=width, height=height, factor=1.5)
            if conflict_bboxes is not None and len(conflict_bboxes) and any(overlaps((x1, y1, x2, y2), other) for other in conflict_bboxes):
                self.roi_stats.roi_aug_skipped_conflict += 1
                continue
            if x2 - x1 < 8 or y2 - y1 < 8:
                self.roi_stats.roi_aug_skipped_small_roi += 1
                continue
            roi = image[y1:y2, x1:x2].copy()
            augmented = apply_roi_operation(roi, op_name, strength)
            blend_roi(image, augmented, x1, y1, x2, y2)
            self.roi_stats.record_applied(class_id)
            applied = True
        return applied

    def _roi_op_available(
        self,
        labels: np.ndarray,
        bboxes: np.ndarray,
        class_id: int,
        *,
        conflict_bboxes: np.ndarray | None = None,
        image_shape: tuple[int, ...],
    ) -> bool:
        height, width = image_shape[:2]
        available = False
        for bbox in bboxes[labels == int(class_id)]:
            x1, y1, x2, y2 = expand_box(bbox, width=width, height=height, factor=1.5)
            if conflict_bboxes is not None and len(conflict_bboxes) and any(overlaps((x1, y1, x2, y2), other) for other in conflict_bboxes):
                self.roi_stats.roi_aug_skipped_conflict += 1
                continue
            if x2 - x1 < 8 or y2 - y1 < 8:
                self.roi_stats.roi_aug_skipped_small_roi += 1
                continue
            available = True
        return available

    def _bypass(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        input_bbox_count: int,
        audit: dict[str, Any],
    ) -> OnlineAugmentResult:
        audit["applied_any_aug"] = False
        audit["validation"] = {
            "skipped": True,
            "reason": "no_augmentation_applied",
            "invalid_bbox_count": 0,
            "bbox_oob_count": 0,
            "class_id_oob_count": 0,
        }
        output_count = int(len(np.asarray(bboxes).reshape(-1, 4))) if np.asarray(bboxes).size else 0
        self.stats.record_sample(input_count=input_bbox_count, output_count=output_count, augmented=False)
        return OnlineAugmentResult(image=image, labels=labels, bboxes=bboxes, audit=audit)

    def _finish(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        input_bbox_count: int,
        audit: dict[str, Any],
    ) -> OnlineAugmentResult:
        height, width = image.shape[:2]
        labels, bboxes, validation = validate_detection_sample(labels, bboxes, width=width, height=height, num_classes=self.num_classes)
        audit["applied_any_aug"] = bool(audit.get("applied_ops"))
        audit["validation"] = validation
        self.stats.record_validation(validation)
        self.stats.record_sample(input_count=input_bbox_count, output_count=len(bboxes), augmented=bool(audit.get("applied_ops")))
        return OnlineAugmentResult(image=image, labels=labels, bboxes=bboxes, audit=audit)


def expand_box(bbox: np.ndarray, *, width: int, height: int, factor: float = 1.5) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    box_w = max(1.0, (x2 - x1) * factor)
    box_h = max(1.0, (y2 - y1) * factor)
    nx1 = max(0, int(round(cx - 0.5 * box_w)))
    ny1 = max(0, int(round(cy - 0.5 * box_h)))
    nx2 = min(width, int(round(cx + 0.5 * box_w)))
    ny2 = min(height, int(round(cy + 0.5 * box_h)))
    return nx1, ny1, nx2, ny2


def apply_roi_operation(roi: np.ndarray, op_name: str, strength: float) -> np.ndarray:
    s = float(np.clip(strength, 0.0, 1.0))
    if op_name == "gamma":
        gamma = 1.0 + (0.6 * (s - 0.5))
        inv = 1.0 / max(0.1, gamma)
        table = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(roi, table)
    if op_name == "clahe":
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip_limit = 1.5 + 2.0 * s
        l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if op_name == "local_contrast":
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced = cv2.createCLAHE(clipLimit=1.5 + 1.5 * s, tileGridSize=(8, 8)).apply(l)
        mixed = cv2.addWeighted(l, 1.0 - 0.5 * s, enhanced, 0.5 * s, 0)
        return cv2.cvtColor(cv2.merge([mixed, a, b]), cv2.COLOR_LAB2BGR)
    if op_name == "sharpen_mild":
        blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(roi, 1.0 + 0.8 * s, blurred, -0.8 * s, 0)
    return roi.copy()


def blend_roi(image: np.ndarray, augmented: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    feather = max(2, min(h, w) // 8)
    mask = np.ones((h, w), dtype=np.float32)
    if feather > 0:
        ramp_x = np.minimum(np.linspace(0, 1, feather), 1.0)
        ramp_y = np.minimum(np.linspace(0, 1, feather), 1.0)
        mask[:, :feather] *= ramp_x[None, :]
        mask[:, -feather:] *= ramp_x[::-1][None, :]
        mask[:feather, :] *= ramp_y[:, None]
        mask[-feather:, :] *= ramp_y[::-1][:, None]
    mask3 = mask[:, :, None]
    image[y1:y2, x1:x2] = np.clip(augmented.astype(np.float32) * mask3 + roi.astype(np.float32) * (1.0 - mask3), 0, 255).astype(
        np.uint8
    )


def _is_stable(policy: dict[str, Any]) -> bool:
    return policy.get("status") == "frozen" or policy.get("state") == "frozen" or policy.get("dominant_issue") == "stable_class"


def _is_high_fp_guarded(policy: dict[str, Any]) -> bool:
    weak = policy.get("weak_image_aug") or {}
    if bool(weak.get("enabled", False)) and bool(weak.get("precision_aware_gate_passed", False)) and bool(
        weak.get("non_active_regression_gate_passed", False)
    ):
        return False
    guards = policy.get("guards", {}) or {}
    return bool(guards.get("high_fp_guarded") or policy.get("dominant_issue") == "high_fp")


def _is_no_aug(policy: dict[str, Any]) -> bool:
    return bool(policy.get("no_aug_class", False))


def _is_domain_prior(policy: dict[str, Any]) -> bool:
    return bool(policy.get("domain_high_fp_prior", False))


def _has_active_ops(policy: dict[str, Any]) -> bool:
    if _is_no_aug(policy) or _is_high_fp_guarded(policy):
        return False
    if policy.get("status") not in {"active", "pending", "accepted"}:
        return False
    return any(float(op.get("prob", 0.0) or 0.0) > 0.0 for op in (policy.get("ops") or {}).values())


def overlaps(box: tuple[int, int, int, int], other: np.ndarray) -> bool:
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = [float(value) for value in other]
    inter_w = max(0.0, min(float(x2), ox2) - max(float(x1), ox1))
    inter_h = max(0.0, min(float(y2), oy2) - max(float(y1), oy1))
    return inter_w * inter_h > 0.0
