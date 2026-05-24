from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from AutoAugment.augmentations import apply_augmentation
from AutoAugment.bbox.clip import clip_filter_bboxes


PHOTOMETRIC_OPS = {
    "brightness",
    "contrast",
    "gamma",
    "gaussian_noise",
    "clahe",
    "local_contrast",
}
BBOX_OPS = {"horizontal_flip", "mild_translate", "mild_scale", "translate", "scale"}
SUPPORTED_ONLINE_OPS = sorted(PHOTOMETRIC_OPS | BBOX_OPS | {"sharpen", "sharpen_mild", "cutout", "cutout_safe"})
COPY_PASTE_OPS = {"copy_paste", "online_copy_paste"}


@dataclass
class OnlineAugmentResult:
    image: np.ndarray
    labels: np.ndarray
    bboxes: np.ndarray
    audit: dict[str, Any]


@dataclass
class OnlineAugmentationStats:
    supported_ops: list[str] = field(default_factory=lambda: list(SUPPORTED_ONLINE_OPS))
    samples_seen: int = 0
    samples_augmented: int = 0
    input_bbox_count: int = 0
    output_bbox_count: int = 0
    invalid_bbox_count: int = 0
    bbox_oob_count: int = 0
    class_id_oob_count: int = 0
    op_counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    cutout: Counter = field(default_factory=Counter)
    copy_paste: Counter = field(default_factory=Counter)
    preview_saved: int = 0

    def record_op(self, name: str, key: str, amount: int = 1) -> None:
        self.op_counts[name][key] += int(amount)

    def record_sample(self, input_count: int, output_count: int, augmented: bool) -> None:
        self.samples_seen += 1
        self.input_bbox_count += int(input_count)
        self.output_bbox_count += int(output_count)
        if augmented:
            self.samples_augmented += 1

    def record_validation(self, validation: dict[str, int]) -> None:
        self.invalid_bbox_count += int(validation.get("invalid_bbox_count", 0))
        self.bbox_oob_count += int(validation.get("bbox_oob_count", 0))
        self.class_id_oob_count += int(validation.get("class_id_oob_count", 0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_ops": list(self.supported_ops),
            "copy_paste_status": "pending_object_bank_design",
            "samples_seen": int(self.samples_seen),
            "samples_augmented": int(self.samples_augmented),
            "input_bbox_count": int(self.input_bbox_count),
            "output_bbox_count": int(self.output_bbox_count),
            "invalid_bbox_count": int(self.invalid_bbox_count),
            "bbox_oob_count": int(self.bbox_oob_count),
            "class_id_oob_count": int(self.class_id_oob_count),
            "preview_saved": int(self.preview_saved),
            "ops": {name: dict(counter) for name, counter in sorted(self.op_counts.items())},
            "cutout_safe": dict(self.cutout),
            "copy_paste": dict(self.copy_paste),
        }


class OnlinePolicyAugmentor:
    """Apply an AutoAugment-style policy dynamically to one detection sample."""

    def __init__(
        self,
        policy: dict[str, Any],
        *,
        seed: int = 42,
        num_classes: int | None = None,
        copy_paste_enabled: bool = False,
        stats: OnlineAugmentationStats | None = None,
    ) -> None:
        self.policy = deepcopy(policy)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.num_classes = num_classes
        self.copy_paste_enabled = bool(copy_paste_enabled)
        self.stats = stats if stats is not None else OnlineAugmentationStats()

    def apply(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> OnlineAugmentResult:
        generator = rng if rng is not None else self.rng
        current_image = np.asarray(image).copy()
        current_labels = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
        current_bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4).copy()
        input_bbox_count = int(len(current_bboxes))
        audit: dict[str, Any] = {"operations": [], "applied_ops": [], "skipped_ops": []}

        for raw_operation in self.policy.get("operations", []) or []:
            operation = dict(raw_operation)
            raw_name = str(operation.get("name", "")).strip().lower()
            name = _normalize_op_name(raw_name)
            prob = float(operation.get("prob", 1.0))
            strength = float(operation.get("strength", 1.0))
            params = dict(operation.get("params", {}) or {})
            draw = float(generator.random())
            op_audit = {
                "name": raw_name,
                "normalized_name": name,
                "prob": prob,
                "strength": strength,
                "draw": draw,
                "applied": False,
                "skip_reason": None,
            }
            self.stats.record_op(raw_name or "<empty>", "seen")

            if draw > prob:
                op_audit["skip_reason"] = "probability"
                audit["skipped_ops"].append(op_audit)
                self.stats.record_op(raw_name, "skipped_probability")
                audit["operations"].append(op_audit)
                continue

            if name in COPY_PASTE_OPS:
                op_audit["skip_reason"] = "copy_paste_pending"
                audit["skipped_ops"].append(op_audit)
                self.stats.record_op(raw_name, "skipped_copy_paste_pending")
                self.stats.copy_paste["pending"] += 1
                audit["operations"].append(op_audit)
                continue

            if name == "cutout_safe":
                cutout_result = cutout_safe(
                    current_image,
                    current_bboxes,
                    params=params,
                    strength=strength,
                    rng=generator,
                )
                current_image = cutout_result.image
                for key, value in cutout_result.stats.items():
                    self.stats.cutout[key] += int(value)
                applied = bool(cutout_result.stats.get("holes_applied", 0))
                op_audit["applied"] = applied
                if applied:
                    audit["applied_ops"].append(op_audit)
                    self.stats.record_op(raw_name, "applied")
                else:
                    op_audit["skip_reason"] = "cutout_safety"
                    audit["skipped_ops"].append(op_audit)
                    self.stats.record_op(raw_name, "skipped_safety")
                audit["operations"].append(op_audit)
                continue

            mapped_name, mapped_params, mapped_strength = _map_supported_op(name, params, strength)
            if mapped_name is None:
                op_audit["skip_reason"] = "unsupported"
                audit["skipped_ops"].append(op_audit)
                self.stats.record_op(raw_name or "<empty>", "skipped_unsupported")
                audit["operations"].append(op_audit)
                continue

            current_image, current_labels, current_bboxes = apply_augmentation(
                mapped_name,
                current_image,
                current_labels,
                current_bboxes,
                params=mapped_params,
                strength=mapped_strength,
                rng=generator,
            )
            op_audit["applied"] = True
            audit["applied_ops"].append(op_audit)
            self.stats.record_op(raw_name, "applied")
            audit["operations"].append(op_audit)

        height, width = current_image.shape[:2]
        current_labels, current_bboxes, validation = validate_detection_sample(
            current_labels,
            current_bboxes,
            width=width,
            height=height,
            num_classes=self.num_classes,
        )
        audit["validation"] = validation
        self.stats.record_validation(validation)
        self.stats.record_sample(
            input_count=input_bbox_count,
            output_count=len(current_bboxes),
            augmented=bool(audit["applied_ops"]),
        )
        return OnlineAugmentResult(
            image=current_image,
            labels=current_labels,
            bboxes=current_bboxes,
            audit=audit,
        )


@dataclass
class CutoutResult:
    image: np.ndarray
    stats: dict[str, int]


def cutout_safe(
    image: np.ndarray,
    bboxes: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> CutoutResult:
    params = params or {}
    generator = rng if rng is not None else np.random.default_rng()
    out = np.asarray(image).copy()
    height, width = out.shape[:2]
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    max_holes = max(1, int(params.get("max_holes", 2)))
    max_fraction = float(params.get("max_fraction", 0.18)) * float(np.clip(strength, 0.0, 1.0))
    min_fraction = min(float(params.get("min_fraction", 0.03)), max(0.01, max_fraction))
    max_overlap_ratio = float(params.get("max_overlap_ratio", params.get("max_overlap", 0.05)))
    max_attempts = max(1, int(params.get("max_attempts", 40)))
    fill_value = params.get("fill_value", None)
    stats = {
        "op_attempted": 1,
        "holes_requested": max_holes,
        "holes_applied": 0,
        "holes_skipped_center": 0,
        "holes_skipped_overlap": 0,
        "holes_skipped_no_safe_region": 0,
    }
    if max_fraction <= 0.0:
        stats["holes_skipped_no_safe_region"] = max_holes
        return CutoutResult(out, stats)

    for _ in range(max_holes):
        placed = False
        last_reason = "no_safe_region"
        for _attempt in range(max_attempts):
            erase_w = max(1, int(width * generator.uniform(min_fraction, max(min_fraction, max_fraction))))
            erase_h = max(1, int(height * generator.uniform(min_fraction, max(min_fraction, max_fraction))))
            x1 = int(generator.integers(0, max(1, width - erase_w + 1)))
            y1 = int(generator.integers(0, max(1, height - erase_h + 1)))
            hole = np.asarray([x1, y1, min(width, x1 + erase_w), min(height, y1 + erase_h)], dtype=np.float32)
            safe, reason = _cutout_hole_is_safe(hole, boxes, max_overlap_ratio=max_overlap_ratio)
            if not safe:
                last_reason = reason
                continue
            x2, y2 = int(hole[2]), int(hole[3])
            out[y1:y2, x1:x2] = _cutout_fill_value(out, fill_value)
            stats["holes_applied"] += 1
            placed = True
            break
        if not placed:
            if last_reason == "center":
                stats["holes_skipped_center"] += 1
            elif last_reason == "overlap":
                stats["holes_skipped_overlap"] += 1
            else:
                stats["holes_skipped_no_safe_region"] += 1
    return CutoutResult(out, stats)


def validate_detection_sample(
    labels: np.ndarray,
    bboxes: np.ndarray,
    *,
    width: int,
    height: int,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if len(labels_arr) != len(boxes):
        raise ValueError("labels and bboxes length mismatch after online augmentation")
    if len(labels_arr) == 0:
        return labels_arr, boxes, {
            "raw_bbox_count": 0,
            "valid_bbox_count": 0,
            "invalid_bbox_count": 0,
            "bbox_oob_count": 0,
            "class_id_oob_count": 0,
        }
    finite = np.isfinite(boxes).all(axis=1)
    class_valid = labels_arr >= 0
    if num_classes is not None:
        class_valid &= labels_arr < int(num_classes)
    bbox_oob = (
        (boxes[:, 0] < 0)
        | (boxes[:, 1] < 0)
        | (boxes[:, 2] > width)
        | (boxes[:, 3] > height)
    )
    clipped, clipped_labels, keep_after_clip = clip_filter_bboxes(
        boxes,
        labels_arr,
        width,
        height,
        min_width=1.0,
        min_height=1.0,
        min_area=1.0,
        return_indices=True,
    )
    valid_mask = finite & class_valid & keep_after_clip
    if np.array_equal(valid_mask, keep_after_clip) and class_valid.all() and finite.all():
        out_labels = clipped_labels.astype(np.int64, copy=False)
        out_boxes = clipped.astype(np.float32, copy=False)
    else:
        clipped_all = np.asarray(boxes, dtype=np.float32).copy()
        clipped_all[:, [0, 2]] = np.clip(clipped_all[:, [0, 2]], 0, width)
        clipped_all[:, [1, 3]] = np.clip(clipped_all[:, [1, 3]], 0, height)
        out_labels = labels_arr[valid_mask].astype(np.int64, copy=False)
        out_boxes = clipped_all[valid_mask].astype(np.float32, copy=False)
    invalid_count = int((~valid_mask).sum())
    return out_labels, out_boxes, {
        "raw_bbox_count": int(len(labels_arr)),
        "valid_bbox_count": int(len(out_labels)),
        "invalid_bbox_count": invalid_count,
        "bbox_oob_count": int(bbox_oob.sum()),
        "class_id_oob_count": int((~class_valid).sum()),
    }


def _normalize_op_name(name: str) -> str:
    return {
        "safe_cutout": "cutout_safe",
        "cutout": "cutout_safe",
        "random_erasing": "cutout_safe",
        "sharpen_mild": "sharpen_mild",
        "mild_translate": "mild_translate",
        "mild_scale": "mild_scale",
    }.get(name, name)


def _map_supported_op(
    name: str,
    params: dict[str, Any],
    strength: float,
) -> tuple[str | None, dict[str, Any], float]:
    if name in PHOTOMETRIC_OPS or name == "horizontal_flip":
        return name, params, strength
    if name == "sharpen":
        mild = dict(params)
        mild["amount"] = min(float(mild.get("amount", 0.8)), 0.8)
        return "sharpen", mild, min(float(strength), 0.6)
    if name == "sharpen_mild":
        mild = dict(params)
        mild.setdefault("amount", 0.55)
        return "sharpen", mild, min(float(strength), 0.5)
    if name in {"mild_translate", "translate"}:
        mild = dict(params)
        mild["max_translate"] = min(float(mild.get("max_translate", 0.05)), 0.08)
        return "translate", mild, min(float(strength), 0.7)
    if name in {"mild_scale", "scale"}:
        mild = dict(params)
        mild["max_delta"] = min(float(mild.get("max_delta", 0.12)), 0.18)
        return "scale", mild, min(float(strength), 0.7)
    return None, params, strength


def _cutout_hole_is_safe(
    hole: np.ndarray,
    bboxes: np.ndarray,
    *,
    max_overlap_ratio: float,
) -> tuple[bool, str | None]:
    if bboxes.size == 0:
        return True, None
    centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    center_inside = (
        (centers_x >= hole[0])
        & (centers_x <= hole[2])
        & (centers_y >= hole[1])
        & (centers_y <= hole[3])
    )
    if bool(center_inside.any()):
        return False, "center"
    lt = np.maximum(hole[:2], bboxes[:, :2])
    rb = np.minimum(hole[2:], bboxes[:, 2:])
    wh = np.maximum(0.0, rb - lt)
    intersection = wh[:, 0] * wh[:, 1]
    bbox_area = np.maximum(1.0, (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
    if bool((intersection / bbox_area > max_overlap_ratio).any()):
        return False, "overlap"
    return True, None


def _cutout_fill_value(image: np.ndarray, fill_value: Any) -> Any:
    if fill_value is not None:
        return fill_value
    if image.ndim == 2:
        return int(np.mean(image))
    return [int(value) for value in np.mean(image.reshape(-1, image.shape[-1]), axis=0)]
