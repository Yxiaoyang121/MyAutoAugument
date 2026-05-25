from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np

from AutoAugment.augmentations import apply_augmentation
from AutoAugment.bbox.clip import clip_filter_bboxes


PHOTOMETRIC_OPS = {
    "brightness",
    "contrast",
    "gamma",
    "gaussian_noise",
    "hsv_jitter",
    "clahe",
    "local_contrast",
    "blur_mild",
}
BBOX_OPS = {
    "horizontal_flip",
    "mild_translate",
    "mild_scale",
    "translate",
    "scale",
    "random_scale_translate",
    "mosaic4",
}
META_OPS = {"randaugment_like"}
SUPPORTED_ONLINE_OPS = sorted(
    PHOTOMETRIC_OPS
    | BBOX_OPS
    | META_OPS
    | {"sharpen", "sharpen_mild", "cutout", "cutout_safe", "random_erasing"}
)
COPY_PASTE_OPS = {"copy_paste", "online_copy_paste", "class_balanced_copy_paste"}
SampleProvider = Callable[[np.random.Generator], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]


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
    mosaic: Counter = field(default_factory=Counter)
    randaugment: Counter = field(default_factory=Counter)
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
            "mosaic4": dict(self.mosaic),
            "randaugment_like": dict(self.randaugment),
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
        sample_provider: SampleProvider | None = None,
        total_epochs: int | None = None,
    ) -> None:
        self.policy = deepcopy(policy)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.num_classes = num_classes
        self.copy_paste_enabled = bool(copy_paste_enabled)
        self.stats = stats if stats is not None else OnlineAugmentationStats()
        self.sample_provider = sample_provider
        self.current_epoch = 0
        self.total_epochs = total_epochs

    def set_policy(self, policy: dict[str, Any]) -> None:
        self.policy = deepcopy(policy)

    def set_sample_provider(self, sample_provider: SampleProvider | None) -> None:
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
        current_image = np.asarray(image).copy()
        current_labels = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
        current_bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4).copy()
        input_bbox_count = int(len(current_bboxes))
        audit: dict[str, Any] = {"operations": [], "applied_ops": [], "skipped_ops": []}

        for raw_operation in self.policy.get("operations", []) or []:
            operation = dict(raw_operation)
            raw_name = str(operation.get("name", "")).strip().lower()
            name = _normalize_op_name(raw_name)
            if not bool(operation.get("enabled", True)):
                self.stats.record_op(raw_name or "<empty>", "skipped_disabled")
                continue
            prob = _operation_probability(operation)
            strength = _operation_strength(operation)
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

            if name == "mosaic4" and _mosaic_closed(operation, self.current_epoch, self.total_epochs):
                op_audit["skip_reason"] = "close_mosaic"
                audit["skipped_ops"].append(op_audit)
                self.stats.record_op(raw_name, "skipped_close_mosaic")
                self.stats.mosaic["skipped_close_mosaic"] += 1
                audit["operations"].append(op_audit)
                continue

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

            if name == "hsv_jitter":
                current_image = hsv_jitter(current_image, params=params, strength=strength, rng=generator)
                op_audit["applied"] = True
                audit["applied_ops"].append(op_audit)
                self.stats.record_op(raw_name, "applied")
                audit["operations"].append(op_audit)
                continue

            if name == "random_scale_translate":
                st_result = random_scale_translate(
                    current_image,
                    current_labels,
                    current_bboxes,
                    params=params,
                    strength=strength,
                    rng=generator,
                )
                current_image = st_result.image
                current_labels = st_result.labels
                current_bboxes = st_result.bboxes
                op_audit["applied"] = True
                op_audit["transform_stats"] = st_result.stats
                audit["applied_ops"].append(op_audit)
                self.stats.record_op(raw_name, "applied")
                for key, value in st_result.stats.items():
                    self.stats.record_op(raw_name, key, int(value))
                audit["operations"].append(op_audit)
                continue

            if name == "mosaic4":
                mosaic_result = mosaic4(
                    current_image,
                    current_labels,
                    current_bboxes,
                    sample_provider=self.sample_provider,
                    params=params,
                    strength=strength,
                    rng=generator,
                )
                for key, value in mosaic_result.stats.items():
                    self.stats.mosaic[key] += int(value)
                if mosaic_result.applied:
                    current_image = mosaic_result.image
                    current_labels = mosaic_result.labels
                    current_bboxes = mosaic_result.bboxes
                    op_audit["applied"] = True
                    audit["applied_ops"].append(op_audit)
                    self.stats.record_op(raw_name, "applied")
                else:
                    op_audit["skip_reason"] = mosaic_result.stats.get("skip_reason", "mosaic_unavailable")
                    audit["skipped_ops"].append(op_audit)
                    self.stats.record_op(raw_name, "skipped_mosaic_unavailable")
                audit["operations"].append(op_audit)
                continue

            if name == "randaugment_like":
                rand_result = randaugment_like(
                    current_image,
                    current_labels,
                    current_bboxes,
                    params=params,
                    strength=strength,
                    rng=generator,
                )
                current_image = rand_result.image
                current_labels = rand_result.labels
                current_bboxes = rand_result.bboxes
                for key, value in rand_result.stats.items():
                    if isinstance(value, int):
                        self.stats.randaugment[key] += int(value)
                for key, value in rand_result.cutout_stats.items():
                    self.stats.cutout[key] += int(value)
                op_audit["applied"] = bool(rand_result.selected_ops)
                op_audit["selected_ops"] = rand_result.selected_ops
                if op_audit["applied"]:
                    audit["applied_ops"].append(op_audit)
                    self.stats.record_op(raw_name, "applied")
                    for selected in rand_result.selected_ops:
                        self.stats.randaugment[f"selected_{selected}"] += 1
                else:
                    op_audit["skip_reason"] = "no_randaugment_ops"
                    audit["skipped_ops"].append(op_audit)
                    self.stats.record_op(raw_name, "skipped_no_ops")
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


@dataclass
class GeometryResult:
    image: np.ndarray
    labels: np.ndarray
    bboxes: np.ndarray
    stats: dict[str, int]


@dataclass
class MosaicResult:
    image: np.ndarray
    labels: np.ndarray
    bboxes: np.ndarray
    applied: bool
    stats: dict[str, Any]


@dataclass
class RandAugmentResult:
    image: np.ndarray
    labels: np.ndarray
    bboxes: np.ndarray
    selected_ops: list[str]
    stats: dict[str, int]
    cutout_stats: dict[str, int]


def hsv_jitter(
    image: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    params = params or {}
    if image.ndim < 3 or image.shape[2] < 3:
        return np.asarray(image).copy()
    generator = rng if rng is not None else np.random.default_rng()
    s = float(np.clip(strength, 0.0, 1.0))
    hsv_h = float(params.get("hsv_h", 0.015)) * s
    hsv_s = float(params.get("hsv_s", 0.7)) * s
    hsv_v = float(params.get("hsv_v", 0.4)) * s
    hue_delta = float(generator.uniform(-hsv_h, hsv_h)) * 180.0
    sat_gain = 1.0 + float(generator.uniform(-hsv_s, hsv_s))
    val_gain = 1.0 + float(generator.uniform(-hsv_v, hsv_v))
    original_dtype = image.dtype
    cv_image = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_gain, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_gain, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out.astype(original_dtype, copy=False) if original_dtype != np.uint8 else out


def random_scale_translate(
    image: np.ndarray,
    labels: np.ndarray,
    bboxes: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> GeometryResult:
    params = params or {}
    generator = rng if rng is not None else np.random.default_rng()
    s = float(np.clip(strength, 0.0, 1.0))
    out_image = np.asarray(image).copy()
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4).copy()
    height, width = out_image.shape[:2]
    if s <= 0.0:
        return GeometryResult(out_image, labels_arr, boxes, {"boxes_dropped_visibility": 0, "boxes_dropped_size": 0})

    max_translate = float(params.get("translate", params.get("max_translate", 0.1))) * s
    max_scale_delta = float(params.get("scale", params.get("max_scale_delta", 0.5))) * s
    scale_min = max(0.05, 1.0 - max_scale_delta)
    scale_max = max(scale_min, 1.0 + max_scale_delta)
    scale_value = float(params.get("scale_value", generator.uniform(scale_min, scale_max)))
    tx = float(params.get("tx", generator.uniform(-max_translate, max_translate))) * width
    ty = float(params.get("ty", generator.uniform(-max_translate, max_translate))) * height
    center_x = width / 2.0
    center_y = height / 2.0
    matrix = np.asarray(
        [
            [scale_value, 0.0, center_x * (1.0 - scale_value) + tx],
            [0.0, scale_value, center_y * (1.0 - scale_value) + ty],
        ],
        dtype=np.float32,
    )
    border_value = tuple(params.get("border_value", (114, 114, 114)))
    warped = cv2.warpAffine(
        out_image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    transformed = _apply_affine_to_boxes_xyxy(boxes, matrix)
    original_areas = np.maximum(1.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if len(boxes) else np.zeros((0,))
    filtered_labels, filtered_boxes, filter_stats = _filter_boxes_with_visibility(
        labels_arr,
        transformed,
        width=width,
        height=height,
        min_width=float(params.get("min_width", 2.0)),
        min_height=float(params.get("min_height", 2.0)),
        min_area=float(params.get("min_area", 4.0)),
        min_visibility=float(params.get("min_visibility", 0.2)),
        original_areas=original_areas,
    )
    filter_stats["scale_translate_applied"] = 1
    return GeometryResult(warped, filtered_labels, filtered_boxes, filter_stats)


def mosaic4(
    image: np.ndarray,
    labels: np.ndarray,
    bboxes: np.ndarray,
    *,
    sample_provider: SampleProvider | None,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> MosaicResult:
    params = params or {}
    generator = rng if rng is not None else np.random.default_rng()
    if sample_provider is None:
        return MosaicResult(
            np.asarray(image).copy(),
            np.asarray(labels, dtype=np.int64).reshape(-1).copy(),
            np.asarray(bboxes, dtype=np.float32).reshape(-1, 4).copy(),
            False,
            {"attempted": 1, "applied": 0, "skip_reason": "no_sample_provider"},
        )

    target_size = int(params.get("target_size", params.get("imgsz", max(image.shape[:2]))))
    target_size = max(2, target_size)
    half = target_size // 2
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (
            np.asarray(image).copy(),
            np.asarray(labels, dtype=np.int64).reshape(-1).copy(),
            np.asarray(bboxes, dtype=np.float32).reshape(-1, 4).copy(),
        )
    ]
    provider_failures = 0
    for _ in range(3):
        sample = sample_provider(generator)
        if sample is None:
            provider_failures += 1
            continue
        sample_image, sample_labels, sample_bboxes = sample
        samples.append(
            (
                np.asarray(sample_image).copy(),
                np.asarray(sample_labels, dtype=np.int64).reshape(-1).copy(),
                np.asarray(sample_bboxes, dtype=np.float32).reshape(-1, 4).copy(),
            )
        )
    if len(samples) < 4:
        return MosaicResult(samples[0][0], samples[0][1], samples[0][2], False, {"attempted": 1, "applied": 0, "provider_failures": provider_failures, "skip_reason": "not_enough_sources"})

    channels = samples[0][0].shape[2] if samples[0][0].ndim == 3 else 1
    fill_value = int(params.get("fill_value", 114))
    if channels == 1:
        canvas = np.full((target_size, target_size), fill_value, dtype=samples[0][0].dtype)
    else:
        canvas = np.full((target_size, target_size, channels), fill_value, dtype=samples[0][0].dtype)
    placements = [
        (0, 0, half, half),
        (half, 0, target_size, half),
        (0, half, half, target_size),
        (half, half, target_size, target_size),
    ]
    out_labels: list[np.ndarray] = []
    out_boxes: list[np.ndarray] = []
    for (sample_image, sample_labels, sample_boxes), (x1, y1, x2, y2) in zip(samples[:4], placements):
        tile_w = max(1, x2 - x1)
        tile_h = max(1, y2 - y1)
        resized = cv2.resize(sample_image, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        canvas[y1:y2, x1:x2] = resized
        h, w = sample_image.shape[:2]
        if len(sample_boxes):
            scaled = sample_boxes.astype(np.float32, copy=True)
            scaled[:, [0, 2]] = scaled[:, [0, 2]] * (tile_w / max(1, w)) + x1
            scaled[:, [1, 3]] = scaled[:, [1, 3]] * (tile_h / max(1, h)) + y1
            out_boxes.append(scaled)
            out_labels.append(sample_labels)
    if out_boxes:
        merged_boxes = np.vstack(out_boxes).astype(np.float32, copy=False)
        merged_labels = np.concatenate(out_labels).astype(np.int64, copy=False)
    else:
        merged_boxes = np.zeros((0, 4), dtype=np.float32)
        merged_labels = np.zeros((0,), dtype=np.int64)
    filtered_labels, filtered_boxes, filter_stats = _filter_boxes_with_visibility(
        merged_labels,
        merged_boxes,
        width=target_size,
        height=target_size,
        min_width=float(params.get("min_width", 1.0)),
        min_height=float(params.get("min_height", 1.0)),
        min_area=float(params.get("min_area", 1.0)),
        min_visibility=float(params.get("min_visibility", 0.05)),
    )
    stats: dict[str, Any] = {"attempted": 1, "applied": 1, "source_images": 4, "output_size": target_size, **filter_stats}
    return MosaicResult(canvas, filtered_labels, filtered_boxes, True, stats)


def randaugment_like(
    image: np.ndarray,
    labels: np.ndarray,
    bboxes: np.ndarray,
    *,
    params: dict[str, Any] | None = None,
    strength: float = 1.0,
    rng: np.random.Generator | None = None,
) -> RandAugmentResult:
    params = params or {}
    generator = rng if rng is not None else np.random.default_rng()
    candidates = list(
        params.get(
            "candidate_ops",
            [
                "brightness",
                "contrast",
                "gamma",
                "sharpen_mild",
                "local_contrast",
                "gaussian_noise",
                "blur_mild",
                "cutout_safe",
            ],
        )
    )
    if not candidates:
        return RandAugmentResult(np.asarray(image).copy(), np.asarray(labels, dtype=np.int64), np.asarray(bboxes, dtype=np.float32).reshape(-1, 4), [], {"attempted": 1, "applied": 0}, {})
    min_ops = max(1, int(params.get("min_ops", 1)))
    max_ops = max(min_ops, int(params.get("max_ops", 2)))
    count = int(generator.integers(min_ops, max_ops + 1))
    count = min(count, len(candidates))
    selected = [str(name) for name in generator.choice(np.asarray(candidates, dtype=object), size=count, replace=False).tolist()]
    current_image = np.asarray(image).copy()
    current_labels = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
    current_bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4).copy()
    cutout_stats: Counter = Counter()
    for raw_name in selected:
        name = _normalize_op_name(raw_name)
        op_params = dict(params.get("op_params", {}).get(raw_name, {}) if isinstance(params.get("op_params"), dict) else {})
        if name == "cutout_safe":
            cutout_result = cutout_safe(current_image, current_bboxes, params=op_params or params, strength=min(float(strength), 0.7), rng=generator)
            current_image = cutout_result.image
            cutout_stats.update(cutout_result.stats)
            continue
        if name == "hsv_jitter":
            current_image = hsv_jitter(current_image, params=op_params, strength=strength, rng=generator)
            continue
        mapped_name, mapped_params, mapped_strength = _map_supported_op(name, op_params, strength)
        if mapped_name is None:
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
    return RandAugmentResult(
        current_image,
        current_labels,
        current_bboxes,
        selected,
        {"attempted": 1, "applied": 1, "selected_count": len(selected)},
        dict(cutout_stats),
    )


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


def _operation_probability(operation: dict[str, Any]) -> float:
    for key in ("prob", "current_prob", "base_prob"):
        if key in operation:
            return float(np.clip(operation[key], 0.0, 1.0))
    return 1.0


def _operation_strength(operation: dict[str, Any]) -> float:
    for key in ("strength", "current_strength", "base_strength"):
        if key in operation:
            return float(np.clip(operation[key], 0.0, 1.0))
    return 1.0


def _mosaic_closed(operation: dict[str, Any], current_epoch: int, total_epochs: int | None) -> bool:
    params = dict(operation.get("params", {}) or {})
    close_mosaic = int(operation.get("close_mosaic", params.get("close_mosaic", 0)) or 0)
    if close_mosaic <= 0 or total_epochs is None or int(total_epochs) <= close_mosaic:
        return False
    return int(current_epoch) >= int(total_epochs) - close_mosaic


def _apply_affine_to_boxes_xyxy(bboxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    corners = np.stack(
        [
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ],
        axis=1,
    )
    ones = np.ones((corners.shape[0], corners.shape[1], 1), dtype=np.float32)
    hom = np.concatenate([corners, ones], axis=2)
    transformed = hom @ np.asarray(matrix, dtype=np.float32).T
    mins = transformed.min(axis=1)
    maxs = transformed.max(axis=1)
    return np.concatenate([mins, maxs], axis=1).astype(np.float32, copy=False)


def _filter_boxes_with_visibility(
    labels: np.ndarray,
    bboxes: np.ndarray,
    *,
    width: int,
    height: int,
    min_width: float = 1.0,
    min_height: float = 1.0,
    min_area: float = 1.0,
    min_visibility: float = 0.0,
    original_areas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if len(labels_arr) == 0:
        return labels_arr, boxes, {"boxes_dropped_visibility": 0, "boxes_dropped_size": 0}
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    widths = clipped[:, 2] - clipped[:, 0]
    heights = clipped[:, 3] - clipped[:, 1]
    areas = widths * heights
    size_keep = (widths >= min_width) & (heights >= min_height) & (areas >= min_area)
    if original_areas is None:
        reference_area = np.maximum(1.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    else:
        reference_area = np.maximum(1.0, np.asarray(original_areas, dtype=np.float32).reshape(-1))
        if len(reference_area) != len(boxes):
            reference_area = np.maximum(1.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    visibility = np.divide(areas, reference_area, out=np.zeros_like(areas, dtype=np.float32), where=reference_area > 0)
    visibility_keep = visibility >= float(min_visibility)
    finite_keep = np.isfinite(clipped).all(axis=1)
    keep = size_keep & visibility_keep & finite_keep
    return (
        labels_arr[keep].astype(np.int64, copy=False),
        clipped[keep].astype(np.float32, copy=False),
        {
            "boxes_dropped_visibility": int((size_keep & ~visibility_keep).sum()),
            "boxes_dropped_size": int((~size_keep).sum()),
        },
    )


def _normalize_op_name(name: str) -> str:
    return {
        "safe_cutout": "cutout_safe",
        "cutout": "cutout_safe",
        "random_erasing": "cutout_safe",
        "erasing": "cutout_safe",
        "random_scale_translate": "random_scale_translate",
        "scale_translate": "random_scale_translate",
        "mosaic": "mosaic4",
        "mosaic4": "mosaic4",
        "randaugment": "randaugment_like",
        "randaugment_like": "randaugment_like",
        "sharpen_mild": "sharpen_mild",
        "mild_translate": "mild_translate",
        "mild_scale": "mild_scale",
        "gaussian_blur": "blur_mild",
    }.get(name, name)


def _map_supported_op(
    name: str,
    params: dict[str, Any],
    strength: float,
) -> tuple[str | None, dict[str, Any], float]:
    if name == "blur_mild":
        mild = dict(params)
        mild["max_kernel"] = min(int(mild.get("max_kernel", 3)), 3)
        return "gaussian_blur", mild, min(float(strength), 0.35)
    if name in (PHOTOMETRIC_OPS - {"hsv_jitter", "blur_mild"}) or name == "horizontal_flip":
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
