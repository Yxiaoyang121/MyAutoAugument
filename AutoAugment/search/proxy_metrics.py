from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_TINY_AREA_THRESHOLD = 0.0005
DEFAULT_SMALL_AREA_THRESHOLD = 0.005
DEFAULT_EDGE_MARGIN = 0.05
DEFAULT_RARE_CLASS_COUNT_THRESHOLD = 5

DATASET2_V1_HARD_FILTERS = {
    "total_bbox_valid_rate": 0.850,
    "original_bbox_retention": 0.650,
    "new_bbox_valid_rate": 0.750,
    "small_object_retention": 0.600,
    "class_coverage_after": 0.500,
    "exposure_score": 0.150,
    "strength_penalty": 0.900,
}

DATASET2_V1_SOFT_TARGETS = {
    "total_bbox_valid_rate": 0.995,
    "original_bbox_retention": 0.970,
    "new_bbox_valid_rate": 0.980,
    "small_object_retention": 0.950,
    "tiny_target_retention": 0.930,
    "edge_target_retention": 0.950,
    "class_coverage_after": 0.900,
    "rare_class_retention": 0.900,
    "exposure_score": 0.650,
    "strength_penalty": 0.450,
}

OP_RISK_WEIGHTS = {
    "brightness": 0.4,
    "contrast": 0.4,
    "gamma": 0.4,
    "clahe": 0.4,
    "sharpen": 0.4,
    "horizontal_flip": 0.2,
    "translate": 0.7,
    "scale": 0.7,
    "rotate": 0.9,
    "affine": 0.9,
    "gaussian_noise": 1.0,
    "gaussian_blur": 1.0,
    "motion_blur": 1.0,
    "cutout": 1.1,
}


@dataclass
class ProxyCandidateSelection:
    selected_index: int
    fallback_used: bool
    fallback_reason: str | None


def bbox_retention_raw(before_box_count: int, after_box_count: int) -> float:
    if before_box_count <= 0:
        return 1.0
    return float(np.clip(after_box_count / before_box_count, 0.0, 1.0))


def yolo_bbox_safe_mask(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2.0
    x2 = x + w / 2.0
    y1 = y - h / 2.0
    y2 = y + h / 2.0
    return (x1 >= 0.0) & (y1 >= 0.0) & (x2 <= 1.0) & (y2 <= 1.0) & (w > 0.0) & (h > 0.0)


def yolo_bbox_edge_mask(boxes: np.ndarray, *, edge_margin: float = DEFAULT_EDGE_MARGIN) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2.0
    x2 = x + w / 2.0
    y1 = y - h / 2.0
    y2 = y + h / 2.0
    return (x1 <= edge_margin) | (y1 <= edge_margin) | (x2 >= 1.0 - edge_margin) | (y2 >= 1.0 - edge_margin)


def compute_strength_penalty(policy: Any) -> float:
    operations = list(getattr(policy, "operations", []) or [])
    if not operations:
        return 0.0
    values: list[float] = []
    for operation in operations:
        if isinstance(operation, dict):
            name = str(operation.get("name", "")).strip().lower()
            prob = float(operation.get("prob", 1.0))
            strength = float(operation.get("strength", 1.0))
        else:
            name = str(getattr(operation, "name", "")).strip().lower()
            prob = float(getattr(operation, "prob", 1.0))
            strength = float(getattr(operation, "strength", 1.0))
        # OperationSpec always has strength. The fallback keeps dict-like or external
        # operation objects conservative if they omit it.
        risk = float(OP_RISK_WEIGHTS.get(name, 0.8))
        values.append(prob * strength * risk)
    return float(np.clip(np.mean(values), 0.0, 1.0))


def compute_proxy_score(metrics: dict[str, Any], *, version: str = "dataset2_v1") -> tuple[float, dict[str, float], list[str]]:
    if version != "dataset2_v1":
        raise ValueError(f"unknown proxy score version: {version}")
    required = [
        "bbox_safe_rate",
        "bbox_valid_rate",
        "total_bbox_valid_rate",
        "bbox_retention_raw",
        "original_bbox_retention",
        "small_target_retention",
        "small_object_retention",
        "tiny_target_retention",
        "class_coverage_after",
        "rare_class_retention",
        "edge_target_retention",
        "exposure_diversity_score",
        "strength_penalty",
    ]
    fallback_keys = {
        "total_bbox_valid_rate": "bbox_valid_rate",
        "original_bbox_retention": "bbox_retention_raw",
        "small_object_retention": "small_target_retention",
    }
    missing = [
        key
        for key in required
        if not _is_number(metrics.get(key)) and not _is_number(metrics.get(fallback_keys.get(key, "")))
    ]

    def value(key: str) -> float:
        if key in missing:
            return 0.0
        raw = _float_or_none(metrics.get(key))
        if raw is None and key in fallback_keys:
            raw = _float_or_none(metrics.get(fallback_keys[key]))
        if raw is None:
            return 0.0
        return float(np.clip(raw, 0.0, 1.0))

    bbox_safety_score = min(value("bbox_safe_rate"), value("bbox_valid_rate"), value("total_bbox_valid_rate"))
    bbox_retention_score = value("original_bbox_retention") if "original_bbox_retention" not in missing else value("bbox_retention_raw")
    small_target_retention_score = value("small_object_retention") if "small_object_retention" not in missing else (
        0.6 * value("small_target_retention") + 0.4 * value("tiny_target_retention")
    )
    class_coverage_score = 0.7 * value("class_coverage_after") + 0.3 * value("rare_class_retention")
    edge_target_retention_score = value("edge_target_retention")
    exposure_diversity_score = value("exposure_diversity_score")
    strength_penalty = value("strength_penalty")

    components = {
        "bbox_safety_score": bbox_safety_score,
        "bbox_retention_score": bbox_retention_score,
        "small_target_retention_score": small_target_retention_score,
        "class_coverage_score": class_coverage_score,
        "edge_target_retention_score": edge_target_retention_score,
        "exposure_diversity_score": exposure_diversity_score,
        "strength_penalty": strength_penalty,
    }
    score = (
        0.24 * bbox_safety_score
        + 0.18 * bbox_retention_score
        + 0.18 * small_target_retention_score
        + 0.12 * class_coverage_score
        + 0.10 * edge_target_retention_score
        + 0.10 * exposure_diversity_score
        + 0.08 * (1.0 - strength_penalty)
    )
    return float(np.clip(score, 0.0, 1.0)), components, missing


def compute_safety_score(metrics: dict[str, Any], *, version: str = "dataset2_v1") -> tuple[float, dict[str, float], list[str]]:
    """Compute the soft safety score used for proxy reranking.

    SafetyScore follows the project contract:
    total bbox validity * original bbox retention * small-object retention * exposure score.
    Missing values are treated as zero and reported in the missing list.
    """

    if version != "dataset2_v1":
        raise ValueError(f"unknown safety score version: {version}")
    keys = ["total_bbox_valid_rate", "original_bbox_retention", "small_object_retention", "exposure_score"]
    fallback = {
        "total_bbox_valid_rate": "bbox_valid_rate",
        "original_bbox_retention": "bbox_retention_raw",
        "small_object_retention": "small_target_retention",
    }
    missing: list[str] = []
    components: dict[str, float] = {}
    for key in keys:
        value = _float_or_none(metrics.get(key))
        if value is None and key in fallback:
            value = _float_or_none(metrics.get(fallback[key]))
        if value is None:
            missing.append(key)
            components[key] = 0.0
        else:
            components[key] = float(np.clip(value, 0.0, 1.0))
    score = 1.0
    for key in keys:
        score *= components[key]
    return float(np.clip(score, 0.0, 1.0)), components, missing


def safety_soft_penalty_reasons(metrics: dict[str, Any], *, profile: str = "dataset2_v1") -> list[str]:
    """Return non-fatal safety risks below target thresholds."""

    if profile != "dataset2_v1":
        raise ValueError(f"unknown proxy hard filter profile: {profile}")
    reasons: list[str] = []
    for key, threshold in DATASET2_V1_SOFT_TARGETS.items():
        if key == "rare_class_retention" and not bool(metrics.get("rare_class_present", False)):
            continue
        if key == "new_bbox_valid_rate" and int(metrics.get("new_bbox_count", 0) or 0) <= 0:
            continue
        value = _float_or_none(metrics.get(key))
        if value is None:
            reasons.append(f"{key} missing")
        elif key == "strength_penalty":
            if value > threshold:
                reasons.append(f"{key} {value:.6g} > soft target {threshold:.6g}")
        elif value < threshold:
            reasons.append(f"{key} {value:.6g} < soft target {threshold:.6g}")
    return reasons


def apply_proxy_hard_filter(metrics: dict[str, Any], *, profile: str = "dataset2_v1") -> tuple[bool, list[str]]:
    if profile != "dataset2_v1":
        raise ValueError(f"unknown proxy hard filter profile: {profile}")
    reasons: list[str] = []
    for key in ["total_bbox_valid_rate", "original_bbox_retention", "small_object_retention", "class_coverage_after", "exposure_score"]:
        threshold = DATASET2_V1_HARD_FILTERS[key]
        value = _metric_with_fallback(metrics, key)
        if value is None:
            reasons.append(f"{key} missing")
        elif value < threshold:
            reasons.append(f"{key} {value:.6g} < {threshold:.6g}")

    if int(metrics.get("new_bbox_count", 0) or 0) > 0:
        threshold = DATASET2_V1_HARD_FILTERS["new_bbox_valid_rate"]
        value = _metric_with_fallback(metrics, "new_bbox_valid_rate")
        if value is None:
            reasons.append("new_bbox_valid_rate missing")
        elif value < threshold:
            reasons.append(f"new_bbox_valid_rate {value:.6g} < {threshold:.6g}")

    class_oob = int(metrics.get("class_out_of_range_count", 0) or 0)
    invalid_boxes = int(metrics.get("invalid_bbox_count", 0) or 0)
    if class_oob > 0:
        reasons.append(f"class_out_of_range_count {class_oob} > 0")
    if invalid_boxes > 0:
        reasons.append(f"invalid_bbox_count {invalid_boxes} > 0")

    strength_threshold = DATASET2_V1_HARD_FILTERS["strength_penalty"]
    strength_value = _float_or_none(metrics.get("strength_penalty"))
    if strength_value is None:
        reasons.append("strength_penalty missing")
    elif strength_value > strength_threshold:
        reasons.append(f"strength_penalty {strength_value:.6g} > {strength_threshold:.6g}")
    return not reasons, reasons


def select_proxy_candidate(candidates: list[dict[str, Any]]) -> ProxyCandidateSelection:
    if not candidates:
        raise ValueError("candidates must not be empty")
    passed = [candidate for candidate in candidates if bool(candidate.get("hard_filter_pass"))]
    if passed:
        selected = max(
            passed,
            key=lambda item: float(item.get("combined_proxy_safety_score", item.get("proxy_score", 0.0)) or 0.0),
        )
        return ProxyCandidateSelection(int(selected["candidate_index"]), False, None)
    selected = min(
        candidates,
        key=lambda item: (
            len(item.get("hard_filter_reasons") or []),
            -float(item.get("combined_proxy_safety_score", item.get("proxy_score", 0.0)) or 0.0),
        ),
    )
    return ProxyCandidateSelection(int(selected["candidate_index"]), True, "no_candidate_passed_hard_filter")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_with_fallback(metrics: dict[str, Any], key: str) -> float | None:
    fallback = {
        "total_bbox_valid_rate": "bbox_valid_rate",
        "original_bbox_retention": "bbox_retention_raw",
        "small_object_retention": "small_target_retention",
    }
    value = _float_or_none(metrics.get(key))
    if value is None and key in fallback:
        value = _float_or_none(metrics.get(fallback[key]))
    return value


def _is_number(value: Any) -> bool:
    return _float_or_none(value) is not None
