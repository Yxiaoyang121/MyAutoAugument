from __future__ import annotations

from typing import Any


DEFAULT_SEARCH_OPERATIONS = [
    "brightness",
    "contrast",
    "gamma",
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "sharpen",
    "clahe",
    "cutout",
    "horizontal_flip",
    "translate",
    "scale",
    "rotate",
    "affine",
]


def generate_augmentation_advice(error_summary: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    overall = error_summary.get("overall", {})
    by_size = error_summary.get("by_size", {})
    by_class = error_summary.get("by_class", [])
    by_position = error_summary.get("by_position", {})
    quality = error_summary.get("quality", {})
    total_gt = int(overall.get("gt_count", 0) or 0)
    fn_count = int(overall.get("fn_count", 0) or 0)
    fp_count = int(overall.get("fp_count", 0) or 0)
    overall_fn_rate = fn_count / max(1, total_gt)

    issues: list[dict[str, Any]] = []
    weights = {name: 1.0 for name in DEFAULT_SEARCH_OPERATIONS}
    strength_ranges: dict[str, list[float]] = {}
    prob_ranges: dict[str, list[float]] = {}
    comments: list[str] = []

    small_stats = _merge_size_stats(by_size, ["tiny", "small"])
    small_fn_rate = small_stats["fn_count"] / max(1, small_stats["gt_count"])
    if small_stats["gt_count"] > 0 and (small_fn_rate >= overall_fn_rate + 0.15 or small_fn_rate >= 0.4):
        issues.append(
            {
                "issue": "small_object_fn_high",
                "severity": _severity(small_fn_rate),
                "observed_fn_rate": small_fn_rate,
                "recommendations": [
                    "increase_tile_overlap",
                    "enable_object_aware_crop",
                    "reduce_blur_noise_strength",
                    "prefer_sharpen_clahe",
                ],
            }
        )
        weights.update({"sharpen": 2.0, "clahe": 1.8, "scale": 1.4, "translate": 1.3})
        weights.update({"motion_blur": 0.25, "gaussian_blur": 0.35, "gaussian_noise": 0.45})
        strength_ranges.update({"motion_blur": [0.0, 0.2], "gaussian_blur": [0.0, 0.2], "gaussian_noise": [0.0, 0.2]})
        comments.append("Small-object misses are high; keep destructive blur/noise weak and prefer sharpening/CLAHE.")

    edge_stats = by_position.get("edge", {})
    edge_fn_rate = float(edge_stats.get("fn_rate", 0.0) or 0.0)
    if int(edge_stats.get("gt_count", 0) or 0) > 0 and (edge_fn_rate >= overall_fn_rate + 0.15 or edge_fn_rate >= 0.35):
        issues.append(
            {
                "issue": "edge_object_fn_high",
                "severity": _severity(edge_fn_rate),
                "observed_fn_rate": edge_fn_rate,
                "recommendations": ["increase_tile_overlap", "increase_translate", "enable_edge_aware_crop"],
            }
        )
        weights.update({"translate": max(weights["translate"], 2.0), "affine": max(weights["affine"], 1.4), "scale": max(weights["scale"], 1.3)})
        prob_ranges["translate"] = [0.35, 0.85]
        comments.append("Edge misses are high; translation/scale policies are weighted up and crop overlap should be reviewed.")

    fn_quality = quality.get("false_negatives", {})
    fn_quality_count = int(fn_quality.get("count", 0) or 0)
    low_contrast_rate = float(fn_quality.get("low_contrast_rate", 0.0) or 0.0)
    dark_rate = float(fn_quality.get("dark_rate", 0.0) or 0.0)
    bright_rate = float(fn_quality.get("bright_rate", 0.0) or 0.0)
    if fn_quality_count > 0 and low_contrast_rate >= 0.35:
        issues.append(
            {
                "issue": "low_contrast_fn_high",
                "severity": _severity(low_contrast_rate),
                "observed_rate": low_contrast_rate,
                "recommendations": ["increase_contrast", "increase_gamma", "increase_clahe", "use_sharpen"],
            }
        )
        weights.update({"contrast": 2.2, "gamma": 1.7, "clahe": 2.2, "sharpen": max(weights["sharpen"], 1.6)})
        strength_ranges.update({"contrast": [0.1, 0.65], "gamma": [0.1, 0.6], "clahe": [0.1, 0.7]})
        comments.append("False negatives are often low contrast; contrast/gamma/CLAHE are weighted up.")
    if fn_quality_count > 0 and (dark_rate >= 0.3 or bright_rate >= 0.3):
        issues.append(
            {
                "issue": "exposure_fn_high",
                "severity": _severity(max(dark_rate, bright_rate)),
                "dark_rate": dark_rate,
                "bright_rate": bright_rate,
                "recommendations": ["brightness", "gamma", "exposure_jitter"],
            }
        )
        weights.update({"brightness": 1.8, "gamma": max(weights["gamma"], 1.7)})
        strength_ranges.setdefault("brightness", [0.1, 0.55])

    high_fn_classes = []
    for item in by_class:
        gt_count = int(item.get("gt_count", 0) or 0)
        fn_rate = float(item.get("fn_rate", 0.0) or 0.0)
        if gt_count > 0 and fn_rate >= overall_fn_rate + 0.2 and fn_rate >= 0.25:
            high_fn_classes.append({"class_id": item.get("class_id"), "class_name": item.get("class_name"), "fn_rate": fn_rate})
    if high_fn_classes:
        issues.append(
            {
                "issue": "class_recall_imbalance",
                "severity": "medium",
                "classes": high_fn_classes,
                "recommendations": ["class_aware_sampling", "increase_samples_for_low_recall_classes", "class_weighted_policy"],
            }
        )
        comments.append("Some classes have higher FN rates; use class-aware sampling outside the random operation sampler.")

    if fp_count > max(3, 0.25 * max(1, total_gt)):
        issues.append(
            {
                "issue": "false_positive_high",
                "severity": "medium",
                "fp_count": fp_count,
                "recommendations": ["add_hard_negative_tiles", "keep_some_empty_tiles", "reduce_strong_noise_texture"],
            }
        )
        weights["gaussian_noise"] = min(weights["gaussian_noise"], 0.4)
        weights["cutout"] = min(weights["cutout"], 0.6)
        comments.append("False positives are non-trivial; consider hard-negative tiles and avoid strong synthetic texture.")

    fallback_used = False
    if not issues:
        fallback_issues = infer_fallback_issues(error_summary, metrics=metrics)
        fallback_used = bool(fallback_issues)
        for issue in fallback_issues:
            issues.append(issue)
            _apply_fallback_issue_to_search_space(issue, weights, strength_ranges, prob_ranges, comments)

    advisor_search_space = {
        "allowed_operations": DEFAULT_SEARCH_OPERATIONS,
        "operation_weights": weights,
        "strength_ranges": strength_ranges,
        "prob_ranges": prob_ranges,
        "forbidden_combinations": [
            ["gaussian_blur", "motion_blur"],
            ["gaussian_noise", "motion_blur", "gaussian_blur"],
        ],
        "max_ops_per_policy": 4,
        "comments": comments or ["No dominant failure mode found; keep the default space with light weighting."],
    }
    return {
        "method": "heuristic_diagnostic_advisor",
        "issues": issues,
        "main_issues": issues,
        "fallback_used": fallback_used,
        "advisor_search_space": advisor_search_space,
        "notes": [
            "This is a heuristic diagnosis, not reinforcement learning or automatic proof of improvement.",
            "Validate any suggested policy against a no-augmentation baseline.",
        ],
    }


def infer_fallback_issues(error_summary: dict[str, Any], metrics: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Infer actionable issues when strict advisor thresholds found no dominant issue."""
    metrics = metrics or {}
    overall = error_summary.get("overall", {}) if error_summary else {}
    by_class = error_summary.get("by_class", []) if error_summary else []
    by_position = error_summary.get("by_position", {}) if error_summary else {}
    quality = error_summary.get("quality", {}) if error_summary else {}
    total_gt = int(overall.get("gt_count", 0) or 0)
    tp_count = int(overall.get("tp_count", 0) or 0)
    fp_count = int(overall.get("fp_count", 0) or 0)
    fn_count = int(overall.get("fn_count", 0) or 0)
    weak_count = int(overall.get("localization_weak_count", 0) or 0)
    precision = _float_or_none(overall.get("precision"))
    recall = _float_or_none(overall.get("recall"))

    inferred: list[dict[str, Any]] = []
    if fp_count > 0 and (precision is None or precision < 0.98):
        inferred.append(
            {
                "issue": "false_positive_precision_gap",
                "severity": "medium" if precision is None or precision < 0.92 else "low",
                "source": "fallback",
                "fp_count": fp_count,
                "precision": precision,
                "recommendations": ["reduce_noise_cutout_texture", "prefer_mild_contrast_gamma", "add_hard_negative_review"],
            }
        )
    if fn_count > 0 and (recall is None or recall < 0.95):
        inferred.append(
            {
                "issue": "false_negative_recall_gap",
                "severity": "medium" if recall is None or recall < 0.85 else "low",
                "source": "fallback",
                "fn_count": fn_count,
                "recall": recall,
                "recommendations": ["increase_visibility_ops", "prefer_sharpen_clahe", "review_sampling_for_missed_objects"],
            }
        )
    weak_rate = weak_count / max(1, total_gt)
    if weak_count > 0 and weak_rate >= 0.05:
        inferred.append(
            {
                "issue": "weak_localization_gap",
                "severity": _severity(weak_rate),
                "source": "fallback",
                "weak_localization_count": weak_count,
                "weak_localization_rate": weak_rate,
                "recommendations": ["use_mild_scale_translate", "reduce_large_rotation", "avoid_strong_blur"],
            }
        )

    overall_fn_rate = fn_count / max(1, total_gt)
    for name, item in by_position.items():
        gt_count = int(item.get("gt_count", 0) or 0)
        fn_rate = float(item.get("fn_rate", 0.0) or 0.0)
        if gt_count > 0 and fn_rate >= max(0.2, overall_fn_rate + 0.1):
            inferred.append(
                {
                    "issue": "position_fn_gap",
                    "severity": _severity(fn_rate),
                    "source": "fallback",
                    "position": name,
                    "observed_fn_rate": fn_rate,
                    "recommendations": ["increase_translate", "review_crop_overlap", "use_mild_affine"],
                }
            )
            break

    class_precision_gaps = []
    for item in by_class:
        class_fp = int(item.get("fp_count", 0) or 0)
        class_precision = float(item.get("precision", 1.0) or 0.0)
        if class_fp > 0 and class_precision < 0.9:
            class_precision_gaps.append(
                {"class_id": item.get("class_id"), "class_name": item.get("class_name"), "precision": class_precision, "fp_count": class_fp}
            )
    if class_precision_gaps:
        inferred.append(
            {
                "issue": "class_precision_imbalance",
                "severity": "low",
                "source": "fallback",
                "classes": class_precision_gaps,
                "recommendations": ["class_aware_hard_negative_review", "avoid_strong_texture_noise"],
            }
        )

    for group_name in ["false_negatives", "false_positives"]:
        q = quality.get(group_name, {})
        count = int(q.get("count", 0) or 0)
        if count <= 0:
            continue
        low_contrast_rate = float(q.get("low_contrast_rate", 0.0) or 0.0)
        dark_rate = float(q.get("dark_rate", 0.0) or 0.0)
        bright_rate = float(q.get("bright_rate", 0.0) or 0.0)
        if low_contrast_rate >= 0.25:
            inferred.append(
                {
                    "issue": f"{group_name}_low_contrast_hint",
                    "severity": _severity(low_contrast_rate),
                    "source": "fallback",
                    "observed_rate": low_contrast_rate,
                    "recommendations": ["increase_contrast", "increase_clahe", "use_sharpen"],
                }
            )
        if dark_rate >= 0.25 or bright_rate >= 0.25:
            inferred.append(
                {
                    "issue": f"{group_name}_exposure_hint",
                    "severity": _severity(max(dark_rate, bright_rate)),
                    "source": "fallback",
                    "dark_rate": dark_rate,
                    "bright_rate": bright_rate,
                    "recommendations": ["brightness", "gamma", "mild_exposure_jitter"],
                }
            )

    metric_issue = _fallback_issue_from_metrics(metrics)
    if metric_issue is not None:
        inferred.append(metric_issue)

    if not inferred:
        inferred.append(
            {
                "issue": "stable_validation_keep_light_policy",
                "severity": "low",
                "source": "fallback",
                "tp_count": tp_count,
                "gt_count": total_gt,
                "recommendations": ["keep_light_visibility_ops", "avoid_destructive_blur_noise", "use_low_risk_geometry"],
            }
        )
    return inferred


def _fallback_issue_from_metrics(metrics: dict[str, Any]) -> dict[str, Any] | None:
    map50 = _float_or_none(metrics.get("yolo_map50") or metrics.get("map50"))
    map50_95 = _float_or_none(metrics.get("yolo_map50_95") or metrics.get("map50_95"))
    precision = _float_or_none(metrics.get("precision"))
    recall = _float_or_none(metrics.get("recall"))
    if map50_95 is not None and map50_95 < 0.6:
        return {
            "issue": "metric_map50_95_gap",
            "severity": "medium" if map50_95 < 0.45 else "low",
            "source": "metrics_fallback",
            "map50_95": map50_95,
            "recommendations": ["prefer_localization_safe_ops", "reduce_destructive_transforms", "use_mild_scale_translate"],
        }
    if map50 is not None and map50 < 0.75:
        return {
            "issue": "metric_map50_gap",
            "severity": "medium" if map50 < 0.5 else "low",
            "source": "metrics_fallback",
            "map50": map50,
            "recommendations": ["increase_visibility_ops", "reduce_blur_noise_strength"],
        }
    if precision is not None and precision < 0.9:
        return {
            "issue": "metric_precision_gap",
            "severity": "medium",
            "source": "metrics_fallback",
            "precision": precision,
            "recommendations": ["reduce_noise_cutout_texture", "add_hard_negative_review"],
        }
    if recall is not None and recall < 0.9:
        return {
            "issue": "metric_recall_gap",
            "severity": "medium",
            "source": "metrics_fallback",
            "recall": recall,
            "recommendations": ["increase_visibility_ops", "prefer_sharpen_clahe"],
        }
    return None


def _apply_fallback_issue_to_search_space(
    issue: dict[str, Any],
    weights: dict[str, float],
    strength_ranges: dict[str, list[float]],
    prob_ranges: dict[str, list[float]],
    comments: list[str],
) -> None:
    name = str(issue.get("issue", ""))
    if name in {"false_positive_precision_gap", "class_precision_imbalance", "metric_precision_gap"}:
        weights.update({"gaussian_noise": 0.35, "cutout": 0.45, "motion_blur": 0.55, "contrast": 1.35, "gamma": 1.25, "clahe": 1.25})
        strength_ranges.update({"gaussian_noise": [0.0, 0.18], "cutout": [0.0, 0.12], "motion_blur": [0.0, 0.18]})
        prob_ranges.update({"gaussian_noise": [0.1, 0.35], "cutout": [0.05, 0.25]})
        comments.append("Fallback found precision/FP risk; destructive texture noise is down-weighted.")
    elif name in {"false_negative_recall_gap", "metric_recall_gap", "metric_map50_gap"}:
        weights.update({"brightness": 1.5, "contrast": 1.6, "gamma": 1.4, "clahe": 1.7, "sharpen": 1.6, "scale": 1.25})
        strength_ranges.update({"contrast": [0.1, 0.55], "gamma": [0.1, 0.5], "clahe": [0.1, 0.6], "sharpen": [0.05, 0.35]})
        comments.append("Fallback found recall/mAP risk; visibility-preserving operations are weighted up.")
    elif name in {"weak_localization_gap", "position_fn_gap", "metric_map50_95_gap"}:
        weights.update({"translate": 1.8, "scale": 1.6, "affine": 1.25, "rotate": 0.65, "motion_blur": 0.45, "gaussian_blur": 0.55})
        strength_ranges.update({"translate": [0.05, 0.28], "scale": [0.05, 0.25], "affine": [0.03, 0.22], "rotate": [0.02, 0.18]})
        prob_ranges.update({"translate": [0.35, 0.8], "scale": [0.3, 0.75]})
        comments.append("Fallback found localization/position risk; mild geometry is preferred over destructive blur.")
    elif "low_contrast" in name:
        weights.update({"contrast": 1.8, "clahe": 2.0, "gamma": 1.4, "sharpen": 1.5})
        strength_ranges.update({"contrast": [0.1, 0.6], "clahe": [0.1, 0.65], "gamma": [0.1, 0.5]})
        comments.append("Fallback found low-contrast hints; contrast/CLAHE are weighted up.")
    elif "exposure" in name:
        weights.update({"brightness": 1.7, "gamma": 1.6, "clahe": 1.3})
        strength_ranges.update({"brightness": [0.1, 0.5], "gamma": [0.1, 0.55]})
        comments.append("Fallback found exposure hints; brightness/gamma are weighted up.")
    else:
        weights.update(
            {
                "brightness": 1.35,
                "contrast": 1.3,
                "gamma": 1.25,
                "clahe": 1.25,
                "sharpen": 1.2,
                "translate": 1.1,
                "scale": 1.1,
                "gaussian_noise": 0.35,
                "gaussian_blur": 0.45,
                "motion_blur": 0.4,
                "cutout": 0.45,
                "rotate": 0.65,
            }
        )
        strength_ranges.update({"gaussian_noise": [0.0, 0.15], "gaussian_blur": [0.0, 0.18], "motion_blur": [0.0, 0.15], "cutout": [0.0, 0.12]})
        comments.append("Fallback found no dominant error; keep a conservative non-uniform search space.")


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_size_stats(by_size: dict[str, Any], names: list[str]) -> dict[str, int]:
    merged = {"gt_count": 0, "fn_count": 0}
    for name in names:
        item = by_size.get(name, {})
        merged["gt_count"] += int(item.get("gt_count", 0) or 0)
        merged["fn_count"] += int(item.get("fn_count", 0) or 0)
    return merged


def _severity(rate: float) -> str:
    if rate >= 0.6:
        return "high"
    if rate >= 0.3:
        return "medium"
    return "low"
