from __future__ import annotations

from typing import Any


ISSUE_TYPES = (
    "texture_boundary_weak",
    "low_contrast_fn",
    "high_fp",
    "low_support",
    "weak_localization",
    "low_recall",
    "stable_class",
)


def attribute_class_issues(per_class_diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Score and rank CATF-v2 issues per class."""

    classes: dict[str, Any] = {}
    for class_id, row in (per_class_diagnosis.get("classes") or {}).items():
        scores = score_class_issues(row)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        dominant = ranked[0][0] if ranked else "stable_class"
        secondary = [name for name, score in ranked[1:] if score >= 0.20][:3]
        classes[str(class_id)] = {
            "class_id": int(row["class_id"]),
            "class_name": row.get("class_name", str(class_id)),
            "dominant_issue": dominant,
            "secondary_issues": secondary,
            "issue_scores": scores,
            "diagnosis_confidence": row.get("diagnosis_confidence", 0.0),
            "strong_update_allowed": bool(row.get("strong_update_allowed", False)),
            "threshold_calibration_candidate": bool(scores.get("high_fp", 0.0) >= 0.45),
            "oversampling_candidate": bool(scores.get("low_support", 0.0) >= 0.50 or scores.get("low_recall", 0.0) >= 0.55),
            "copy_paste_candidate": bool(scores.get("low_support", 0.0) >= 0.50),
            "copy_paste_status": "pending_object_bank_design",
        }
    return {
        "epoch": per_class_diagnosis.get("epoch"),
        "classes": classes,
        "summary": summarize(classes),
    }


def score_class_issues(row: dict[str, Any]) -> dict[str, float]:
    precision = float(row.get("Precision", 0.0) or 0.0)
    recall = float(row.get("Recall", 0.0) or 0.0)
    ap50 = float(row.get("AP50", 0.0) or 0.0)
    ap95 = float(row.get("AP50_95", 0.0) or 0.0)
    fp_rate = float(row.get("FP_rate", 0.0) or 0.0)
    fn_rate = float(row.get("FN_rate", 0.0) or 0.0)
    confidence = float(row.get("diagnosis_confidence", 0.0) or 0.0)
    low_support = bool(row.get("low_support", False))
    stable = bool(row.get("stable_class", False))
    raw_scores = {
        "texture_boundary_weak": max(0.0, min(1.0, (ap50 - ap95) / 0.30)) if row.get("texture_boundary_weak") else 0.0,
        "low_contrast_fn": max(0.0, min(1.0, row.get("low_contrast_fn_count", 0) / max(1, row.get("FN", 0)))) if row.get("low_contrast_fn") else 0.0,
        "high_fp": max(fp_rate, max(0.0, (0.70 - precision) / 0.70)) if row.get("high_fp") else 0.0,
        "low_support": 1.0 if low_support else 0.0,
        "weak_localization": max(0.0, min(1.0, (ap50 - ap95) / 0.25)) if row.get("weak_localization") else 0.0,
        "low_recall": max(fn_rate, max(0.0, (0.70 - recall) / 0.70)) if row.get("low_recall") else 0.0,
        "stable_class": 1.0 if stable else 0.0,
    }
    if low_support:
        raw_scores["low_contrast_fn"] = min(raw_scores["low_contrast_fn"], 0.25)
        raw_scores["texture_boundary_weak"] = min(raw_scores["texture_boundary_weak"], 0.25)
    if stable:
        for name in ISSUE_TYPES:
            if name != "stable_class":
                raw_scores[name] = min(raw_scores[name], 0.10)
    weighted = {name: round(float(score) * max(0.25, confidence), 4) for name, score in raw_scores.items()}
    if low_support:
        weighted["low_support"] = 1.0
    if stable:
        weighted["stable_class"] = 1.0
    return weighted


def summarize(classes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts = {name: 0 for name in ISSUE_TYPES}
    for item in classes.values():
        issue = str(item.get("dominant_issue", "stable_class"))
        counts[issue] = counts.get(issue, 0) + 1
    return {
        "class_count": len(classes),
        "dominant_issue_counts": counts,
        "threshold_calibration_candidates": [
            int(item["class_id"]) for item in classes.values() if item.get("threshold_calibration_candidate")
        ],
        "oversampling_candidates": [int(item["class_id"]) for item in classes.values() if item.get("oversampling_candidate")],
        "copy_paste_candidates": [int(item["class_id"]) for item in classes.values() if item.get("copy_paste_candidate")],
    }
