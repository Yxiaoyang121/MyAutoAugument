from __future__ import annotations

from pathlib import Path
from typing import Any

from AutoAugment.diagnostics import generate_augmentation_advice
from AutoAugment.diagnostics.yolo_error_analysis import analyze_yolo_errors
from AutoAugment.diagnostic_pipeline.common import write_json, write_markdown


ISSUE_TYPE_MAP = {
    "small_object_fn_high": "small_object_low_recall",
    "low_contrast_fn_high": "low_contrast_missed_defect",
    "exposure_fn_high": "low_contrast_missed_defect",
    "weak_localization_gap": "localization_bias",
    "position_fn_gap": "localization_bias",
    "metric_map50_95_gap": "localization_bias",
    "false_positive_high": "high_false_positive",
    "false_positive_precision_gap": "high_false_positive",
    "metric_precision_gap": "high_false_positive",
    "class_recall_imbalance": "class_imbalance",
    "class_precision_imbalance": "class_imbalance",
    "edge_object_fn_high": "background_interference",
}

SUGGESTED_AUGMENTATIONS = {
    "small_object_low_recall": ["tiling", "object-aware-crop", "scale", "copy-paste", "mild-geometry"],
    "low_contrast_missed_defect": ["contrast", "gamma", "clahe", "brightness", "sharpen"],
    "localization_bias": ["mild-scale", "mild-translate", "reduce-rotate", "reduce-shear", "reduce-perspective"],
    "high_false_positive": ["hard-negative-review", "reduce-noise", "reduce-blur", "mild-lighting"],
    "class_imbalance": ["class-aware-sampling", "targeted-augmentation", "class-balanced-policy"],
    "background_interference": ["background-diversity", "light-noise", "illumination-jitter"],
    "stable_validation_keep_light_policy": ["light-visibility-ops", "avoid-destructive-transforms"],
}


def run_error_diagnosis(
    *,
    val_images_dir: str | Path,
    val_labels_dir: str | Path,
    predictions_dir: str | Path,
    output_dir: str | Path,
    class_names: dict[int, str] | None = None,
    match_iou: float = 0.5,
    localization_weak_iou: float = 0.3,
) -> dict[str, Any]:
    """Analyze validation prediction errors and write diagnosis artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analysis = analyze_yolo_errors(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        predictions_dir=predictions_dir,
        class_names=class_names,
        match_iou=match_iou,
        localization_weak_iou=localization_weak_iou,
    )
    advice = generate_augmentation_advice(analysis["summary"])
    diagnosis = build_diagnosis_schema(analysis=analysis, advice=advice)
    write_json(output / "diagnosis.json", diagnosis)
    write_json(output / "raw_error_analysis.json", analysis)
    write_json(output / "augmentation_advice.json", advice)
    write_diagnosis_summary(output / "diagnosis_summary.md", diagnosis)
    return diagnosis


def write_dry_run_diagnosis(output_dir: str | Path) -> dict[str, Any]:
    """Write a planned diagnosis artifact for dry-run execution."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    diagnosis = {
        "stage": "error_diagnosis",
        "status": "planned",
        "dry_run": True,
        "global": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "localization_weak": 0},
        "issues": [
            {
                "type": "stable_validation_keep_light_policy",
                "severity": "low",
                "evidence": {"reason": "dry_run_without_predictions"},
                "suggested_augmentations": SUGGESTED_AUGMENTATIONS["stable_validation_keep_light_policy"],
            }
        ],
        "per_class": {},
        "per_image": [],
        "notes": ["Dry-run does not execute YOLO prediction; this placeholder keeps downstream planning auditable."],
    }
    write_json(output / "diagnosis.json", diagnosis)
    write_diagnosis_summary(output / "diagnosis_summary.md", diagnosis)
    return diagnosis


def build_diagnosis_schema(*, analysis: dict[str, Any], advice: dict[str, Any]) -> dict[str, Any]:
    """Convert raw analyzer output into the stable diagnosis.json schema."""

    summary = analysis.get("summary", {})
    overall = summary.get("overall", {})
    issues = [_normalize_issue(item) for item in advice.get("issues", [])]
    issues.extend(_extra_structural_issues(summary, existing_types={item["type"] for item in issues}))
    per_class = {
        str(item["class_id"]): {
            "class_id": int(item["class_id"]),
            "class_name": item.get("class_name", str(item["class_id"])),
            "gt": int(item.get("gt_count", 0)),
            "tp": int(item.get("tp_count", 0)),
            "fp": int(item.get("fp_count", 0)),
            "fn": int(item.get("fn_count", 0)),
            "precision": float(item.get("precision", 0.0)),
            "recall": float(item.get("recall", 0.0)),
            "fn_rate": float(item.get("fn_rate", 0.0)),
            "fp_rate": float(item.get("fp_rate", 0.0)),
        }
        for item in summary.get("by_class", [])
    }
    return {
        "stage": "error_diagnosis",
        "status": "completed",
        "dry_run": False,
        "global": {
            "tp": int(overall.get("tp_count", 0)),
            "fp": int(overall.get("fp_count", 0)),
            "fn": int(overall.get("fn_count", 0)),
            "precision": float(overall.get("precision", 0.0)),
            "recall": float(overall.get("recall", 0.0)),
            "localization_weak": int(overall.get("localization_weak_count", 0)),
            "gt": int(overall.get("gt_count", 0)),
        },
        "issues": issues,
        "per_class": per_class,
        "per_image": analysis.get("per_image_errors", []),
        "size_recall": summary.get("by_size", {}),
        "quality": summary.get("quality", {}),
        "position": summary.get("by_position", {}),
    }


def write_diagnosis_summary(path: str | Path, diagnosis: dict[str, Any]) -> None:
    """Write a compact Markdown diagnosis summary."""

    global_metrics = diagnosis.get("global", {})
    lines = [
        "# Validation Error Diagnosis",
        "",
        f"- Status: {diagnosis.get('status')}",
        f"- TP: {global_metrics.get('tp', 0)}",
        f"- FP: {global_metrics.get('fp', 0)}",
        f"- FN: {global_metrics.get('fn', 0)}",
        f"- Precision: {float(global_metrics.get('precision', 0.0)):.4f}",
        f"- Recall: {float(global_metrics.get('recall', 0.0)):.4f}",
        "",
        "## Issues",
    ]
    for issue in diagnosis.get("issues", []):
        lines.append(f"- {issue.get('type')} severity={issue.get('severity')} suggestions={', '.join(issue.get('suggested_augmentations', []))}")
    lines.extend(["", "## Per-Class Summary"])
    for item in diagnosis.get("per_class", {}).values():
        lines.append(
            f"- {item['class_name']}: gt={item['gt']} tp={item['tp']} fp={item['fp']} "
            f"fn={item['fn']} precision={item['precision']:.4f} recall={item['recall']:.4f}"
        )
    write_markdown(path, lines)


def _normalize_issue(issue: dict[str, Any]) -> dict[str, Any]:
    source = str(issue.get("issue", issue.get("type", "unknown_issue")))
    issue_type = ISSUE_TYPE_MAP.get(source, source)
    suggested = SUGGESTED_AUGMENTATIONS.get(issue_type, list(issue.get("recommendations", [])))
    evidence = {key: value for key, value in issue.items() if key not in {"recommendations"}}
    return {
        "type": issue_type,
        "source_issue": source,
        "severity": str(issue.get("severity", "medium")),
        "evidence": evidence,
        "suggested_augmentations": suggested,
    }


def _extra_structural_issues(summary: dict[str, Any], *, existing_types: set[str]) -> list[dict[str, Any]]:
    extras: list[dict[str, Any]] = []
    by_class = summary.get("by_class", [])
    gt_counts = [int(item.get("gt_count", 0) or 0) for item in by_class if int(item.get("gt_count", 0) or 0) > 0]
    if gt_counts and max(gt_counts) / max(1, min(gt_counts)) >= 3.0 and "class_imbalance" not in existing_types:
        extras.append(
            {
                "type": "class_imbalance",
                "source_issue": "class_count_distribution",
                "severity": "medium",
                "evidence": {"min_class_count": min(gt_counts), "max_class_count": max(gt_counts)},
                "suggested_augmentations": SUGGESTED_AUGMENTATIONS["class_imbalance"],
            }
        )
    overall = summary.get("overall", {})
    weak_count = int(overall.get("localization_weak_count", 0) or 0)
    if weak_count > 0 and "localization_bias" not in existing_types:
        extras.append(
            {
                "type": "localization_bias",
                "source_issue": "localization_weak_count",
                "severity": "medium",
                "evidence": {"localization_weak_count": weak_count},
                "suggested_augmentations": SUGGESTED_AUGMENTATIONS["localization_bias"],
            }
        )
    return extras
