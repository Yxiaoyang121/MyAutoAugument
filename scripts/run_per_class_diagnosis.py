from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "20260518_tiled1024_safe_no_ok_position_per_class_diagnosis"
BASELINE_RUN_ID = "20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep"
TOP3_RUN_ID = "20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline"
COUNTERFACTUAL_RUN_ID = "20260518_tiled1024_safe_no_ok_position_counterfactual_diagnosis"

DATA_YAML = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position" / "data.yaml"
DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
BASELINE_METRICS = PROJECT_ROOT / "outputs" / "experiments" / BASELINE_RUN_ID / "reports" / "baseline_50ep_metrics.json"
BASELINE_DIAGNOSIS = PROJECT_ROOT / "outputs" / "experiments" / TOP3_RUN_ID / "diagnosis" / "diagnosis.json"
BASELINE_ERROR_INSTANCES = PROJECT_ROOT / "outputs" / "experiments" / COUNTERFACTUAL_RUN_ID / "baseline_error_instances.json"
BASELINE_PREDICTIONS = PROJECT_ROOT / "outputs" / "experiments" / COUNTERFACTUAL_RUN_ID / "baseline_predictions.json"
COUNTERFACTUAL_SUMMARY = PROJECT_ROOT / "outputs" / "experiments" / COUNTERFACTUAL_RUN_ID / "counterfactual_summary.json"
COUNTERFACTUAL_INSTANCES = PROJECT_ROOT / "outputs" / "experiments" / COUNTERFACTUAL_RUN_ID / "counterfactual_instances.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID

CANONICAL_CLASS_NAMES = {
    0: "OK2",
    1: "OK3",
    2: "加强筋打伤",
    3: "开裂",
    4: "油污",
    5: "浅划伤",
    6: "漏背锡",
    7: "碰伤",
    8: "脏污",
    9: "轮廓划伤",
    10: "锡丝残留",
    11: "锡尖",
    12: "锡膏",
}

OK_CLASS_IDS = {0, 1}
TEXTURE_CLASS_IDS = {4, 5, 7, 8, 9, 12}
LOW_SUPPORT_VAL_THRESHOLD = 10
LOW_TRAIN_SUPPORT_THRESHOLD = 100


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "policies").mkdir(parents=True, exist_ok=True)

    baseline_metrics = read_json(BASELINE_METRICS)
    baseline_diagnosis = read_json(BASELINE_DIAGNOSIS)
    baseline_errors = read_json(BASELINE_ERROR_INSTANCES)
    baseline_predictions = read_json(BASELINE_PREDICTIONS)
    counterfactual_summary = read_json(COUNTERFACTUAL_SUMMARY)
    counterfactual_instances = read_json(COUNTERFACTUAL_INSTANCES)

    train_counts = count_split_instances(DATASET_ROOT / "labels" / "train")
    val_counts = count_split_instances(DATASET_ROOT / "labels" / "val")
    diagnosis = build_per_class_diagnosis(
        baseline_metrics=baseline_metrics,
        baseline_diagnosis=baseline_diagnosis,
        baseline_errors=baseline_errors,
        baseline_predictions=baseline_predictions,
        counterfactual_summary=counterfactual_summary,
        counterfactual_instances=counterfactual_instances,
        train_counts=train_counts,
        val_counts=val_counts,
    )
    write_json(OUTPUT_DIR / "reports" / "per_class_diagnosis.json", diagnosis)
    write_per_class_report(OUTPUT_DIR / "reports" / "per_class_diagnosis_report.md", diagnosis)

    recommendations = build_classwise_recommendations(diagnosis)
    write_json(OUTPUT_DIR / "reports" / "classwise_recommendations.json", recommendations)
    write_recommendations_md(OUTPUT_DIR / "reports" / "classwise_recommendations.md", recommendations)

    policy = build_class_aware_policy(diagnosis, recommendations)
    write_json(OUTPUT_DIR / "policies" / "class_aware_mixed_policy.json", policy)
    write_policy_md(OUTPUT_DIR / "policies" / "class_aware_mixed_policy.md", policy)

    score = score_class_aware_policy(policy, diagnosis)
    write_json(OUTPUT_DIR / "policies" / "class_aware_policy_score.json", score)
    write_policy_score_md(OUTPUT_DIR / "policies" / "class_aware_policy_score.md", score)

    write_json(
        OUTPUT_DIR / "run_config.json",
        {
            "run_id": RUN_ID,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "class-aware error attribution and policy generation",
            "no_training_executed": True,
            "baseline_run": BASELINE_RUN_ID,
            "data_yaml": rel(DATA_YAML),
            "baseline_metrics": rel(BASELINE_METRICS),
            "baseline_diagnosis": rel(BASELINE_DIAGNOSIS),
            "counterfactual_summary": rel(COUNTERFACTUAL_SUMMARY),
            "counterfactual_instances": rel(COUNTERFACTUAL_INSTANCES),
        },
    )
    update_state_docs(diagnosis, policy, score)
    print(json.dumps({"run_id": RUN_ID, "final_policy_score": score["final_policy_score"], "branches": policy["branch_names"]}, ensure_ascii=False, indent=2))


def build_per_class_diagnosis(
    *,
    baseline_metrics: dict[str, Any],
    baseline_diagnosis: dict[str, Any],
    baseline_errors: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    counterfactual_summary: dict[str, Any],
    counterfactual_instances: list[dict[str, Any]],
    train_counts: dict[int, int],
    val_counts: dict[int, int],
) -> dict[str, Any]:
    yolo_by_id = {int(item["class_id"]): item for item in baseline_metrics["validation"]["per_class"]}
    diag_by_id = {int(key): item for key, item in baseline_diagnosis.get("per_class", {}).items()}
    fn_features = feature_summary_by_class(baseline_errors, status="FN")
    weak_counts = Counter(int(row["class_id"]) for row in baseline_errors if row.get("original_status") == "localization_weak")
    cf_by_id = counterfactual_summary_by_class(counterfactual_instances)
    confusion_by_id = compute_confusions(baseline_errors, baseline_predictions)

    class_ids = sorted(set(CANONICAL_CLASS_NAMES) | set(train_counts) | set(val_counts) | set(yolo_by_id) | set(diag_by_id))
    rows = []
    for class_id in class_ids:
        class_name = CANONICAL_CLASS_NAMES.get(class_id, str(class_id))
        yolo = yolo_by_id.get(class_id, {})
        diag = diag_by_id.get(class_id, {})
        train_instances = int(train_counts.get(class_id, 0))
        val_instances = int(val_counts.get(class_id, yolo.get("instances", diag.get("gt", 0)) or 0))
        tp = int(diag.get("tp", 0) or 0)
        fp = int(diag.get("fp", 0) or 0)
        fn = int(diag.get("fn", max(0, val_instances - tp)) or 0)
        precision = float(yolo.get("precision", diag.get("precision", 0.0)) or 0.0)
        recall = float(yolo.get("recall", diag.get("recall", 0.0)) or 0.0)
        ap50 = float(yolo.get("ap50", diag.get("ap50", 0.0)) or 0.0)
        ap50_95 = float(yolo.get("ap50_95", diag.get("ap50_95", 0.0)) or 0.0)
        fn_rate = fn / max(1, val_instances)
        fp_rate = fp / max(1, tp + fp)
        feature = fn_features.get(class_id, empty_feature_summary())
        cf = cf_by_id.get(class_id, empty_counterfactual_summary())
        weak = int(weak_counts.get(class_id, 0))
        support_confidence = confidence_from_instances(val_instances)
        confidence_score = support_confidence["score"]
        severity_score = compute_severity_score(
            val_instances=val_instances,
            train_instances=train_instances,
            precision=precision,
            recall=recall,
            ap50=ap50,
            fn_rate=fn_rate,
            fp_rate=fp_rate,
            low_contrast_rate=feature["low_contrast_rate"],
            dark_rate=feature["dark_rate"],
            small_object_rate=feature["small_object_rate"],
            edge_object_rate=feature["edge_object_rate"],
            weak_localization_count=weak,
        )
        attributions = infer_error_attributions(
            class_id=class_id,
            class_name=class_name,
            train_instances=train_instances,
            val_instances=val_instances,
            precision=precision,
            recall=recall,
            ap50=ap50,
            fn_rate=fn_rate,
            fp_rate=fp_rate,
            feature=feature,
            counterfactual=cf,
            weak_localization_count=weak,
            confusions=confusion_by_id.get(class_id, []),
        )
        primary_error_type = choose_primary_error_type(attributions, class_id=class_id, val_instances=val_instances, recall=recall, fp_rate=fp_rate)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "train_instances": train_instances,
                "val_instances": val_instances,
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ap50_95": ap50_95,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "fn_rate": fn_rate,
                "fp_rate": fp_rate,
                "primary_error_type": primary_error_type,
                "error_attributions": attributions,
                "diagnosis_evidence": {
                    "low_contrast_rate": feature["low_contrast_rate"],
                    "dark_rate": feature["dark_rate"],
                    "small_object_rate": feature["small_object_rate"],
                    "edge_object_rate": feature["edge_object_rate"],
                    "border_touch_rate": feature["border_touch_rate"],
                    "weak_localization_count": weak,
                    "weak_localization_rate": weak / max(1, val_instances),
                    "counterfactual_recovered_count": cf["recovered_unique_fn_count"],
                    "counterfactual_recovery_rate": cf["unique_recovery_rate"],
                    "photometric_recovered_count": cf["photometric_recovered_unique_fn_count"],
                    "photometric_recovery_rate": cf["photometric_unique_recovery_rate"],
                    "sharpen_recovered_count": cf["sharpen_recovered_unique_fn_count"],
                    "sharpen_recovery_rate": cf["sharpen_unique_recovery_rate"],
                    "scale_recovered_count": cf["scale_recovered_unique_fn_count"],
                    "scale_recovery_rate": cf["scale_unique_recovery_rate"],
                    "top_confused_target_classes": confusion_by_id.get(class_id, [])[:3],
                    "support_confidence": support_confidence["label"],
                    "support_confidence_reason": support_confidence["reason"],
                },
                "severity_score": severity_score,
                "confidence_score": confidence_score,
                "priority_score": round(severity_score * confidence_score, 6),
            }
        )
    rows.sort(key=lambda item: item["class_id"])
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "per-class error attribution and policy generation only; no training",
        "baseline_run_id": BASELINE_RUN_ID,
        "source_files": {
            "baseline_metrics": rel(BASELINE_METRICS),
            "baseline_diagnosis": rel(BASELINE_DIAGNOSIS),
            "counterfactual_summary": rel(COUNTERFACTUAL_SUMMARY),
            "counterfactual_instances": rel(COUNTERFACTUAL_INSTANCES),
        },
        "global_context": {
            "baseline_precision": baseline_metrics["validation"]["overall"]["precision"],
            "baseline_recall": baseline_metrics["validation"]["overall"]["recall"],
            "baseline_map50": baseline_metrics["validation"]["overall"]["map50"],
            "baseline_map50_95": baseline_metrics["validation"]["overall"]["map50_95"],
            "diagnosis_triggered_issues": [item.get("type") for item in baseline_diagnosis.get("issues", [])],
            "counterfactual_photometric_unique_recovery_rate": counterfactual_summary.get("diag_policy_001_support", {}).get("any_diag_policy_001_op_recovery_rate"),
        },
        "classes": rows,
        "answers": build_report_answers(rows, counterfactual_summary),
    }


def feature_summary_by_class(rows: list[dict[str, Any]], *, status: str) -> dict[int, dict[str, float]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("original_status") == status and row.get("gt_bbox") is not None:
            grouped[int(row["class_id"])].append(row)
    out: dict[int, dict[str, float]] = {}
    for class_id, items in grouped.items():
        out[class_id] = {
            "fn_count": len(items),
            "low_contrast_rate": mean_bool(items, "is_low_contrast"),
            "dark_rate": mean_bool(items, "is_dark"),
            "small_object_rate": sum(1 for item in items if item.get("area_bin") in {"tiny", "small"}) / max(1, len(items)),
            "edge_object_rate": sum(1 for item in items if item.get("tile_position") in {"edge", "corner"} or item.get("near_edge")) / max(1, len(items)),
            "border_touch_rate": mean_bool(items, "touches_tile_border"),
            "avg_brightness": mean_value(items, "brightness"),
            "avg_contrast": mean_value(items, "contrast"),
        }
    return out


def empty_feature_summary() -> dict[str, float]:
    return {
        "fn_count": 0,
        "low_contrast_rate": 0.0,
        "dark_rate": 0.0,
        "small_object_rate": 0.0,
        "edge_object_rate": 0.0,
        "border_touch_rate": 0.0,
        "avg_brightness": 0.0,
        "avg_contrast": 0.0,
    }


def counterfactual_summary_by_class(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["class_id"])][instance_key(row)].append(row)
    out: dict[int, dict[str, Any]] = {}
    for class_id, by_instance in grouped.items():
        tested = len(by_instance)
        recovered = 0
        photometric = 0
        sharpen = 0
        scale = 0
        for instance_rows in by_instance.values():
            if any(row.get("recovered") for row in instance_rows):
                recovered += 1
            if any(row.get("recovered") and is_photometric_transform(str(row.get("transform_name"))) for row in instance_rows):
                photometric += 1
            if any(row.get("recovered") and str(row.get("transform_name")) == "sharpen_mild" for row in instance_rows):
                sharpen += 1
            if any(row.get("recovered") and str(row.get("transform_name")) == "zoom_in_context" for row in instance_rows):
                scale += 1
        out[class_id] = {
            "tested_unique_fn_count": tested,
            "recovered_unique_fn_count": recovered,
            "unique_recovery_rate": recovered / max(1, tested),
            "photometric_recovered_unique_fn_count": photometric,
            "photometric_unique_recovery_rate": photometric / max(1, tested),
            "sharpen_recovered_unique_fn_count": sharpen,
            "sharpen_unique_recovery_rate": sharpen / max(1, tested),
            "scale_recovered_unique_fn_count": scale,
            "scale_unique_recovery_rate": scale / max(1, tested),
        }
    return out


def empty_counterfactual_summary() -> dict[str, Any]:
    return {
        "tested_unique_fn_count": 0,
        "recovered_unique_fn_count": 0,
        "unique_recovery_rate": 0.0,
        "photometric_recovered_unique_fn_count": 0,
        "photometric_unique_recovery_rate": 0.0,
        "sharpen_recovered_unique_fn_count": 0,
        "sharpen_unique_recovery_rate": 0.0,
        "scale_recovered_unique_fn_count": 0,
        "scale_unique_recovery_rate": 0.0,
    }


def compute_confusions(baseline_errors: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    pred_by_image = {str(Path(item["image_path"]).resolve()): item for item in predictions}
    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    for row in baseline_errors:
        if row.get("original_status") not in {"FN", "localization_weak"} or row.get("gt_bbox") is None:
            continue
        pred = pred_by_image.get(str(Path(row["image_path"]).resolve()))
        if not pred:
            continue
        labels = [int(value) for value in pred.get("pred_labels", [])]
        boxes = pred.get("pred_boxes", [])
        if not boxes:
            continue
        gt = [float(value) for value in row["gt_bbox"]]
        best_label = None
        best_iou = 0.0
        for label, box in zip(labels, boxes):
            iou = xyxy_iou(gt, [float(v) for v in box])
            if iou > best_iou:
                best_iou = iou
                best_label = label
        if best_label is not None and best_label != int(row["class_id"]) and best_iou >= 0.10:
            confusion[int(row["class_id"])][int(best_label)] += 1
    return {
        class_id: [
            {"target_class_id": target, "target_class_name": CANONICAL_CLASS_NAMES.get(target, str(target)), "count": count}
            for target, count in counter.most_common(5)
        ]
        for class_id, counter in confusion.items()
    }


def infer_error_attributions(
    *,
    class_id: int,
    class_name: str,
    train_instances: int,
    val_instances: int,
    precision: float,
    recall: float,
    ap50: float,
    fn_rate: float,
    fp_rate: float,
    feature: dict[str, float],
    counterfactual: dict[str, Any],
    weak_localization_count: int,
    confusions: list[dict[str, Any]],
) -> list[str]:
    labels: list[str] = []
    if val_instances < LOW_SUPPORT_VAL_THRESHOLD:
        labels.append("low_support_class")
    if class_id not in OK_CLASS_IDS and train_instances < LOW_TRAIN_SUPPORT_THRESHOLD:
        labels.append("class_sample_imbalance")
    if val_instances >= LOW_SUPPORT_VAL_THRESHOLD and recall < 0.65:
        labels.append("low_recall_class")
    if val_instances >= LOW_SUPPORT_VAL_THRESHOLD and (precision < 0.55 or fp_rate >= 0.45):
        labels.append("high_fp_class")
    if feature["low_contrast_rate"] >= 0.50 or counterfactual["photometric_unique_recovery_rate"] >= 0.05:
        labels.append("low_contrast_error")
    if feature["dark_rate"] >= 0.40:
        labels.append("dark_object_error")
    if class_id in TEXTURE_CLASS_IDS and (recall < 0.75 or counterfactual["sharpen_unique_recovery_rate"] >= 0.04):
        labels.append("texture_confusion")
    if weak_localization_count > 0 and weak_localization_count / max(1, val_instances) >= 0.02:
        labels.append("weak_localization")
    if feature["small_object_rate"] >= 0.60:
        labels.append("small_object_error")
    if feature["edge_object_rate"] >= 0.35 or feature["border_touch_rate"] >= 0.05:
        labels.append("tile_edge_error")
    if confusions:
        labels.append("class_confusion")
    if not labels:
        labels.append("stable_or_low_priority")
    return labels


def choose_primary_error_type(attributions: list[str], *, class_id: int, val_instances: int, recall: float, fp_rate: float) -> str:
    order = [
        "low_support_class",
        "class_sample_imbalance",
        "high_fp_class",
        "low_recall_class",
        "texture_confusion",
        "low_contrast_error",
        "dark_object_error",
        "small_object_error",
        "weak_localization",
        "tile_edge_error",
        "class_confusion",
        "stable_or_low_priority",
    ]
    if class_id in OK_CLASS_IDS and recall >= 0.95 and fp_rate < 0.15:
        return "stable_or_low_priority"
    for item in order:
        if item in attributions:
            return item
    return attributions[0]


def compute_severity_score(**kwargs: Any) -> float:
    val_instances = int(kwargs["val_instances"])
    train_instances = int(kwargs["train_instances"])
    recall = float(kwargs["recall"])
    precision = float(kwargs["precision"])
    ap50 = float(kwargs["ap50"])
    fn_rate = float(kwargs["fn_rate"])
    fp_rate = float(kwargs["fp_rate"])
    low_contrast_rate = float(kwargs["low_contrast_rate"])
    dark_rate = float(kwargs["dark_rate"])
    small_object_rate = float(kwargs["small_object_rate"])
    edge_object_rate = float(kwargs["edge_object_rate"])
    weak_count = int(kwargs["weak_localization_count"])
    recall_gap = 1.0 - recall
    precision_gap = 1.0 - precision
    ap_gap = 1.0 - ap50
    low_support_component = 0.0
    if val_instances < LOW_SUPPORT_VAL_THRESHOLD:
        low_support_component = (LOW_SUPPORT_VAL_THRESHOLD - val_instances) / LOW_SUPPORT_VAL_THRESHOLD
    imbalance_component = 0.0 if train_instances >= LOW_TRAIN_SUPPORT_THRESHOLD else (LOW_TRAIN_SUPPORT_THRESHOLD - train_instances) / LOW_TRAIN_SUPPORT_THRESHOLD
    weak_component = min(1.0, weak_count / max(1, val_instances))
    score = (
        0.24 * recall_gap
        + 0.16 * fn_rate
        + 0.14 * ap_gap
        + 0.10 * max(precision_gap, fp_rate)
        + 0.10 * max(low_contrast_rate, dark_rate)
        + 0.08 * small_object_rate
        + 0.06 * edge_object_rate
        + 0.07 * low_support_component
        + 0.05 * imbalance_component
        + 0.04 * weak_component
    )
    return round(clip01(score), 6)


def confidence_from_instances(val_instances: int) -> dict[str, Any]:
    if val_instances >= 50:
        return {"label": "high", "score": 1.0, "reason": "val instances >= 50"}
    if val_instances >= 20:
        return {"label": "medium_high", "score": 0.80, "reason": "20 <= val instances < 50"}
    if val_instances >= 10:
        return {"label": "medium", "score": 0.60, "reason": "10 <= val instances < 20"}
    if val_instances >= 5:
        return {"label": "low", "score": 0.35, "reason": "5 <= val instances < 10; unstable class-level conclusion"}
    if val_instances > 0:
        return {"label": "very_low", "score": 0.20, "reason": "val instances < 5; do not over-interpret recall"}
    return {"label": "none", "score": 0.0, "reason": "no val instances"}


def build_classwise_recommendations(diagnosis: dict[str, Any]) -> dict[str, Any]:
    recommendations = []
    for row in diagnosis["classes"]:
        recs: list[str] = []
        reasons: list[str] = []
        attrs = set(row["error_attributions"])
        if attrs & {"low_contrast_error", "dark_object_error"}:
            recs.extend(["clahe", "contrast", "gamma", "brightness", "sharpen_mild"])
            reasons.append("low_contrast_error / dark_object_error")
        if attrs & {"low_support_class", "class_sample_imbalance"} and row["class_id"] not in OK_CLASS_IDS:
            recs.extend(["class_balanced_copy_paste", "targeted_augmentation", "oversampling", "collect_more_samples_warning"])
            reasons.append("low_support_class / class_sample_imbalance")
        if "texture_confusion" in attrs:
            recs.extend(["sharpen", "local_contrast", "mild_noise", "hard_negative_mining"])
            reasons.append("texture_confusion")
        if "weak_localization" in attrs:
            recs.extend(["mild_translate", "mild_scale", "avoid_strong_rotate", "avoid_perspective"])
            reasons.append("weak_localization")
        if "small_object_error" in attrs:
            recs.extend(["copy_paste_prefer_small", "scale_aware_augmentation", "keep_high_resolution_tiling"])
            reasons.append("small_object_error")
        if "tile_edge_error" in attrs:
            recs.extend(["stricter_tiling", "larger_tile_size_candidate", "avoid_border_truncated_bbox"])
            reasons.append("tile_edge_error")
        recommendations.append(
            {
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "primary_error_type": row["primary_error_type"],
                "error_attributions": row["error_attributions"],
                "priority_score": row["priority_score"],
                "confidence_score": row["confidence_score"],
                "recommendations": unique(recs),
                "reasons": unique(reasons),
                "evidence": row["diagnosis_evidence"],
            }
        )
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "recommendations": recommendations,
    }


def build_class_aware_policy(diagnosis: dict[str, Any], recommendations: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in diagnosis["classes"] if row["class_id"] not in OK_CLASS_IDS]
    by_id = {row["class_id"]: row for row in rows}
    branches = []

    photometric_ids = [
        row["class_id"]
        for row in rows
        if {"low_contrast_error", "dark_object_error"} & set(row["error_attributions"])
        and (row["priority_score"] >= 0.10 or row["diagnosis_evidence"]["photometric_recovered_count"] > 0)
    ]
    copy_paste_ids = [
        row["class_id"]
        for row in rows
        if {"low_support_class", "class_sample_imbalance"} & set(row["error_attributions"])
        and (row["priority_score"] >= 0.08 or row["val_instances"] < LOW_SUPPORT_VAL_THRESHOLD)
    ]
    texture_ids = [
        row["class_id"]
        for row in rows
        if "texture_confusion" in row["error_attributions"] and row["priority_score"] >= 0.08
    ]
    localization_ids = [
        row["class_id"]
        for row in rows
        if "weak_localization" in row["error_attributions"] and row["priority_score"] >= 0.08
    ]

    if photometric_ids:
        branches.append(make_branch("photometric_branch", photometric_ids, by_id, ["clahe", "contrast", "gamma", "brightness", "sharpen_mild"], "low_contrast_error / dark_object_error"))
    if copy_paste_ids:
        branches.append(make_branch("copy_paste_branch", copy_paste_ids, by_id, ["class_balanced_copy_paste"], "low_support_class / class_sample_imbalance"))
    if texture_ids:
        branches.append(make_branch("texture_branch", texture_ids, by_id, ["sharpen", "local_contrast", "mild_noise"], "texture_confusion / weak texture boundary"))
    if localization_ids:
        branches.append(make_branch("localization_branch", localization_ids, by_id, ["mild_translate", "mild_scale"], "weak_localization"))

    return {
        "policy_id": "class_aware_policy_001",
        "type": "class_aware_mixed_policy",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "baseline_diagnosis": rel(BASELINE_DIAGNOSIS),
            "counterfactual_diagnosis": rel(COUNTERFACTUAL_SUMMARY),
            "per_class_diagnosis": f"outputs/experiments/{RUN_ID}/reports/per_class_diagnosis.json",
        },
        "selection_rules": [
            "Do not apply all augmentations globally.",
            "copy_paste is restricted to low-support or class-imbalance classes.",
            "photometric operations are assigned only to classes with low-contrast/dark evidence or counterfactual recovery.",
            "prob and strength are dynamic functions of severity_score and confidence_score.",
        ],
        "branch_names": [branch["branch_name"] for branch in branches],
        "branches": branches,
    }


def make_branch(branch_name: str, class_ids: list[int], by_id: dict[int, dict[str, Any]], ops: list[str], reason: str) -> dict[str, Any]:
    target_rows = [by_id[class_id] for class_id in sorted(set(class_ids))]
    avg_severity = sum(row["severity_score"] for row in target_rows) / max(1, len(target_rows))
    avg_confidence = sum(row["confidence_score"] for row in target_rows) / max(1, len(target_rows))
    avg_priority = sum(row["priority_score"] for row in target_rows) / max(1, len(target_rows))
    op_specs = []
    for op in ops:
        prob = branch_prob(op, avg_priority, avg_confidence)
        strength = branch_strength(op, avg_severity, avg_confidence)
        op_specs.append(
            {
                "name": op,
                "prob": prob,
                "strength": strength,
                "params": operation_params(op, target_rows),
                "prob_formula": "clip(base_prob + priority_gain*avg_priority + confidence_gain*avg_confidence)",
                "strength_formula": "clip(base_strength + severity_gain*avg_severity + confidence_gain*avg_confidence)",
            }
        )
    return {
        "branch_name": branch_name,
        "target_class_ids": [row["class_id"] for row in target_rows],
        "target_classes": [row["class_name"] for row in target_rows],
        "ops": op_specs,
        "reason": reason,
        "avg_severity_score": round(avg_severity, 6),
        "avg_confidence_score": round(avg_confidence, 6),
        "avg_priority_score": round(avg_priority, 6),
        "target_class_evidence": [
            {
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "primary_error_type": row["primary_error_type"],
                "error_attributions": row["error_attributions"],
                "severity_score": row["severity_score"],
                "confidence_score": row["confidence_score"],
                "priority_score": row["priority_score"],
                "evidence": row["diagnosis_evidence"],
            }
            for row in target_rows
        ],
    }


def branch_prob(op: str, avg_priority: float, avg_confidence: float) -> float:
    base = {
        "class_balanced_copy_paste": 0.18,
        "mild_translate": 0.12,
        "mild_scale": 0.14,
        "mild_noise": 0.12,
    }.get(op, 0.16)
    return round(clip(base + 0.45 * avg_priority + 0.10 * avg_confidence, 0.08, 0.65), 3)


def branch_strength(op: str, avg_severity: float, avg_confidence: float) -> float:
    base = {
        "class_balanced_copy_paste": 0.35,
        "mild_translate": 0.20,
        "mild_scale": 0.25,
        "mild_noise": 0.18,
    }.get(op, 0.25)
    return round(clip(base + 0.45 * avg_severity + 0.08 * avg_confidence, 0.10, 0.75), 3)


def operation_params(op: str, target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_ids = [row["class_id"] for row in target_rows]
    if op == "clahe":
        return {"clip_limit_range": [2.0, 3.0], "tile_grid_size": 8, "target_class_ids": class_ids}
    if op == "contrast":
        return {"alpha_range": [1.10, 1.30], "target_class_ids": class_ids}
    if op == "gamma":
        return {"gamma_range": [0.75, 0.90], "target_class_ids": class_ids}
    if op == "brightness":
        return {"beta_range": [10, 30], "target_class_ids": class_ids}
    if op in {"sharpen", "sharpen_mild"}:
        return {"amount_range": [0.50, 0.90], "sigma": 1.0, "target_class_ids": class_ids}
    if op == "class_balanced_copy_paste":
        return {"target_class_ids": class_ids, "prefer_small": True, "class_balanced": True, "max_paste_count": 2, "max_overlap": 0.20}
    if op == "local_contrast":
        return {"clip_limit_range": [1.5, 2.5], "target_class_ids": class_ids}
    if op == "mild_noise":
        return {"std_range": [0.01, 0.03], "target_class_ids": class_ids}
    if op == "mild_translate":
        return {"max_translate": 0.04, "target_class_ids": class_ids}
    if op == "mild_scale":
        return {"scale_range": [0.92, 1.08], "target_class_ids": class_ids}
    return {"target_class_ids": class_ids}


def score_class_aware_policy(policy: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    defect_rows = [row for row in diagnosis["classes"] if row["class_id"] not in OK_CLASS_IDS]
    total_priority = sum(row["priority_score"] for row in defect_rows)
    covered_ids = {class_id for branch in policy["branches"] for class_id in branch["target_class_ids"]}
    covered_rows = [row for row in defect_rows if row["class_id"] in covered_ids]
    covered_priority = sum(row["priority_score"] for row in covered_rows)
    diagnosis_alignment_score = sum(branch["avg_priority_score"] for branch in policy["branches"]) / max(1, len(policy["branches"]))
    class_priority_coverage = covered_priority / max(1e-9, total_priority)
    recall_gap_coverage = sum((1.0 - row["recall"]) * row["priority_score"] for row in covered_rows) / max(1e-9, total_priority)
    cf_signal = sum(row["diagnosis_evidence"]["counterfactual_recovery_rate"] * row["priority_score"] for row in covered_rows) / max(1e-9, total_priority)
    expected_recall_gain_score = clip01(0.75 * recall_gap_coverage + 0.25 * cf_signal)
    copy_paste_branch = next((branch for branch in policy["branches"] if branch["branch_name"] == "copy_paste_branch"), None)
    risk_penalty = 0.04 * len(policy["branches"])
    if copy_paste_branch:
        risk_penalty += 0.05
        # Targeted copy-paste has lower risk than global copy-paste, but still
        # changes bbox count and object placement.
        risk_penalty -= 0.02
    if any("OK" in name for branch in policy["branches"] for name in branch["target_classes"]):
        risk_penalty += 0.20
    risk_penalty = round(clip01(risk_penalty), 6)
    safety_score = round(clip01(1.0 - risk_penalty), 6)
    final_policy_score = round(
        clip01(
            0.35 * diagnosis_alignment_score
            + 0.25 * class_priority_coverage
            + 0.25 * expected_recall_gain_score
            + 0.15 * safety_score
        ),
        6,
    )
    return {
        "policy_id": policy["policy_id"],
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "diagnosis_alignment_score": round(diagnosis_alignment_score, 6),
        "class_priority_coverage": round(class_priority_coverage, 6),
        "expected_recall_gain_score": round(expected_recall_gain_score, 6),
        "risk_penalty": risk_penalty,
        "safety_score": safety_score,
        "final_policy_score": final_policy_score,
        "formula": "0.35*diagnosis_alignment + 0.25*priority_coverage + 0.25*expected_recall_gain + 0.15*safety_score",
        "covered_class_ids": sorted(covered_ids),
        "covered_classes": [CANONICAL_CLASS_NAMES.get(class_id, str(class_id)) for class_id in sorted(covered_ids)],
        "notes": [
            "This is a diagnosis/proxy score, not a training result.",
            "The policy must be validated by short-training before being treated as final.",
        ],
    }


def build_report_answers(rows: list[dict[str, Any]], counterfactual_summary: dict[str, Any]) -> dict[str, Any]:
    photometric = [row["class_name"] for row in rows if row["class_id"] not in OK_CLASS_IDS and {"low_contrast_error", "dark_object_error"} & set(row["error_attributions"])]
    copy_paste = [row["class_name"] for row in rows if row["class_id"] not in OK_CLASS_IDS and {"low_support_class", "class_sample_imbalance"} & set(row["error_attributions"])]
    texture = [row["class_name"] for row in rows if row["class_id"] not in OK_CLASS_IDS and "texture_confusion" in row["error_attributions"]]
    low_support = [row["class_name"] for row in rows if "low_support_class" in row["error_attributions"]]
    return {
        "is_low_contrast_still_largest_problem": False,
        "low_contrast_context": (
            "Low contrast is still real, but counterfactual photometric recovery is partial "
            f"({counterfactual_summary.get('diag_policy_001_support', {}).get('any_diag_policy_001_op_recovery_rate', 0.0):.4f}); "
            "class imbalance, texture confusion, and localization/size errors also matter."
        ),
        "photometric_suitable_classes": photometric,
        "copy_paste_or_class_balance_classes": copy_paste,
        "texture_confusion_classes": texture,
        "low_support_classes": low_support,
        "diag_policy_001_limitation": "It applies one global photometric branch and cannot target low-support, texture, localization, or class-balance failures.",
        "class_aware_expected_improvement": "It covers multiple error causes with class-specific branches and keeps copy_paste targeted to sparse/imbalanced classes.",
        "recommend_short_training": True,
    }


def write_per_class_report(path: Path, payload: dict[str, Any]) -> None:
    answers = payload["answers"]
    lines = [
        "# Per-Class Diagnosis Report",
        "",
        "- Scope: class-aware error attribution and policy generation only.",
        "- No YOLO training, no 50 epoch run, and no top3 short-training were executed.",
        f"- Baseline: `{BASELINE_RUN_ID}`",
        f"- Existing diagnosis: `{rel(BASELINE_DIAGNOSIS)}`",
        f"- Counterfactual diagnosis: `{rel(COUNTERFACTUAL_SUMMARY)}`",
        "",
        "## Required Answers",
        "",
        f"1. Current largest issue: {answers['low_contrast_context']}",
        f"2. Photometric-suitable classes: `{', '.join(answers['photometric_suitable_classes']) or '-'}`",
        f"3. copy-paste / class-balance classes: `{', '.join(answers['copy_paste_or_class_balance_classes']) or '-'}`",
        f"4. Texture-confusion classes: `{', '.join(answers['texture_confusion_classes']) or '-'}`",
        f"5. Low-support classes: `{', '.join(answers['low_support_classes']) or '-'}`",
        f"6. diag_policy_001 limitation: {answers['diag_policy_001_limitation']}",
        f"7. class-aware mixed policy expected improvement: {answers['class_aware_expected_improvement']}",
        f"8. Recommend top1 short-training next: `{answers['recommend_short_training']}`",
        "",
        "## Per-Class Attribution",
        "",
        "| class | train | val | P | R | AP50 | AP50-95 | TP | FP | FN | primary_error | attributions | severity | confidence | priority |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|",
    ]
    for row in payload["classes"]:
        lines.append(
            f"| {row['class_name']} | {row['train_instances']} | {row['val_instances']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['ap50']:.3f} | {row['ap50_95']:.3f} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | `{row['primary_error_type']}` | "
            f"{', '.join(row['error_attributions'])} | {row['severity_score']:.3f} | "
            f"{row['confidence_score']:.3f} | {row['priority_score']:.3f} |"
        )
    lines.extend(["", "## Evidence Details", ""])
    for row in payload["classes"]:
        ev = row["diagnosis_evidence"]
        lines.append(
            f"- `{row['class_name']}`: low_contrast={ev['low_contrast_rate']:.3f}, dark={ev['dark_rate']:.3f}, "
            f"small={ev['small_object_rate']:.3f}, edge={ev['edge_object_rate']:.3f}, border_touch={ev['border_touch_rate']:.3f}, "
            f"weak_loc={ev['weak_localization_count']}, cf_recovered={ev['counterfactual_recovered_count']}, "
            f"cf_rate={ev['counterfactual_recovery_rate']:.3f}, support={ev['support_confidence']}"
        )
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendations_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Classwise Recommendations",
        "",
        "| class | primary_error | priority | recommendations | reasons |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["recommendations"]:
        lines.append(
            f"| {row['class_name']} | `{row['primary_error_type']}` | {row['priority_score']:.3f} | "
            f"{', '.join(row['recommendations']) or '-'} | {', '.join(row['reasons']) or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_policy_md(path: Path, policy: dict[str, Any]) -> None:
    lines = [
        "# Class-Aware Mixed Policy",
        "",
        f"- Policy ID: `{policy['policy_id']}`",
        f"- Type: `{policy['type']}`",
        "- Scope: policy generation only; no augmented dataset was built and no training was run.",
        "",
        "## Branches",
        "",
    ]
    for branch in policy["branches"]:
        lines.extend(
            [
                f"### {branch['branch_name']}",
                "",
                f"- Target classes: `{', '.join(branch['target_classes'])}`",
                f"- Reason: {branch['reason']}",
                f"- Avg severity/confidence/priority: `{branch['avg_severity_score']:.3f}` / `{branch['avg_confidence_score']:.3f}` / `{branch['avg_priority_score']:.3f}`",
                "",
                "| operation | prob | strength | params |",
                "|---|---:|---:|---|",
            ]
        )
        for op in branch["ops"]:
            lines.append(f"| `{op['name']}` | {op['prob']:.3f} | {op['strength']:.3f} | `{json.dumps(op['params'], ensure_ascii=False)}` |")
        lines.extend(["", "Target evidence:", ""])
        for ev in branch["target_class_evidence"]:
            lines.append(
                f"- `{ev['class_name']}`: primary=`{ev['primary_error_type']}`, priority=`{ev['priority_score']:.3f}`, "
                f"attributions=`{', '.join(ev['error_attributions'])}`"
            )
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_policy_score_md(path: Path, score: dict[str, Any]) -> None:
    lines = [
        "# Class-Aware Policy Score",
        "",
        f"- Policy ID: `{score['policy_id']}`",
        f"- diagnosis_alignment_score: `{score['diagnosis_alignment_score']:.6f}`",
        f"- class_priority_coverage: `{score['class_priority_coverage']:.6f}`",
        f"- expected_recall_gain_score: `{score['expected_recall_gain_score']:.6f}`",
        f"- risk_penalty: `{score['risk_penalty']:.6f}`",
        f"- safety_score: `{score['safety_score']:.6f}`",
        f"- final_policy_score: `{score['final_policy_score']:.6f}`",
        f"- Formula: `{score['formula']}`",
        f"- Covered classes: `{', '.join(score['covered_classes'])}`",
        "",
        "This score is diagnosis/proxy evidence only. It is not a model metric.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_state_docs(diagnosis: dict[str, Any], policy: dict[str, Any], score: dict[str, Any]) -> None:
    block = "\n".join(
        [
            "<!-- CLASS_AWARE_DIAGNOSIS_START -->",
            "## Class-Aware Per-Class Diagnosis and Policy Generation",
            "",
            f"- Run ID: `{RUN_ID}`",
            "- Scope: upgraded diagnosis and policy generation only; no training, no 50 epoch run, no short-training.",
            "- Method upgrade: global policy selection -> class-aware error attribution policy generation.",
            "- Inputs: baseline 50 epoch metrics, baseline diagnosis, and counterfactual diagnosis.",
            f"- Policy generated: `{policy['policy_id']}` with branches `{', '.join(policy['branch_names'])}`",
            f"- final_policy_score: `{score['final_policy_score']:.6f}`",
            "- Next step: run short-training for `class_aware_policy_001` before any formal 50 epoch rerun.",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/per_class_diagnosis_report.md`",
            f"- Policy: `outputs/experiments/{RUN_ID}/policies/class_aware_mixed_policy.json`",
            f"- Score: `outputs/experiments/{RUN_ID}/policies/class_aware_policy_score.json`",
            "<!-- CLASS_AWARE_DIAGNOSIS_END -->",
        ]
    )
    for rel_path in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
        update_marked_block(PROJECT_ROOT / rel_path, "CLASS_AWARE_DIAGNOSIS", block)


def count_split_instances(labels_dir: Path) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for path in labels_dir.rglob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                counts[int(float(parts[0]))] += 1
    return dict(counts)


def mean_bool(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if bool(item.get(key))) / len(items)


def mean_value(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def instance_key(row: dict[str, Any]) -> str:
    return f"{row.get('image_path')}::{row.get('gt_index')}"


def is_photometric_transform(name: str) -> bool:
    return name.startswith(("clahe", "contrast_up", "gamma_brighten", "brightness_up", "combined_photometric"))


def xyxy_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-7, area_a + area_b - inter)


def unique(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def clip01(value: float) -> float:
    return clip(value, 0.0, 1.0)


def clip(value: float, low: float, high: float) -> float:
    if math.isnan(value):
        return low
    return max(low, min(high, float(value)))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_marked_block(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    if start in text and end in text:
        text = re.sub(re.escape(start) + r".*?" + re.escape(end), block, text, flags=re.DOTALL)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
