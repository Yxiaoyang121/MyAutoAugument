from __future__ import annotations

from pathlib import Path
from typing import Any

from AutoAugment.diagnostic_pipeline.common import write_json, write_markdown


def write_metric_consistency_audit(
    *,
    output_dir: str | Path,
    baseline_record: dict[str, Any],
    prediction_record: dict[str, Any],
    diagnosis: dict[str, Any],
    conf: float,
    iou: float,
    match_iou: float,
) -> dict[str, Any]:
    """Audit why YOLO val metrics and diagnosis TP/FP/FN can differ."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    yolo_metrics = dict(baseline_record.get("metrics", {}) or {})
    global_metrics = dict(diagnosis.get("global", {}) or {})
    tp = int(global_metrics.get("tp", 0) or 0)
    fp = int(global_metrics.get("fp", 0) or 0)
    fn = int(global_metrics.get("fn", 0) or 0)
    diagnosis_precision = tp / max(1, tp + fp)
    diagnosis_recall = tp / max(1, tp + fn + int(global_metrics.get("localization_weak", 0) or 0))
    yolo_precision = _float_or_none(yolo_metrics.get("precision"))
    yolo_recall = _float_or_none(yolo_metrics.get("recall"))
    differences = {
        "precision_abs_delta": abs(yolo_precision - diagnosis_precision) if yolo_precision is not None else None,
        "recall_abs_delta": abs(yolo_recall - diagnosis_recall) if yolo_recall is not None else None,
    }
    likely_causes = [
        "YOLO val reports aggregate metrics from Ultralytics validation, while diagnosis replays saved predict labels.",
        f"Diagnosis uses a fixed predict confidence threshold conf={conf} and NMS IoU={iou}.",
        f"Diagnosis TP/FP/FN uses a single greedy class-aware match_iou={match_iou}.",
        "mAP50/mAP50-95 are area-under-curve metrics across confidence thresholds, so they are not expected to equal single-threshold TP/FP/FN.",
        "localization_weak detections are counted outside TP/FN in diagnosis global metrics and can change recall denominators.",
    ]
    payload = {
        "stage": "metric_consistency_audit",
        "status": "completed",
        "yolo_val_metrics": yolo_metrics,
        "diagnosis_global": global_metrics,
        "diagnosis_recomputed": {
            "precision": diagnosis_precision,
            "recall": diagnosis_recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "differences": differences,
        "prediction_record": prediction_record,
        "likely_causes": likely_causes,
        "conclusion": "Differences are expected unless YOLO val and diagnosis use identical prediction files, thresholds, and matching rules.",
    }
    write_json(output / "metric_consistency_audit.json", payload)
    write_markdown(
        output / "metric_consistency_audit.md",
        [
            "# Metric Consistency Audit",
            "",
            f"- YOLO precision: {yolo_precision}",
            f"- YOLO recall: {yolo_recall}",
            f"- Diagnosis precision: {diagnosis_precision:.4f}",
            f"- Diagnosis recall: {diagnosis_recall:.4f}",
            f"- TP/FP/FN: {tp}/{fp}/{fn}",
            f"- Precision delta: {differences['precision_abs_delta']}",
            f"- Recall delta: {differences['recall_abs_delta']}",
            "",
            "## Conclusion",
            payload["conclusion"],
            "",
            "## Likely Causes",
            *[f"- {item}" for item in likely_causes],
        ],
    )
    return payload


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
