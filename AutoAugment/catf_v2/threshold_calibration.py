from __future__ import annotations

from pathlib import Path
from typing import Any

import json


THRESHOLDS = [round(0.10 + 0.05 * index, 2) for index in range(13)]


class ThresholdCalibrationAnalyzer:
    """Per-class threshold analysis for CATF-v2 reports.

    The analyzer is report-only in this version. It does not change training or
    validation metrics.
    """

    def __init__(self, *, default_threshold: float = 0.25) -> None:
        self.default_threshold = float(default_threshold)

    def analyze(self, per_class_diagnosis: dict[str, Any], issue_attribution: dict[str, Any] | None = None) -> dict[str, Any]:
        issue_attribution = issue_attribution or {}
        rows: dict[str, Any] = {}
        for class_id, row in (per_class_diagnosis.get("classes") or {}).items():
            attr = (issue_attribution.get("classes") or {}).get(str(class_id), {})
            high_fp = bool(row.get("high_fp") or attr.get("dominant_issue") == "high_fp")
            low_recall = bool(row.get("low_recall") or attr.get("dominant_issue") in {"low_recall", "low_contrast_fn"})
            if high_fp:
                recommended = min(0.70, self.default_threshold + 0.15)
                reason = "high_fp_raise_threshold"
            elif low_recall and not high_fp:
                recommended = max(0.10, self.default_threshold - 0.10)
                reason = "low_recall_lower_threshold"
            else:
                recommended = self.default_threshold
                reason = "keep_default"
            rows[str(class_id)] = {
                "class_id": int(row["class_id"]),
                "class_name": row.get("class_name", str(class_id)),
                "default_threshold": self.default_threshold,
                "recommended_threshold": round(float(recommended), 2),
                "threshold_range": THRESHOLDS,
                "reason": reason,
                "analysis_only": True,
            }
        return {
            "analysis_only": True,
            "threshold_range": THRESHOLDS,
            "classes": rows,
        }

    def write(self, output_json: str | Path, payload: dict[str, Any]) -> None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        path.with_suffix(".md").write_text(build_threshold_markdown(payload), encoding="utf-8")


def build_threshold_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CATF-v2 Threshold Calibration",
        "",
        "- Status: analysis only; training and validation metrics are unchanged.",
        "",
        "| class_id | class_name | default | recommended | reason |",
        "|---:|---|---:|---:|---|",
    ]
    for class_id, item in sorted((payload.get("classes") or {}).items(), key=lambda pair: int(pair[0])):
        lines.append(
            f"| {class_id} | {item.get('class_name')} | {float(item.get('default_threshold', 0.25)):.2f} | "
            f"{float(item.get('recommended_threshold', 0.25)):.2f} | {item.get('reason')} |"
        )
    return "\n".join(lines) + "\n"
