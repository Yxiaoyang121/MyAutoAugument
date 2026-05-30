from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from AutoAugment.catf_v2.issue_attribution import attribute_class_issues
from AutoAugment.catf_v2.per_class_diagnosis import build_per_class_diagnosis
from AutoAugment.catf_v2.policy_matrix import (
    ClassAwarePolicyMatrix,
    active_class_ids,
    frozen_class_ids,
)
from AutoAugment.catf_v2.threshold_calibration import ThresholdCalibrationAnalyzer


class ClassAwareCATFController:
    """CATF-v2 class-aware feedback controller."""

    def __init__(
        self,
        policy_matrix: dict[str, Any],
        *,
        history_dir: str | Path,
        class_names: dict[int, str],
        train_instances: dict[int, int] | None = None,
        top_k_active_classes: int = 3,
        top_m_ops_per_class: int = 2,
        freeze_epoch: int = 40,
        threshold_calibration_report: bool = True,
    ) -> None:
        self.history_dir = Path(history_dir)
        self.catf_dir = self.history_dir / "catf_v2"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.catf_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = dict(class_names)
        self.train_instances = dict(train_instances or {})
        self.policy = ClassAwarePolicyMatrix(
            policy_matrix,
            top_k=top_k_active_classes,
            top_m=top_m_ops_per_class,
            freeze_epoch=freeze_epoch,
        )
        self.history: list[dict[str, Any]] = []
        self.class_policy_history: list[dict[str, Any]] = []
        self.threshold_calibration_report = bool(threshold_calibration_report)

    @property
    def policy_matrix(self) -> dict[str, Any]:
        return deepcopy(self.policy.matrix)

    def update(
        self,
        diagnosis: dict[str, Any],
        *,
        epoch: int,
        metrics: dict[str, Any],
        reference_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        before = deepcopy(self.policy.matrix)
        per_class = build_per_class_diagnosis(
            diagnosis,
            class_names=self.class_names,
            train_instances=self.train_instances,
            epoch=epoch,
        )
        attribution = attribute_class_issues(per_class)
        sample_weight_map = build_sample_weight_map(per_class, attribution)
        write_json(self._root_file("per_class_diagnosis", epoch), per_class)
        write_json(self._catf_file("per_class_diagnosis", epoch), per_class)
        write_json(self._root_file("issue_attribution", epoch), attribution)
        write_json(self._catf_file("issue_attribution", epoch), attribution)
        write_json(self._root_file("policy_matrix", epoch, suffix="before"), before)
        write_json(self._catf_file("policy_matrix", epoch, suffix="before"), before)

        new_matrix = self.policy.update(
            epoch=epoch,
            per_class_diagnosis=per_class,
            issue_attribution=attribution,
            metrics=metrics,
            reference_metrics=reference_metrics,
        )

        write_json(self._root_file("policy_matrix", epoch, suffix="after"), new_matrix)
        write_json(self._catf_file("policy_matrix", epoch, suffix="after"), new_matrix)
        write_json(self._root_file("sample_weight_map", epoch), sample_weight_map)
        write_json(self._catf_file("sample_weight_map", epoch), sample_weight_map)
        if self.threshold_calibration_report:
            threshold_payload = ThresholdCalibrationAnalyzer().analyze(per_class, attribution)
            ThresholdCalibrationAnalyzer().write(self.history_dir / "threshold_calibration.json", threshold_payload)
            ThresholdCalibrationAnalyzer().write(self.catf_dir / "threshold_calibration.json", threshold_payload)

        record = deepcopy(self.policy.history[-1])
        record["diagnosis_summary"] = {
            "per_class": per_class.get("summary", {}),
            "issue_attribution": attribution.get("summary", {}),
        }
        record["per_class_diagnosis_path"] = str(self._root_file("per_class_diagnosis", epoch))
        record["issue_attribution_path"] = str(self._root_file("issue_attribution", epoch))
        record["sample_weight_map_path"] = str(self._root_file("sample_weight_map", epoch))
        record["active_classes"] = active_class_ids(new_matrix)
        record["frozen_classes"] = frozen_class_ids(new_matrix)
        record["high_fp_guarded_classes"] = high_fp_guarded_class_ids(new_matrix)
        record["low_contrast_classes"] = class_ids_for_issue(attribution, "low_contrast_fn")
        record["texture_classes"] = class_ids_for_issue(attribution, "texture_boundary_weak") + class_ids_for_issue(attribution, "weak_localization")
        record["low_support_classes"] = class_ids_for_issue(attribution, "low_support")
        self.history.append(record)
        for action in record.get("class_actions", []):
            self.class_policy_history.append(
                {
                    "epoch": int(epoch),
                    "class_id": int(action.get("class_id")),
                    "action": action.get("action"),
                    "adjustment_count": len(action.get("adjustments", [])),
                    "before_status": (action.get("before") or {}).get("status"),
                    "after_status": (action.get("after") or {}).get("status"),
                    "dominant_issue": (action.get("after") or {}).get("dominant_issue"),
                }
            )
        self._write_histories(new_matrix)
        return deepcopy(new_matrix)

    def summary(self) -> dict[str, Any]:
        matrix = self.policy.matrix
        return {
            "active_classes": active_class_ids(matrix),
            "frozen_classes": frozen_class_ids(matrix),
            "high_fp_guarded_classes": high_fp_guarded_class_ids(matrix),
            "class_policy_history_count": len(self.class_policy_history),
            "history_count": len(self.history),
        }

    def _write_histories(self, latest_policy: dict[str, Any]) -> None:
        write_json(self.history_dir / "policy_history.json", {"history": self.history, "latest_policy": latest_policy})
        write_json(self.history_dir / "class_policy_history.json", {"history": self.class_policy_history})
        write_json(self.catf_dir / "policy_history.json", {"history": self.history, "latest_policy": latest_policy})
        write_json(self.catf_dir / "class_policy_history.json", {"history": self.class_policy_history})
        write_policy_history_md(self.history_dir / "policy_history.md", self.history)
        write_class_history_csv(self.history_dir / "class_policy_history.csv", self.class_policy_history)

    def _root_file(self, stem: str, epoch: int, *, suffix: str | None = None) -> Path:
        end = f"_{suffix}" if suffix else ""
        return self.history_dir / f"{stem}_epoch_{int(epoch)}{end}.json"

    def _catf_file(self, stem: str, epoch: int, *, suffix: str | None = None) -> Path:
        end = f"_{suffix}" if suffix else ""
        return self.catf_dir / f"{stem}_epoch_{int(epoch)}{end}.json"


def build_sample_weight_map(per_class: dict[str, Any], attribution: dict[str, Any]) -> dict[str, Any]:
    class_weights: dict[str, Any] = {}
    for class_id, row in (per_class.get("classes") or {}).items():
        attr = (attribution.get("classes") or {}).get(str(class_id), {})
        weight = 1.0
        reasons = []
        if row.get("low_support"):
            weight += 0.2
            reasons.append("low_support")
        if row.get("low_recall"):
            weight += 0.15
            reasons.append("low_recall")
        if row.get("high_fp"):
            weight = min(weight, 1.0)
            reasons.append("high_fp_no_oversampling")
        if row.get("stable_class"):
            weight = 1.0
            reasons.append("stable_class")
        class_weights[str(class_id)] = {
            "class_id": int(row["class_id"]),
            "class_name": row.get("class_name", str(class_id)),
            "weight": round(min(1.5, weight), 4),
            "reasons": reasons,
            "oversampling_candidate": bool(attr.get("oversampling_candidate", False)),
            "copy_paste_candidate": bool(attr.get("copy_paste_candidate", False)),
            "copy_paste_status": "pending_object_bank_design",
        }
    return {
        "class_weights": class_weights,
        "image_weights": {},
        "weighted_sampler_integration": "pending",
        "rules": {
            "low_support": "+0.2",
            "low_recall": "+0.15",
            "high_fp": "no increase",
            "stable_class": "1.0",
            "max_weight": 1.5,
        },
    }


def high_fp_guarded_class_ids(matrix: dict[str, Any]) -> list[int]:
    out = []
    for class_id, item in sorted((matrix.get("classes") or {}).items(), key=lambda pair: int(pair[0])):
        guards = item.get("guards", {}) or {}
        if guards.get("high_fp_guarded"):
            out.append(int(class_id))
    return out


def class_ids_for_issue(attribution: dict[str, Any], issue: str) -> list[int]:
    return [
        int(item["class_id"])
        for item in (attribution.get("classes") or {}).values()
        if item.get("dominant_issue") == issue or issue in (item.get("secondary_issues") or [])
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_policy_history_md(path: Path, history: list[dict[str, Any]]) -> None:
    lines = [
        "# CATF-v2 Class-Aware Policy History",
        "",
        "| epoch | action | active_classes | frozen_classes | high_fp_guarded | adjustments |",
        "|---:|---|---|---|---|---:|",
    ]
    for record in history:
        lines.append(
            f"| {record.get('epoch')} | {record.get('action')} | {record.get('active_classes', [])} | "
            f"{record.get('frozen_classes', [])} | {record.get('high_fp_guarded_classes', [])} | {len(record.get('adjustments', []))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_class_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "class_id", "action", "adjustment_count", "before_status", "after_status", "dominant_issue"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)
