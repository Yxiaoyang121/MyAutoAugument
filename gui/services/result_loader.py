from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class ResultLoader:
    """Load experiment output files without letting partial JSON crash the GUI."""

    FILES = {
        "summary": "summary.json",
        "trial_record": "trial_record.json",
        "diagnosis": "diagnosis.json",
        "policy": "policy.json",
        "metrics": "metrics.json",
    }

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self.mock_output_dir = (
            self.project_root / "gui" / "mock_outputs" / "demo_experiment"
        )

    def load(self, output_dir: str | Path) -> dict[str, Any]:
        base = Path(output_dir)
        if not base.is_absolute():
            base = self.project_root / base

        result: dict[str, Any] = {
            "output_dir": str(base),
            "missing_files": [],
            "read_errors": [],
        }
        for key, filename in self.FILES.items():
            path = base / filename
            if path.exists():
                result[key] = self._read_json(path, result["read_errors"])
            else:
                result[key] = {} if key != "trial_record" else []
                result["missing_files"].append(filename)

        result["trials_json"] = self._read_json(base / "trials.json", result["read_errors"]) if (base / "trials.json").exists() else []
        result["best_policy"] = self._read_json(base / "best_policy.json", result["read_errors"]) if (base / "best_policy.json").exists() else {}
        result["run_config"] = self._read_json(base / "run_config.json", result["read_errors"]) if (base / "run_config.json").exists() else {}
        result["stage_config"] = self._read_json(base / "stage_config.json", result["read_errors"]) if (base / "stage_config.json").exists() else {}
        result["trials_csv"] = self._read_trials_csv(base / "trials.csv", result["read_errors"])
        self._normalize(result)
        return result

    def load_mock(self) -> dict[str, Any]:
        return self.load(self.mock_output_dir)

    def has_result_files(self, output_dir: str | Path) -> bool:
        base = Path(output_dir)
        if not base.is_absolute():
            base = self.project_root / base
        names = list(self.FILES.values()) + ["trials.json", "trials.csv", "best_policy.json", "run_config.json"]
        return any((base / filename).exists() for filename in names)

    def _read_json(self, path: Path, errors: list[str]) -> Any:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            return {}

    def _read_trials_csv(self, path: Path, errors: list[str]) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    parsed = dict(row)
                    for field in ["metrics_json", "policy_json", "diagnosis_json", "before_policy_json", "after_policy_json"]:
                        if field in parsed and parsed[field]:
                            try:
                                parsed[field[:-5] if field.endswith("_json") else field] = json.loads(parsed[field])
                            except json.JSONDecodeError:
                                pass
                    rows.append(parsed)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
        return rows

    def _normalize(self, result: dict[str, Any]) -> None:
        records = result.get("trial_record", [])
        if isinstance(records, dict):
            records = records.get("trials", [])
        if not records and isinstance(result.get("trials_json"), list):
            records = result["trials_json"]
        if not records and result.get("trials_csv"):
            records = result["trials_csv"]
        if not isinstance(records, list):
            records = []
        result["trials"] = records

        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}

        if not metrics and records:
            metrics = self._metrics_from_trials(records)
        result["metrics"] = metrics

        if not result.get("policy") and result.get("best_policy"):
            result["policy"] = result["best_policy"]

        result["policies"] = self._policies_from_trials(records, result.get("policy"))
        result["diagnoses"] = self._diagnoses_from_trials(records, result.get("diagnosis"))
        result["best_trial"] = self._best_trial(records)
        if not result.get("summary"):
            result["summary"] = self._summary_from_trials(records, result["best_trial"], result)

    def _metrics_from_trials(self, records: list[dict[str, Any]]) -> dict[str, list[Any]]:
        metrics: dict[str, list[Any]] = {
            "trial_index": [],
            "mAP50": [],
            "mAP50_95": [],
            "score": [],
            "proxy_score": [],
            "bbox_retention": [],
            "bbox_valid_rate": [],
            "diversity_score": [],
        }
        for index, row in enumerate(records):
            row_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row.get("metrics_json", {})
            if not isinstance(row_metrics, dict):
                row_metrics = {}
            metrics["trial_index"].append(row.get("trial_index", row.get("trial", index)))
            metrics["mAP50"].append(row_metrics.get("yolo_map50", row.get("mAP50")))
            metrics["mAP50_95"].append(row_metrics.get("yolo_map50_95", row.get("mAP50_95")))
            metrics["score"].append(row.get("score", row_metrics.get("final_score", row_metrics.get("proxy_score"))))
            metrics["proxy_score"].append(row_metrics.get("proxy_score"))
            metrics["bbox_retention"].append(row_metrics.get("bbox_retention_raw", row_metrics.get("bbox_retention")))
            metrics["bbox_valid_rate"].append(row_metrics.get("bbox_valid_rate"))
            metrics["diversity_score"].append(row_metrics.get("diversity_score"))
        return metrics

    def _policies_from_trials(self, records: list[dict[str, Any]], top_level_policy: Any) -> list[dict[str, Any]]:
        policies: list[dict[str, Any]] = []
        for index, row in enumerate(records):
            policy = row.get("policy")
            if not isinstance(policy, dict):
                policy = row.get("policy_json") if isinstance(row.get("policy_json"), dict) else {}
            policies.append({"trial": row.get("trial_index", row.get("trial", index)), "policy": policy})
        if not policies and isinstance(top_level_policy, dict) and top_level_policy:
            policies.append({"trial": "best", "policy": top_level_policy})
        return policies

    def _diagnoses_from_trials(self, records: list[dict[str, Any]], top_level_diagnosis: Any) -> list[dict[str, Any]]:
        diagnoses: list[dict[str, Any]] = []
        for index, row in enumerate(records):
            diagnosis = row.get("diagnosis")
            if not isinstance(diagnosis, dict):
                diagnosis = row.get("diagnosis_json") if isinstance(row.get("diagnosis_json"), dict) else {}
            if diagnosis:
                diagnoses.append({"trial": row.get("trial_index", row.get("trial", index)), "diagnosis": diagnosis})
        if not diagnoses and isinstance(top_level_diagnosis, dict) and top_level_diagnosis:
            diagnoses.append({"trial": "top-level", "diagnosis": top_level_diagnosis})
        return diagnoses

    def _best_trial(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if not records:
            return {}
        def score(row: dict[str, Any]) -> float:
            try:
                return float(row.get("score", 0.0))
            except (TypeError, ValueError):
                return 0.0
        return max(records, key=score)

    def _summary_from_trials(self, records: list[dict[str, Any]], best_trial: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        return {
            "total_trials": len(records),
            "best_trial": best_trial.get("trial_index", best_trial.get("trial", "-")) if best_trial else "-",
            "best_score": best_trial.get("score") if best_trial else None,
            "output_dir": result.get("output_dir"),
            "status": "已加载" if records else "未找到试验记录",
        }
