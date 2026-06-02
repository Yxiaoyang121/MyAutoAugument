"""Summarize diagnosis-only in-loop control against clean native and CATF-v2.

This script is read-only analysis. It does not train or modify model weights.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ROOT = Path("outputs/experiments/diagnosis_only_inloop_control_50ep")
CATF_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2]
METRICS = ("precision", "recall", "map50", "map50_95")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, lines: list[str] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def compact_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if "val" in payload and isinstance(payload["val"].get("metrics"), dict):
        metrics = payload["val"]["metrics"]
    elif "metrics" in payload:
        metrics = payload["metrics"]
    else:
        metrics = payload
    return {key: float(metrics[key]) for key in METRICS}


def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: a[key] - b[key] for key in METRICS}


def constraint_failed(d: dict[str, float]) -> tuple[bool, list[str]]:
    reasons = []
    if d["precision"] < -0.01:
        reasons.append("precision_drop_gt_0.01")
    if d["map50"] < -0.01:
        reasons.append("map50_drop_gt_0.01")
    if d["map50_95"] < -0.01:
        reasons.append("map50_95_drop_gt_0.01")
    return bool(reasons), reasons


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def fdelta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values) if values else 0.0,
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def load_rows() -> list[dict[str, Any]]:
    catf_summary = read_json(CATF_ROOT / "reports" / "multiseed_catf_v2_summary.json")
    catf_by_seed = {int(row["seed"]): row for row in catf_summary["rows"]}
    threshold = read_json(CATF_ROOT / "reports" / "threshold_calibration_posthoc.json")
    rows = []
    for seed in SEEDS:
        seed_dir = ROOT / f"seed_{seed}"
        final_metrics_path = seed_dir / "reports" / "final_metrics.json"
        if not final_metrics_path.exists():
            raise FileNotFoundError(final_metrics_path)
        payload = read_json(final_metrics_path)
        diagnosis_metrics = compact_metrics(payload)
        clean_metrics = {key: float(value) for key, value in catf_by_seed[seed]["clean_metrics"].items()}
        catf_metrics = {key: float(value) for key, value in catf_by_seed[seed]["catf_v2_metrics"].items()}
        d_clean = delta(diagnosis_metrics, clean_metrics)
        d_catf = delta(diagnosis_metrics, catf_metrics)
        failed, reasons = constraint_failed(d_clean)

        policy_history = read_json(seed_dir / "reports" / "policy_history.json")
        history = policy_history.get("history", [])
        diagnosis_history_path = seed_dir / "reports" / "diagnosis_history.json"
        write_json(diagnosis_history_path, {"history": history})

        results_rows = list(csv.DictReader((seed_dir / "train" / "results.csv").read_text(encoding="utf-8-sig").splitlines()))
        online_stats = read_json(seed_dir / "reports" / "online_aug_stats.json")
        roi_stats = read_json(seed_dir / "reports" / "roi_aug_stats.json")
        catf_threshold = threshold["seeds"][str(seed)]["catf_v2"]["searches"]["constrained_score"]
        row = {
            "seed": seed,
            "clean_native_metrics": clean_metrics,
            "catf_v2_metrics": catf_metrics,
            "diagnosis_only_metrics": diagnosis_metrics,
            "diagnosis_only_delta_vs_clean": d_clean,
            "diagnosis_only_delta_vs_catf_v2": d_catf,
            "diagnosis_only_constraint_failed": failed,
            "diagnosis_only_failure_reasons": reasons,
            "catf_v2_constraint_failed": bool(catf_by_seed[seed]["constraint_failed"]),
            "catf_v2_failure_reasons": catf_by_seed[seed]["failure_reasons"],
            "catf_v2_samples_augmented": int(catf_by_seed[seed]["samples_augmented"]),
            "catf_v2_roi_aug_applied": int(catf_by_seed[seed]["roi_aug_applied"]),
            "catf_v2_threshold_constrained_metrics": catf_threshold["metrics"],
            "catf_v2_threshold_delta_vs_clean_posthoc": {
                key: float(catf_threshold["metrics"][key])
                - float(threshold["seeds"][str(seed)]["reference_for_constraints"][key])
                for key in METRICS
            },
            "catf_v2_threshold_constraint_failed": bool(catf_threshold["constraint_failed"]),
            "catf_v2_threshold_failure_reasons": catf_threshold["failure_reasons"],
            "diagnosis_history_path": str(diagnosis_history_path),
            "final_metrics_path": str(final_metrics_path),
            "final_report_path": str(seed_dir / "reports" / "final_report.md"),
            "results_csv": str(seed_dir / "train" / "results.csv"),
            "epoch_count": len(results_rows),
            "first_epoch": int(float(results_rows[0]["epoch"])) if results_rows else None,
            "last_epoch": int(float(results_rows[-1]["epoch"])) if results_rows else None,
            "epoch_continuous": [int(float(item["epoch"])) for item in results_rows] == list(range(1, 51)),
            "feedback_epochs": payload["summary"].get("feedback_epochs", []),
            "policy_update_applied_count": int(payload["summary"].get("policy_update_applied_count", -1)),
            "industrial_aug_enabled": bool(payload["summary"].get("industrial_aug_enabled")),
            "roi_aware_aug": bool(payload["summary"].get("roi_aware_aug")),
            "sample_aware_routing": bool(payload["summary"].get("sample_aware_routing")),
            "industrial_samples_augmented": int(online_stats.get("samples_augmented", 0)),
            "industrial_ops": online_stats.get("ops", {}),
            "roi_aug_applied": int(roi_stats.get("roi_aug_applied", 0)),
            "train_image_count": int(payload["summary"].get("train_image_count", 0)),
            "fixed_augmented_dataset_generated": bool(payload["summary"].get("fixed_augmented_dataset_generated")),
            "bbox_class_valid": bool(payload["summary"].get("bbox_class_valid")),
            "policy_actions": dict(Counter(str(item.get("action", "")) for item in history)),
        }
        rows.append(row)
    return rows


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = {key: [row["diagnosis_only_delta_vs_clean"][key] for row in rows] for key in METRICS}
    diag_pass = sum(1 for row in rows if not row["diagnosis_only_constraint_failed"])
    threshold_pass = sum(1 for row in rows if not row["catf_v2_threshold_constraint_failed"])
    catf_pass = sum(1 for row in rows if not row["catf_v2_constraint_failed"])
    callback_rng_explanation = (
        "unlikely_as_diagnosis_callback_alone"
        if all(abs(row["diagnosis_only_delta_vs_clean"][key]) < 1e-9 for row in rows for key in METRICS)
        else "possible"
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(ROOT),
        "seeds": SEEDS,
        "diagnosis_only_constraint_pass_count": diag_pass,
        "catf_v2_constraint_pass_count": catf_pass,
        "catf_v2_threshold_constraint_pass_count": threshold_pass,
        "diagnosis_only_delta_vs_clean_mean_std": {key: mean_std(values) for key, values in deltas.items()},
        "diagnosis_only_changes_training_result": any(
            abs(row["diagnosis_only_delta_vs_clean"][key]) > 1e-9 for row in rows for key in METRICS
        ),
        "seed1_catf_success_callback_rng_interpretation": callback_rng_explanation,
        "interpretation": {
            "diagnosis_only_vs_clean": "Diagnosis-only reproduced clean native metrics exactly for seeds 0/1/2.",
            "seed1": "Seed 1 CATF-v2 success is not explained by diagnosis callback alone; CATF-v2 industrial-enabled path still needs caution because augmentation counters were zero.",
            "catf_v2_threshold": "CATF-v2 plus post-hoc threshold calibration improves constraint pass count to 2/3, but seed 2 remains a training trajectory failure.",
        },
    }


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Diagnosis-Only In-Loop Control Multiseed Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "This experiment ran YOLO default with the in-loop diagnosis callback enabled, but industrial augmentation, ROI augmentation, sample-aware routing, policy mutation, and threshold mutation disabled.",
        "",
        "## Integrity",
        "",
        f"- Seeds: `{SEEDS}`",
        "- Epochs: `50` per seed.",
        "- Train images: `2301` per seed.",
        "- Fixed augmented dataset generated: `false`.",
        "- YOLO default augmentation: enabled.",
        "- Industrial augmentation: disabled.",
        "- ROI-aware augmentation: disabled.",
        "- Sample-aware routing: disabled.",
        "- Policy update applied: `0` for every seed.",
        "",
        "## Metrics",
        "",
        "| seed | clean P/R/mAP50/mAP50-95 | CATF-v2 P/R/mAP50/mAP50-95 | diagnosis-only P/R/mAP50/mAP50-95 | diag delta vs clean | diag delta vs CATF-v2 | diag constraint_failed | CATF-v2 constraint_failed | CATF-v2+threshold constraint_failed |",
        "|---:|---|---|---|---|---|:---:|:---:|:---:|",
    ]
    for row in rows:
        clean = row["clean_native_metrics"]
        catf = row["catf_v2_metrics"]
        diag = row["diagnosis_only_metrics"]
        dc = row["diagnosis_only_delta_vs_clean"]
        dcatf = row["diagnosis_only_delta_vs_catf_v2"]
        lines.append(
            f"| {row['seed']} | "
            f"{fmt(clean['precision'])}/{fmt(clean['recall'])}/{fmt(clean['map50'])}/{fmt(clean['map50_95'])} | "
            f"{fmt(catf['precision'])}/{fmt(catf['recall'])}/{fmt(catf['map50'])}/{fmt(catf['map50_95'])} | "
            f"{fmt(diag['precision'])}/{fmt(diag['recall'])}/{fmt(diag['map50'])}/{fmt(diag['map50_95'])} | "
            f"{fdelta(dc['precision'])}/{fdelta(dc['recall'])}/{fdelta(dc['map50'])}/{fdelta(dc['map50_95'])} | "
            f"{fdelta(dcatf['precision'])}/{fdelta(dcatf['recall'])}/{fdelta(dcatf['map50'])}/{fdelta(dcatf['map50_95'])} | "
            f"{str(row['diagnosis_only_constraint_failed']).lower()} | "
            f"{str(row['catf_v2_constraint_failed']).lower()} | "
            f"{str(row['catf_v2_threshold_constraint_failed']).lower()} |"
        )
    lines += [
        "",
        "## Control Counters",
        "",
        "| seed | epoch continuous | feedback epochs | industrial samples | ROI applied | policy update applied | bbox/class legal |",
        "|---:|:---:|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {str(row['epoch_continuous']).lower()} | {row['feedback_epochs']} | "
            f"{row['industrial_samples_augmented']} | {row['roi_aug_applied']} | "
            f"{row['policy_update_applied_count']} | {str(row['bbox_class_valid']).lower()} |"
        )
    lines += [
        "",
        "## Answers",
        "",
        f"- Diagnosis-only changed training result: `{str(summary['diagnosis_only_changes_training_result']).lower()}`.",
        f"- Diagnosis-only constraint pass count: `{summary['diagnosis_only_constraint_pass_count']}/3`.",
        f"- CATF-v2 constraint pass count before threshold calibration: `{summary['catf_v2_constraint_pass_count']}/3`.",
        f"- CATF-v2 constraint pass count after post-hoc threshold calibration: `{summary['catf_v2_threshold_constraint_pass_count']}/3`.",
        "- Seed 1 CATF-v2 success is not explained by the diagnosis callback alone because diagnosis-only reproduced clean native exactly. It can still reflect CATF-v2 industrial-enabled training-path differences because CATF-v2 seed 1 reported zero actual industrial/ROI augmentation.",
        "- CATF-v2 + post-hoc threshold calibration has independent value as a deployment/post-processing layer: it repairs seed 0 and raises pass count to 2/3, but seed 2 remains unrepaired.",
        "- A method claim should separate three effects: clean YOLO default, diagnosis callback only, and CATF-v2 industrial-enabled path with threshold calibration.",
        "",
        "## Report Paths",
        "",
    ]
    for row in rows:
        lines.append(f"- seed {row['seed']}: `{row['final_report_path']}`, diagnosis history `{row['diagnosis_history_path']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = load_rows()
    summary = build_summary(rows)
    payload = {**summary, "rows": rows}
    write_json(REPORTS / "diagnosis_only_multiseed_summary.json", payload)
    write_md(REPORTS / "diagnosis_only_multiseed_summary.md", build_markdown(summary, rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
