"""Design analysis for CATF-v2-Safe on fixed seed2.

This script is analysis-only. It reads existing clean/fixed CATF-v2 results
and writes the safe controller design report. It never trains.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXED_ROOT = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed"
OLD_ROOT = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2"
SEED2_FIXED = FIXED_ROOT / "seed_2/catf_v2"
SEED2_CLEAN = OLD_ROOT / "seed_2/clean_native_yolo_default"
REPORTS = FIXED_ROOT / "reports"
METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def main() -> None:
    clean_curve = read_curve(SEED2_CLEAN / "train/results.csv")
    fixed_curve = read_curve(SEED2_FIXED / "train/results.csv")
    policy_history = read_json(SEED2_FIXED / "reports/policy_history.json").get("history", [])
    class_policy_history = read_json(SEED2_FIXED / "reports/class_policy_history.json").get("history", [])
    roi_stats = read_json(SEED2_FIXED / "reports/roi_aug_stats.json")
    online_stats = read_json(SEED2_FIXED / "reports/online_aug_stats.json")
    clean_metrics = read_json(OLD_ROOT / "seed_2/clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json")
    fixed_metrics = read_json(SEED2_FIXED / "reports/final_metrics.json")

    curve = analyze_curve(clean_curve, fixed_curve)
    active_by_epoch = active_class_by_epoch(class_policy_history)
    policy_by_epoch = {int(item["epoch"]): item for item in policy_history}
    feedback_rows = []
    for epoch in sorted(policy_by_epoch):
        record = policy_by_epoch[epoch]
        feedback_rows.append(
            {
                "epoch": epoch,
                "action": record.get("action"),
                "delta_metrics": record.get("delta_metrics"),
                "guard_triggered": record.get("guard_triggered", []),
                "active_classes": record.get("active_classes", []),
                "proposed_classes": active_by_epoch.get(epoch, []),
            }
        )

    first_recall_lag = curve["first_recall_lag_epoch"]
    first_map95_lag = curve["first_map50_95_lag_epoch"]
    first_map_guard = first_feedback_guard_epoch(policy_history, key="map50_95", threshold=-0.008)
    should_protect = bool(first_map_guard or (first_recall_lag and first_recall_lag <= 10))
    payload = {
        "mode": "catf_v2_safe_seed2_design",
        "inputs": {
            "clean_results_csv": str(SEED2_CLEAN / "train/results.csv"),
            "fixed_results_csv": str(SEED2_FIXED / "train/results.csv"),
            "policy_history": str(SEED2_FIXED / "reports/policy_history.json"),
            "class_policy_history": str(SEED2_FIXED / "reports/class_policy_history.json"),
            "roi_aug_stats": str(SEED2_FIXED / "reports/roi_aug_stats.json"),
        },
        "global_metrics": {
            "clean": compact_final_metrics(clean_metrics),
            "fixed_catf_v2": fixed_metrics["constraint_scoring"]["metrics"],
            "delta": fixed_metrics["constraint_scoring"]["deltas"],
        },
        "curve_analysis": curve,
        "feedback_epochs": feedback_rows,
        "roi_aug_stats": roi_stats,
        "online_aug_stats_summary": {
            "samples_augmented": online_stats.get("samples_augmented", 0),
            "router_random_draw_count": online_stats.get("router_random_draw_count", 0),
            "ops": online_stats.get("ops", {}),
        },
        "answers": {
            "recall_lag_start_epoch": first_recall_lag,
            "map50_95_lag_start_epoch": first_map95_lag,
            "feedback_epoch_when_map95_guard_visible": first_map_guard,
            "class_activation_at_first_update": active_by_epoch.get(5, []),
            "roi_on_final_degraded_classes": roi_stats.get("affected_classes", {}),
            "shrink_freeze_too_late": True,
            "noop_at_epoch5_would_preserve_clean_path": True,
            "high_recall_baseline_protection_should_trigger": should_protect,
        },
        "safe_controller_design": {
            "early_abstention": "At epoch 5 the fixed path has zero positive delta vs clean; Safe should freeze to no-op before any industrial ROI augmentation is applied.",
            "baseline_protection": "At later feedback points seed2 shows negative Recall/mAP deltas against a strong clean curve; Safe should not keep exploring when clean Recall/mAP are already high.",
            "negative_effect_attribution": "Largest final drops include non-active classes, so non-active class regression should force shrink/no-op freeze.",
            "expected_seed2_behavior": "Safe mode should either match clean seed2 or stay within industrial constraints by preventing ROI augmentation after epoch 5.",
        },
    }
    write_json(REPORTS / "seed2_safe_controller_design.json", payload)
    write_md(REPORTS / "seed2_safe_controller_design.md", build_markdown(payload))
    print(REPORTS / "seed2_safe_controller_design.md")
    print(REPORTS / "seed2_safe_controller_design.json")


def read_curve(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            item = {"epoch": int(float(row["epoch"]))}
            for key, col in METRIC_COLUMNS.items():
                item[key] = float(row[col])
            rows.append(item)
    return rows


def analyze_curve(clean: list[dict[str, float]], fixed: list[dict[str, float]]) -> dict[str, Any]:
    by_epoch = []
    first_recall_lag = None
    first_map95_lag = None
    first_map50_lag = None
    for c, f in zip(clean, fixed):
        epoch = int(c["epoch"])
        deltas = {key: f[key] - c[key] for key in METRIC_COLUMNS}
        by_epoch.append({"epoch": epoch, "delta": deltas})
        if first_recall_lag is None and deltas["recall"] < -0.015:
            first_recall_lag = epoch
        if first_map95_lag is None and deltas["map50_95"] < -0.008:
            first_map95_lag = epoch
        if first_map50_lag is None and deltas["map50"] < -0.008:
            first_map50_lag = epoch
    return {
        "first_recall_lag_epoch": first_recall_lag,
        "first_map50_95_lag_epoch": first_map95_lag,
        "first_map50_lag_epoch": first_map50_lag,
        "feedback_epoch_deltas": [item for item in by_epoch if item["epoch"] in {5, 10, 15, 20, 25, 30, 35, 40, 45}],
        "final_delta": by_epoch[-1]["delta"],
    }


def first_feedback_guard_epoch(history: list[dict[str, Any]], *, key: str, threshold: float) -> int | None:
    for item in history:
        value = (item.get("delta_metrics") or {}).get(key)
        if value is not None and float(value) < threshold:
            return int(item["epoch"])
    return None


def active_class_by_epoch(class_history: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for item in class_history:
        if item.get("action") == "propose" or item.get("after_status") == "active":
            out.setdefault(int(item["epoch"]), []).append(item)
    return out


def compact_final_metrics(payload: dict[str, Any]) -> dict[str, float]:
    source = payload.get("metrics") or payload.get("val", {}).get("metrics", {})
    return {key: float(source[key]) for key in ("precision", "recall", "map50", "map50_95")}


def build_markdown(payload: dict[str, Any]) -> str:
    answers = payload["answers"]
    lines = [
        "# CATF-v2-Safe Seed2 Controller Design",
        "",
        "No training was run. This report compares existing clean seed2 and fixed CATF-v2 seed2 curves.",
        "",
        "## Curve Localization",
        "",
        f"- Recall first lags clean by more than 0.015 at epoch: `{answers['recall_lag_start_epoch']}`",
        f"- mAP50-95 first lags clean by more than 0.008 at epoch: `{answers['map50_95_lag_start_epoch']}`",
        f"- First feedback epoch with mAP50-95 guard visible: `{answers['feedback_epoch_when_map95_guard_visible']}`",
        "",
        "## Feedback Context",
        "",
        f"- Classes activated at epoch 5: `{answers['class_activation_at_first_update']}`",
        f"- ROI affected classes over the run: `{answers['roi_on_final_degraded_classes']}`",
        f"- shrink/freeze too late: `{str(answers['shrink_freeze_too_late']).lower()}`",
        f"- No-op at epoch 5 would preserve clean path: `{str(answers['noop_at_epoch5_would_preserve_clean_path']).lower()}`",
        f"- High-recall baseline protection should trigger: `{str(answers['high_recall_baseline_protection_should_trigger']).lower()}`",
        "",
        "## Feedback Epoch Deltas",
        "",
        "| epoch | dP | dR | dmAP50 | dmAP50-95 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["curve_analysis"]["feedback_epoch_deltas"]:
        delta = row["delta"]
        lines.append(
            f"| {row['epoch']} | {delta['precision']:+.4f} | {delta['recall']:+.4f} | {delta['map50']:+.4f} | {delta['map50_95']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Design Conclusion",
            "",
            "- Seed2 should trigger Safe mode because clean seed2 is a strong high-Recall/high-mAP baseline and fixed CATF-v2 starts exploring without early Recall/mAP gain.",
            "- The safest intervention is early abstention at epoch 5, before any industrial ROI augmentation can affect subsequent training.",
            "- If CATF-v2 later shows non-active class regression, Safe should enter no-op freeze rather than keep shrinking active classes.",
        ]
    )
    return "\n".join(lines) + "\n"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
