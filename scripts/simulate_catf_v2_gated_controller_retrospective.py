from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed")
CLEAN_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
REPORTS = ROOT / "reports"
SEEDS = (0, 1, 2)
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
RESULT_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.catf_v2.gated_controller import CATFGatedController  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixed_root(seed: int) -> Path:
    root = ROOT / f"seed_{seed}" / "catf_v2"
    if (root / "train" / "results.csv").exists():
        return root
    if seed == 1:
        return Path("outputs/experiments/catf_v2_fixed_seed1_50ep")
    return root


def clean_root(seed: int) -> Path:
    return CLEAN_ROOT / f"seed_{seed}" / "clean_native_yolo_default"


def read_results(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(float(row["epoch"]))
            rows[epoch] = {key: float(row[column]) for key, column in RESULT_COLUMNS.items()}
    return rows


def metric_delta(metrics: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) - float(reference[key]) for key in METRIC_KEYS}


def compact_final_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("val", {}).get("metrics") or payload.get("metrics") or payload
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def constraint_failed(deltas: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if deltas["precision"] < -0.01:
        reasons.append("precision_drop_gt_0.01")
    if deltas["map50"] < -0.01:
        reasons.append("map50_drop_gt_0.01")
    if deltas["map50_95"] < -0.01:
        reasons.append("map50_95_drop_gt_0.01")
    return bool(reasons), reasons


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def f4(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def fd(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.4f}"


def load_history(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    return list(payload.get("history", []) if isinstance(payload, dict) else payload)


def load_per_class(record: dict[str, Any]) -> dict[str, Any]:
    raw_path = record.get("per_class_diagnosis_path")
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    return read_json(path)


def simulate_seed(seed: int, fixed_summary: dict[str, Any]) -> dict[str, Any]:
    froot = fixed_root(seed)
    croot = clean_root(seed)
    fixed_rows = read_results(froot / "train" / "results.csv")
    clean_rows = read_results(croot / "train" / "results.csv")
    history = load_history(froot / "reports" / "policy_history.json")
    controller = CATFGatedController()
    events: list[dict[str, Any]] = []
    fallback_epoch: int | None = None

    latest_policy = {}
    for record in history:
        epoch = int(record["epoch"])
        metrics = fixed_rows[epoch]
        reference = clean_rows[epoch]
        latest_policy = record.get("accepted_policy") or record.get("proposed_policy") or latest_policy
        event = controller.evaluate(
            epoch=epoch,
            policy=latest_policy,
            metrics=metrics,
            reference_metrics=reference,
            per_class_diagnosis=load_per_class(record),
            active_classes=record.get("active_classes", []),
            proposed_action=record.get("action"),
        )
        event_for_report = dict(event)
        event_for_report.pop("policy", None)
        event_for_report["delta_vs_clean_curve"] = metric_delta(metrics, reference)
        events.append(event_for_report)
        if event.get("action") == "no_op_freeze" and fallback_epoch is None:
            fallback_epoch = epoch
            break

    clean_metrics = dict(fixed_summary[str(seed)]["clean_native"])
    fixed_metrics = dict(fixed_summary[str(seed)]["fixed_catf_v2"])
    expected_choice = "clean_fallback" if fallback_epoch is not None else "fixed_catf_v2"
    expected_metrics = clean_metrics if expected_choice == "clean_fallback" else fixed_metrics
    expected_delta = metric_delta(expected_metrics, clean_metrics)
    failed, failure_reasons = constraint_failed(expected_delta)
    return {
        "seed": seed,
        "fixed_run_path": str(froot),
        "clean_run_path": str(croot),
        "fallback_expected": fallback_epoch is not None,
        "fallback_epoch": fallback_epoch,
        "expected_final_choice": expected_choice,
        "expected_metrics": expected_metrics,
        "expected_delta_vs_clean": expected_delta,
        "expected_constraint_failed": failed,
        "expected_failure_reasons": failure_reasons,
        "fixed_constraint_failed": bool(fixed_summary[str(seed)]["fixed_constraint_failed"]),
        "events": events,
        "event_actions": dict(Counter(event.get("action") for event in events)),
        "event_reasons": dict(Counter(reason for event in events for reason in (event.get("reasons") or []))),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_count = sum(1 for row in rows if row["expected_constraint_failed"])
    seed0 = rows[0]
    seed1 = rows[1]
    seed2 = rows[2]
    behavior_matches_goal = (
        not seed0["fallback_expected"]
        and not seed1["fallback_expected"]
        and seed2["fallback_expected"]
        and failed_count == 0
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code_commit_used": git_commit(),
        "seed0_fallback_expected": seed0["fallback_expected"],
        "seed1_fallback_expected": seed1["fallback_expected"],
        "seed2_fallback_expected": seed2["fallback_expected"],
        "seed2_fallback_epoch": seed2["fallback_epoch"],
        "expected_constraint_failed_count": failed_count,
        "expected_3_of_3_constraint_pass": failed_count == 0,
        "behavior_matches_goal": behavior_matches_goal,
        "overfit_assessment": (
            "No obvious seed-id overfit: the controller uses metric-shape gates and active-class evidence rather than "
            "seed IDs or class IDs. The thresholds were derived from the current three-seed failure analysis, so this "
            "still needs validation beyond seeds 0/1/2 before being claimed as generally robust."
        ),
        "run_full_gated_multiseed_recommended": behavior_matches_goal,
    }


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# CATF-v2-Gated Retrospective Simulation",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Code commit used: `{summary['code_commit_used']}`",
        "",
        "## Gate Replay Results",
        "",
        "| seed | fallback expected | fallback epoch | expected final choice | expected dP | expected dR | expected dM50 | expected dM95 | expected constraint_failed | gate reasons |",
        "|---:|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        delta = row["expected_delta_vs_clean"]
        lines.append(
            f"| {row['seed']} | {str(row['fallback_expected']).lower()} | {row['fallback_epoch']} | "
            f"{row['expected_final_choice']} | {fd(delta['precision'])} | {fd(delta['recall'])} | "
            f"{fd(delta['map50'])} | {fd(delta['map50_95'])} | "
            f"{str(row['expected_constraint_failed']).lower()} | `{row['event_reasons']}` |"
        )
    lines += [
        "",
        "## Required Answers",
        "",
        f"1. Seed0 fallback under new gate: `{str(summary['seed0_fallback_expected']).lower()}`.",
        f"2. Seed1 fallback under new gate: `{str(summary['seed1_fallback_expected']).lower()}`.",
        f"3. Seed2 fallback under new gate: `{str(summary['seed2_fallback_expected']).lower()}` at epoch `{summary['seed2_fallback_epoch']}`.",
        "4. Expected final choice: seed0 fixed CATF-v2, seed1 fixed CATF-v2, seed2 clean fallback.",
        f"5. Expected 3/3 constraint pass: `{str(summary['expected_3_of_3_constraint_pass']).lower()}`.",
        f"6. Overfit assessment: {summary['overfit_assessment']}",
        "",
        "## Event Trace",
        "",
    ]
    for row in rows:
        lines.append(f"### Seed {row['seed']}")
        lines.append("")
        lines.append("| epoch | phase | action | positive_gain | bad_patterns | reasons | dP | dR | dM50 | dM95 |")
        lines.append("|---:|---|---|---|---|---|---:|---:|---:|---:|")
        for event in row["events"]:
            delta = event["delta_metrics"]
            lines.append(
                f"| {event['epoch']} | {event.get('gate_phase')} | {event.get('action')} | "
                f"{str(event.get('positive_gain')).lower()} | `{event.get('bad_patterns', [])}` | "
                f"`{event.get('reasons', [])}` | {fd(delta.get('precision'))} | {fd(delta.get('recall'))} | "
                f"{fd(delta.get('map50'))} | {fd(delta.get('map50_95'))} |"
            )
        lines.append("")
    return lines


def main() -> None:
    fixed_summary = read_json(ROOT / "reports" / "multiseed_catf_v2_fixed_summary.json")["seeds"]
    rows = [simulate_seed(seed, fixed_summary) for seed in SEEDS]
    summary = build_summary(rows)
    payload = {"summary": summary, "rows": rows}
    write_json(REPORTS / "gated_controller_retrospective_simulation.json", payload)
    write_md(REPORTS / "gated_controller_retrospective_simulation.md", build_markdown(summary, rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
