from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.catf_v2.adaptive_burnin import AdaptiveBurninConfig, AdaptiveBurninController  # noqa: E402
from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix  # noqa: E402


FIXED_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed")
CLEAN_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
GATED_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated")
SAFE_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe")
REPORTS = FIXED_ROOT / "reports"
SEEDS = (0, 1, 2)
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
RESULT_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_results(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(float(row["epoch"]))
            out[epoch] = {key: float(row[column]) for key, column in RESULT_COLUMNS.items()}
    return out


def clean_run(seed: int) -> Path:
    return CLEAN_ROOT / f"seed_{seed}" / "clean_native_yolo_default"


def fixed_run(seed: int) -> Path:
    root = FIXED_ROOT / f"seed_{seed}" / "catf_v2"
    if seed == 1 and not (root / "reports" / "policy_history.json").exists():
        return Path("outputs/experiments/catf_v2_fixed_seed1_50ep")
    return root


def safe_run(seed: int) -> Path:
    return SAFE_ROOT / f"seed_{seed}" / "catf_v2_safe"


def gated_run(seed: int) -> Path:
    return GATED_ROOT / f"seed_{seed}" / "catf_v2_gated"


def load_history(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    return list(payload.get("history", []) if isinstance(payload, dict) else payload)


def load_per_class(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("per_class_diagnosis_path")
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    return read_json(path)


def compact_final_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics") or payload.get("val", {}).get("metrics") or payload
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def simulate_seed(seed: int, fixed_summary: dict[str, Any]) -> dict[str, Any]:
    clean_rows = read_results(clean_run(seed) / "train" / "results.csv")
    fixed_history = {int(row["epoch"]): row for row in load_history(fixed_run(seed) / "reports" / "policy_history.json")}
    clean_metrics = dict(fixed_summary[str(seed)]["clean_native"])
    class_names = class_names_from_policy_history(fixed_history)
    old_policy = initial_policy_matrix(class_names)
    controller = AdaptiveBurninController(AdaptiveBurninConfig())
    events: list[dict[str, Any]] = []
    final_policy = old_policy

    for epoch in (5, 10, 15):
        record = fixed_history.get(epoch, {})
        proposed_policy = record.get("accepted_policy") or record.get("proposed_policy") or final_policy
        metric_history = [clean_rows[index] for index in sorted(clean_rows) if index <= epoch]
        event = controller.evaluate(
            epoch=epoch,
            metrics=clean_rows[epoch],
            reference_metrics=clean_rows[epoch],
            clean_reference_metrics=clean_metrics,
            metric_history=metric_history,
            per_class_diagnosis=load_per_class(record),
            old_policy=final_policy,
            proposed_policy=proposed_policy,
            close_mosaic_start_epoch=40,
        )
        final_policy = event.get("policy", final_policy)
        stored = dict(event)
        stored.pop("policy", None)
        events.append(stored)
        if event.get("candidate_branch_started") or event.get("action") == "no_op_fallback":
            break

    start_event = next((event for event in events if event.get("candidate_branch_started")), None)
    noop_event = next((event for event in events if event.get("action") == "no_op_fallback"), None)
    safe_events = read_json(safe_run(seed) / "reports" / "safe_controller_events.json").get("events", [])
    gated_events = read_json(gated_run(seed) / "reports" / "gated_controller_events.json").get("events", [])
    return {
        "seed": seed,
        "adaptive_start_epoch": start_event.get("epoch") if start_event else None,
        "candidate_started": bool(start_event),
        "candidate_action": start_event.get("action") if start_event else None,
        "noop_fallback": bool(noop_event),
        "noop_epoch": noop_event.get("epoch") if noop_event else None,
        "strong_clean_baseline_protection": any(
            "strong_clean_baseline_protection" in (event.get("reasons") or []) for event in events
        ),
        "expected_final_path": "clean_noop_fallback" if noop_event else "adaptive_rb_candidate",
        "safe_epoch5_noop": any(int(event.get("epoch", -1)) == 5 and event.get("action") == "no_op_freeze" for event in safe_events),
        "gated_seed2_epoch10_fallback": bool(
            seed == 2 and any(int(event.get("epoch", -1)) == 10 and event.get("action") == "no_op_freeze" for event in gated_events)
        ),
        "fixed_constraint_failed": bool(fixed_summary[str(seed)]["fixed_constraint_failed"]),
        "events": events,
    }


def class_names_from_policy_history(history: dict[int, dict[str, Any]]) -> dict[int, str]:
    for record in history.values():
        policy = record.get("accepted_policy") or record.get("proposed_policy") or {}
        classes = policy.get("classes") or {}
        if classes:
            return {int(class_id): row.get("class_name", str(class_id)) for class_id, row in classes.items()}
    return {index: f"class{index}" for index in range(13)}


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seed0, seed1, seed2 = rows
    reasonable = (
        seed0["candidate_started"]
        and seed1["candidate_started"]
        and seed2["noop_fallback"]
        and seed2["strong_clean_baseline_protection"]
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code_commit_used": git_commit(),
        "seed0_adaptive_start_epoch": seed0["adaptive_start_epoch"],
        "seed1_adaptive_start_epoch": seed1["adaptive_start_epoch"],
        "seed2_candidate_started": seed2["candidate_started"],
        "seed2_noop_fallback_epoch": seed2["noop_epoch"],
        "seed2_strong_clean_baseline_protection": seed2["strong_clean_baseline_protection"],
        "avoids_safe_epoch5_early_noop": all(row["events"][0]["action"] == "burnin_observe" for row in rows),
        "avoids_gated_seed2_epoch10_late_fallback": seed2["noop_fallback"] and seed2["noop_epoch"] == 15,
        "simulation_reasonable": reasonable,
        "real_training_validation_recommended": reasonable,
        "overfit_assessment": (
            "The trigger uses metric stability, diagnosis evidence, support guards, no-aug/high-FP guards, and final clean-baseline protection. "
            "It does not branch on seed id or class id, but the default thresholds are still calibrated from the current three-seed audit and need larger validation."
        ),
    }


def f(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Adaptive Burn-in CATF-v2 Retrospective Simulation",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Code commit used: `{summary['code_commit_used']}`",
        "",
        "## Seed Decisions",
        "",
        "| seed | adaptive start epoch | candidate started | no-op fallback | no-op epoch | strong clean baseline protection | expected path |",
        "|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['adaptive_start_epoch']} | {str(row['candidate_started']).lower()} | "
            f"{str(row['noop_fallback']).lower()} | {row['noop_epoch']} | "
            f"{str(row['strong_clean_baseline_protection']).lower()} | {row['expected_final_path']} |"
        )
    lines += [
        "",
        "## Required Answers",
        "",
        f"1. Seed0 adaptive trigger epoch: `{summary['seed0_adaptive_start_epoch']}`.",
        f"2. Seed1 adaptive trigger epoch: `{summary['seed1_adaptive_start_epoch']}`.",
        f"3. Seed2 candidate trigger: `{str(summary['seed2_candidate_started']).lower()}`; strong clean baseline protection: `{str(summary['seed2_strong_clean_baseline_protection']).lower()}`.",
        f"4. Avoids Safe epoch5 early no-op: `{str(summary['avoids_safe_epoch5_early_noop']).lower()}`.",
        f"5. Avoids Gated seed2 epoch10 late fallback: `{str(summary['avoids_gated_seed2_epoch10_late_fallback']).lower()}`.",
        "6. Start conditions are recorded per seed in the event trace below.",
        "7. Key blocking/triggering conditions: metric stability, max-burnin low-risk force for seed0/1, and final strong clean-baseline protection for seed2.",
        f"8. Overfit assessment: {summary['overfit_assessment']}",
        f"9. Recommended to enter real training validation: `{str(summary['real_training_validation_recommended']).lower()}`.",
        "",
        "## Event Trace",
        "",
    ]
    for row in rows:
        lines.append(f"### Seed {row['seed']}")
        lines.append("")
        lines.append("| epoch | action | start_condition | reasons | map50_range | recall_range | eligible_classes | medium_issue_classes |")
        lines.append("|---:|---|---|---|---:|---:|---|---|")
        for event in row["events"]:
            cond = event.get("start_condition") or {}
            stability = cond.get("metric_stability") or {}
            lines.append(
                f"| {event.get('epoch')} | {event.get('action')} | {str(event.get('start_condition_met')).lower()} | "
                f"`{event.get('reasons', [])}` | {f(stability.get('map50_range'))} | {f(stability.get('recall_range'))} | "
                f"`{cond.get('eligible_active_classes', [])}` | `{cond.get('medium_confidence_issue_classes', [])}` |"
            )
        lines.append("")
    return lines


def main() -> None:
    fixed_summary = read_json(FIXED_ROOT / "reports" / "multiseed_catf_v2_fixed_summary.json")["seeds"]
    rows = [simulate_seed(seed, fixed_summary) for seed in SEEDS]
    summary = build_summary(rows)
    payload = {"summary": summary, "rows": rows}
    write_json(REPORTS / "adaptive_burnin_retrospective_simulation.json", payload)
    write_md(REPORTS / "adaptive_burnin_retrospective_simulation.md", build_markdown(summary, rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
