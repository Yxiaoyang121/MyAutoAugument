from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated")
CLEAN_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
FIXED_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed")
SAFE_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe")
REPORTS = ROOT / "reports"
SEEDS = (0, 1, 2)
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def compact_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if isinstance(payload.get("val"), dict) and isinstance(payload["val"].get("metrics"), dict):
        metrics = payload["val"]["metrics"]
    elif isinstance(payload.get("metrics"), dict):
        metrics = payload["metrics"]
    else:
        metrics = payload
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def delta(metrics: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) - float(reference[key]) for key in METRIC_KEYS}


def constraint_failed(deltas: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if deltas["precision"] < -0.01:
        reasons.append("precision_drop_gt_0.01")
    if deltas["map50"] < -0.01:
        reasons.append("map50_drop_gt_0.01")
    if deltas["map50_95"] < -0.01:
        reasons.append("map50_95_drop_gt_0.01")
    return bool(reasons), reasons


def f4(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def fd(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.4f}"


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def gated_root(seed: int) -> Path:
    return ROOT / f"seed_{seed}" / "catf_v2_gated"


def clean_root(seed: int) -> Path:
    return CLEAN_ROOT / f"seed_{seed}" / "clean_native_yolo_default"


def safe_root(seed: int) -> Path:
    return SAFE_ROOT / f"seed_{seed}" / "catf_v2_safe"


def count_epochs(results_csv: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(results_csv.read_text(encoding="utf-8-sig").splitlines()))
    epochs = [int(float(row["epoch"])) for row in rows]
    return {
        "epoch_count": len(epochs),
        "first_epoch": epochs[0] if epochs else None,
        "last_epoch": epochs[-1] if epochs else None,
        "epoch_sequence": epochs,
        "epoch_continuous": epochs == list(range(1, 51)),
    }


def class_policy_counts(path: Path) -> tuple[dict[str, int], bool, int]:
    payload = read_json(path)
    history = payload.get("history", []) if isinstance(payload, dict) else payload
    active = Counter()
    ok3_active = False
    for item in history:
        action = str(item.get("action", ""))
        class_id = int(item.get("class_id", -1))
        if action in {"propose", "accept"}:
            active[str(class_id)] += 1
            if class_id == 1:
                ok3_active = True
    return dict(active), ok3_active, int(active.get("1", 0))


def event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    first_fallback_epoch = None
    reasons = Counter()
    actions = Counter()
    bad_patterns = Counter()
    for event in events:
        actions[str(event.get("action", ""))] += 1
        for reason in event.get("reasons", []) or []:
            reasons[str(reason)] += 1
        for pattern in event.get("bad_patterns", []) or []:
            bad_patterns[str(pattern)] += 1
        if event.get("action") == "no_op_freeze" and first_fallback_epoch is None:
            first_fallback_epoch = int(event.get("epoch"))
    return {
        "event_count": len(events),
        "actions": dict(actions),
        "reasons": dict(reasons),
        "bad_patterns": dict(bad_patterns),
        "fallback_triggered": first_fallback_epoch is not None,
        "first_fallback_epoch": first_fallback_epoch,
    }


def strict_noop_after_fallback(root: Path, fallback_epoch: int | None) -> bool:
    if fallback_epoch is None:
        return True
    checked = 0
    for path in sorted((root / "configs").glob("active_policy_epoch_*.json")):
        epoch = int(path.stem.rsplit("_", 1)[-1])
        if epoch < fallback_epoch:
            continue
        policy = read_json(path)
        probs = [
            float(op.get("prob", 0.0) or 0.0)
            for row in (policy.get("classes") or {}).values()
            for op in (row.get("ops") or {}).values()
        ]
        checked += 1
        if probs and max(probs) > 0.0:
            return False
    return checked > 0


def load_seed(seed: int, fixed_summary: dict[str, Any], safe_rows_by_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    groot = gated_root(seed)
    g_payload = read_json(groot / "reports" / "final_metrics.json")
    clean_payload = read_json(clean_root(seed) / "reports" / "clean_native_yolo_default_metrics.json")
    safe_row = safe_rows_by_seed[seed]

    clean_metrics = compact_metrics(clean_payload)
    fixed_metrics = dict(fixed_summary[str(seed)]["fixed_catf_v2"])
    safe_metrics = dict(safe_row["safe_metrics"])
    gated_metrics = compact_metrics(g_payload)
    gated_delta_clean = delta(gated_metrics, clean_metrics)
    gated_delta_fixed = delta(gated_metrics, fixed_metrics)
    gated_delta_safe = delta(gated_metrics, safe_metrics)
    failed, failure_reasons = constraint_failed(gated_delta_clean)

    online_stats = read_json(groot / "reports" / "online_aug_stats.json")
    roi_stats = read_json(groot / "reports" / "roi_aug_stats.json")
    active_classes, ok3_active, ok3_active_count = class_policy_counts(groot / "reports" / "class_policy_history.json")
    events = g_payload.get("gated_controller_events") or []
    events_summary = event_summary(events)
    fallback_epoch = events_summary["first_fallback_epoch"]

    return {
        "seed": seed,
        "clean_metrics": clean_metrics,
        "fixed_catf_v2_metrics": fixed_metrics,
        "safe_metrics": safe_metrics,
        "gated_metrics": gated_metrics,
        "fixed_delta_vs_clean": dict(fixed_summary[str(seed)]["delta_fixed_vs_clean"]),
        "safe_delta_vs_clean": dict(safe_row["safe_delta_vs_clean"]),
        "gated_delta_vs_clean": gated_delta_clean,
        "gated_delta_vs_fixed": gated_delta_fixed,
        "gated_delta_vs_safe": gated_delta_safe,
        "fixed_constraint_failed": bool(fixed_summary[str(seed)]["fixed_constraint_failed"]),
        "safe_constraint_failed": bool(safe_row["safe_constraint_failed"]),
        "gated_constraint_failed": failed,
        "gated_failure_reasons": failure_reasons,
        "epoch_integrity": count_epochs(groot / "train" / "results.csv"),
        "train_success": bool(g_payload.get("train", {}).get("success")),
        "val_success": bool(g_payload.get("val", {}).get("success")),
        "gated_events": events_summary,
        "gated_event_details": events,
        "fallback_strict_noop_after_epoch": strict_noop_after_fallback(groot, fallback_epoch),
        "industrial_samples_augmented": int(online_stats.get("samples_augmented", 0) or 0),
        "router_random_draw_count": int(online_stats.get("router_random_draw_count", 0) or 0),
        "online_aug_ops": online_stats.get("ops", {}),
        "roi_aug_applied": int(roi_stats.get("roi_aug_applied", 0) or 0),
        "roi_affected_classes": roi_stats.get("affected_classes", {}),
        "ok3_active": ok3_active,
        "ok3_active_count": ok3_active_count,
        "ok3_roi_applied": int((roi_stats.get("affected_classes") or {}).get("1", 0) or 0),
        "active_classes": active_classes,
        "train_image_count": int(g_payload.get("summary", {}).get("train_image_count", 0)),
        "fixed_augmented_dataset_generated": bool(g_payload.get("summary", {}).get("fixed_augmented_dataset_generated")),
        "bbox_class_valid": bool(g_payload.get("summary", {}).get("bbox_class_valid")),
        "copy_paste_status": online_stats.get("copy_paste_status"),
        "catf_gated_mode": bool(g_payload.get("catf_gated_mode")),
        "yolo_default_augmentation_enabled": bool(g_payload.get("summary", {}).get("yolo_default_augmentation_enabled")),
        "gated_run_path": str(groot),
    }


def build_summary(rows: list[dict[str, Any]], simulation: dict[str, Any] | None) -> dict[str, Any]:
    failed_count = sum(1 for row in rows if row["gated_constraint_failed"])
    all_epoch_continuous = all(row["epoch_integrity"]["epoch_continuous"] for row in rows)
    ok3_ever_active = any(row["ok3_active"] for row in rows)
    ok3_roi_total = sum(row["ok3_roi_applied"] for row in rows)
    gated_deltas = {key: [row["gated_delta_vs_clean"][key] for row in rows] for key in METRIC_KEYS}
    seed0 = rows[0]
    seed1 = rows[1]
    seed2 = rows[2]
    seed0_fixed_map_gain_retained = (
        seed0["gated_delta_vs_clean"]["map50"] >= seed0["fixed_delta_vs_clean"]["map50"] - 1e-9
        and seed0["gated_delta_vs_clean"]["map50_95"] >= seed0["fixed_delta_vs_clean"]["map50_95"] - 1e-9
    )
    seed1_fixed_gain_retained = (
        seed1["gated_delta_vs_clean"]["recall"] >= seed1["fixed_delta_vs_clean"]["recall"] - 1e-9
        and seed1["gated_delta_vs_clean"]["map50"] >= seed1["fixed_delta_vs_clean"]["map50"] - 1e-9
        and seed1["gated_delta_vs_clean"]["map50_95"] >= seed1["fixed_delta_vs_clean"]["map50_95"] - 1e-9
    )
    seed2_fallback = bool(seed2["gated_events"]["fallback_triggered"])
    simulation_expected_pass = None
    if simulation:
        simulation_expected_pass = bool(simulation.get("summary", {}).get("expected_3_of_3_constraint_pass"))
    recommend_main = (
        failed_count == 0
        and seed0_fixed_map_gain_retained
        and seed1_fixed_gain_retained
        and seed2_fallback
        and not ok3_ever_active
        and ok3_roi_total == 0
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_group": ROOT.name,
        "code_commit_used": git_commit(),
        "seeds": [row["seed"] for row in rows],
        "gated_constraint_failed_count": failed_count,
        "gated_constraint_pass_count": len(rows) - failed_count,
        "achieved_3_of_3": failed_count == 0,
        "all_epoch_continuous": all_epoch_continuous,
        "ok3_ever_active": ok3_ever_active,
        "ok3_roi_applied_total": ok3_roi_total,
        "gated_delta_vs_clean_mean_std": {key: mean_std(values) for key, values in gated_deltas.items()},
        "seed0_fixed_map_gain_retained": seed0_fixed_map_gain_retained,
        "seed1_fixed_gain_retained": seed1_fixed_gain_retained,
        "seed2_fallback_triggered": seed2_fallback,
        "seed2_fallback_epoch": seed2["gated_events"]["first_fallback_epoch"],
        "seed2_constraint_failed_after_fallback": seed2["gated_constraint_failed"],
        "retrospective_expected_3_of_3": simulation_expected_pass,
        "retrospective_actual_mismatch": bool(simulation_expected_pass and failed_count != 0),
        "gated_recommended_as_paper_main_method": recommend_main,
        "paper_position": (
            "CATF-v2-Gated retains fixed CATF-v2 gains on seed0 and seed1, but it is not sufficient as the paper "
            "main method because seed2 still violates the industrial constraints after an epoch10 fallback. The result "
            "shows that strict no-op after fallback is not equivalent to clean fallback unless the damaging tentative "
            "updates are prevented or rewound."
        ),
    }


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# CATF-v2-Gated Multiseed Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Code commit used: `{summary['code_commit_used']}`.",
        "",
        "## Per-Seed Metrics",
        "",
        "| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Safe P | Safe R | Safe mAP50 | Safe mAP50-95 | Gated P | Gated R | Gated mAP50 | Gated mAP50-95 | Gated dP vs clean | Gated dR vs clean | Gated dM50 vs clean | Gated dM95 vs clean | constraint_failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        c = row["clean_metrics"]
        f = row["fixed_catf_v2_metrics"]
        s = row["safe_metrics"]
        g = row["gated_metrics"]
        d = row["gated_delta_vs_clean"]
        lines.append(
            f"| {row['seed']} | {f4(c['precision'])} | {f4(c['recall'])} | {f4(c['map50'])} | {f4(c['map50_95'])} | "
            f"{f4(f['precision'])} | {f4(f['recall'])} | {f4(f['map50'])} | {f4(f['map50_95'])} | "
            f"{f4(s['precision'])} | {f4(s['recall'])} | {f4(s['map50'])} | {f4(s['map50_95'])} | "
            f"{f4(g['precision'])} | {f4(g['recall'])} | {f4(g['map50'])} | {f4(g['map50_95'])} | "
            f"{fd(d['precision'])} | {fd(d['recall'])} | {fd(d['map50'])} | {fd(d['map50_95'])} | "
            f"{str(row['gated_constraint_failed']).lower()} |"
        )
    lines += [
        "",
        "## Constraint Summary",
        "",
        f"- CATF-v2-Gated constraint_failed count: `{summary['gated_constraint_failed_count']}/3`.",
        f"- CATF-v2-Gated achieved 3/3 pass: `{str(summary['achieved_3_of_3']).lower()}`.",
        f"- Retrospective expected 3/3 pass: `{str(summary['retrospective_expected_3_of_3']).lower()}`.",
        f"- Retrospective vs actual mismatch: `{str(summary['retrospective_actual_mismatch']).lower()}`.",
        f"- All Gated runs have results.csv epoch 1..50 continuous: `{str(summary['all_epoch_continuous']).lower()}`.",
        "",
        "## Gated Controller Behavior",
        "",
        "| Seed | fallback | fallback epoch | strict no-op after fallback | industrial samples | router draws | ROI applied | active classes | OK3 active | OK3 ROI | reasons |",
        "|---:|---|---:|---|---:|---:|---:|---|---|---:|---|",
    ]
    for row in rows:
        events = row["gated_events"]
        lines.append(
            f"| {row['seed']} | {str(events['fallback_triggered']).lower()} | {events['first_fallback_epoch']} | "
            f"{str(row['fallback_strict_noop_after_epoch']).lower()} | {row['industrial_samples_augmented']} | "
            f"{row['router_random_draw_count']} | {row['roi_aug_applied']} | `{row['active_classes']}` | "
            f"{str(row['ok3_active']).lower()} | {row['ok3_roi_applied']} | `{events['reasons']}` |"
        )
    lines += [
        "",
        "## Required Answers",
        "",
        f"- Seed0 retained fixed CATF-v2 mAP gain: `{str(summary['seed0_fixed_map_gain_retained']).lower()}`.",
        f"- Seed1 retained fixed CATF-v2 clear gain: `{str(summary['seed1_fixed_gain_retained']).lower()}`.",
        f"- Seed2 triggered fallback: `{str(summary['seed2_fallback_triggered']).lower()}` at epoch `{summary['seed2_fallback_epoch']}`.",
        f"- Seed2 passed constraints after fallback: `{str(not summary['seed2_constraint_failed_after_fallback']).lower()}`.",
        f"- OK3 ever active: `{str(summary['ok3_ever_active']).lower()}`.",
        f"- OK3 ROI applied total: `{summary['ok3_roi_applied_total']}`.",
        f"- CATF-v2-Gated recommended as paper main method: `{str(summary['gated_recommended_as_paper_main_method']).lower()}`.",
        "",
        "## Interpretation",
        "",
        summary["paper_position"],
        "",
        "## Reported Paths",
        "",
    ]
    for row in rows:
        lines.append(f"- Seed {row['seed']} Gated: `{row['gated_run_path']}`")
    return lines


def main() -> None:
    fixed_summary = read_json(FIXED_ROOT / "reports" / "multiseed_catf_v2_fixed_summary.json")["seeds"]
    safe_payload = read_json(SAFE_ROOT / "reports" / "multiseed_catf_v2_safe_summary.json")
    safe_rows_by_seed = {int(row["seed"]): row for row in safe_payload["rows"]}
    simulation_path = FIXED_ROOT / "reports" / "gated_controller_retrospective_simulation.json"
    simulation = read_json(simulation_path) if simulation_path.exists() else None
    rows = [load_seed(seed, fixed_summary, safe_rows_by_seed) for seed in SEEDS]
    summary = build_summary(rows, simulation)
    payload = {"summary": summary, "rows": rows}
    write_json(REPORTS / "multiseed_catf_v2_gated_summary.json", payload)
    write_md(REPORTS / "multiseed_catf_v2_gated_summary.md", build_markdown(summary, rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
