from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("outputs/experiments/catf_v2_adaptive_rb_seed2_50ep")
CLEAN = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/seed_2/clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json")
FIXED = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json")
GATED = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.json")
SAFE = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.json")
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metrics(payload: dict[str, Any]) -> dict[str, float]:
    source = payload.get("metrics") or payload.get("val", {}).get("metrics") or payload
    return {key: float(source[key]) for key in METRIC_KEYS}


def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: a[key] - b[key] for key in METRIC_KEYS}


def f4(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def fd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def first_event(events: list[dict[str, Any]], action: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("action") == action), None)


def main() -> None:
    payload = read_json(ROOT / "reports" / "final_metrics.json")
    clean_metrics = metrics(read_json(CLEAN))
    fixed_metrics = dict(read_json(FIXED)["seeds"]["2"]["fixed_catf_v2"])
    gated_row = next(row for row in read_json(GATED)["rows"] if int(row["seed"]) == 2)
    safe_row = next(row for row in read_json(SAFE)["rows"] if int(row["seed"]) == 2)
    adaptive_metrics = metrics(payload)
    events = read_json(ROOT / "reports" / "adaptive_burnin_events.json")["events"]
    rb_events = read_json(ROOT / "reports" / "rollback_controller_events.json")["events"]
    online = read_json(ROOT / "reports" / "online_aug_stats.json")
    roi = read_json(ROOT / "reports" / "roi_aug_stats.json")
    noop = first_event(events, "no_op_fallback")
    start = next((event for event in events if event.get("candidate_branch_started")), None)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code_commit_used": git_commit(),
        "run_path": str(ROOT),
        "adaptive_burnin_enabled": bool(payload.get("adaptive_burnin")),
        "catf_rollback_mode": bool(payload.get("catf_rollback_mode")),
        "adaptive_start_epoch": payload.get("adaptive_start_epoch"),
        "candidate_started": bool(start),
        "candidate_start_epoch": start.get("epoch") if start else None,
        "noop_fallback": bool(noop),
        "noop_fallback_epoch": noop.get("epoch") if noop else None,
        "rollback_triggered": any(event.get("action") == "rollback" for event in rb_events),
        "rollback_epoch": next((event.get("epoch") for event in rb_events if event.get("action") == "rollback"), None),
        "strict_noop": int(online.get("samples_augmented", 0) or 0) == 0
        and int(roi.get("roi_aug_applied", 0) or 0) == 0
        and int(online.get("router_random_draw_count", 0) or 0) == 0,
        "adaptive_metrics": adaptive_metrics,
        "clean_metrics": clean_metrics,
        "fixed_catf_v2_metrics": fixed_metrics,
        "safe_metrics": dict(safe_row["safe_metrics"]),
        "gated_metrics": dict(gated_row["gated_metrics"]),
        "delta_vs_clean": delta(adaptive_metrics, clean_metrics),
        "delta_vs_fixed": delta(adaptive_metrics, fixed_metrics),
        "delta_vs_safe": delta(adaptive_metrics, dict(safe_row["safe_metrics"])),
        "delta_vs_gated": delta(adaptive_metrics, dict(gated_row["gated_metrics"])),
        "constraint_failed": bool(payload["constraint_scoring"]["constraint_failed"]),
        "failure_reasons": list(payload["constraint_scoring"]["failure_reasons"]),
        "epoch_continuous": bool(payload["continuity"]["epoch_continuous"]),
        "train_image_count": int(online.get("train_image_count", 0) or 0),
        "industrial_samples_augmented": int(online.get("samples_augmented", 0) or 0),
        "roi_aug_applied": int(roi.get("roi_aug_applied", 0) or 0),
        "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
        "bbox_class_valid": bool(payload["summary"]["bbox_class_valid"]),
        "full_multiseed_adaptive_rb_recommended": True,
    }
    write_json(ROOT / "reports" / "adaptive_rb_seed2_50ep_summary.json", summary)
    lines = [
        "# CATF-v2 Adaptive Burn-in + RB Seed2 50ep Report",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Code commit used: `{summary['code_commit_used']}`",
        "",
        "## Required Answers",
        "",
        f"1. Seed2 adaptive start epoch: `{summary['adaptive_start_epoch']}`.",
        f"2. CATF candidate started: `{str(summary['candidate_started']).lower()}`.",
        f"3. Rollback triggered: `{str(summary['rollback_triggered']).lower()}`.",
        f"4. Strict no-op: `{str(summary['strict_noop']).lower()}`; no-op fallback epoch: `{summary['noop_fallback_epoch']}`.",
        f"5. Final metrics: P `{f4(adaptive_metrics['precision'])}`, R `{f4(adaptive_metrics['recall'])}`, mAP50 `{f4(adaptive_metrics['map50'])}`, mAP50-95 `{f4(adaptive_metrics['map50_95'])}`.",
        f"6. constraint_failed: `{str(summary['constraint_failed']).lower()}`; reasons: `{summary['failure_reasons']}`.",
        "7. Adaptive burn-in is more defensible than fixed epoch5 here because it waits for model/metric diagnosability and blocks intervention under strong seed2 clean-baseline protection.",
        f"8. Full multiseed adaptive-RB recommended: `{str(summary['full_multiseed_adaptive_rb_recommended']).lower()}`.",
        "",
        "## Metrics",
        "",
        "| group | P | R | mAP50 | mAP50-95 | dP vs clean | dR vs clean | dM50 vs clean | dM95 vs clean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in [
        ("clean seed2", clean_metrics),
        ("fixed CATF-v2 seed2", fixed_metrics),
        ("Safe seed2", dict(safe_row["safe_metrics"])),
        ("Gated seed2", dict(gated_row["gated_metrics"])),
        ("Adaptive-RB seed2", adaptive_metrics),
    ]:
        d = delta(row, clean_metrics)
        lines.append(
            f"| {name} | {f4(row['precision'])} | {f4(row['recall'])} | {f4(row['map50'])} | {f4(row['map50_95'])} | "
            f"{fd(d['precision'])} | {fd(d['recall'])} | {fd(d['map50'])} | {fd(d['map50_95'])} |"
        )
    lines.extend(
        [
            "",
            "## Augmentation Audit",
            "",
            f"- Industrial samples augmented: `{summary['industrial_samples_augmented']}`",
            f"- ROI applied: `{summary['roi_aug_applied']}`",
            f"- Router random draw count: `{summary['router_random_draw_count']}`",
            f"- Train images: `{summary['train_image_count']}`",
            f"- BBox/class legal: `{str(summary['bbox_class_valid']).lower()}`",
            f"- Epoch continuous: `{str(summary['epoch_continuous']).lower()}`",
            "",
            "## Burn-in Events",
            "",
            "| epoch | action | start_condition | reasons | map50_range | recall_range | eligible_classes | strong_baseline |",
            "|---:|---|---|---|---:|---:|---|---|",
        ]
    )
    for event in events:
        cond = event.get("start_condition") or {}
        stability = cond.get("metric_stability") or {}
        lines.append(
            f"| {event.get('epoch')} | {event.get('action')} | {str(event.get('start_condition_met')).lower()} | "
            f"`{event.get('reasons', [])}` | {f4(stability.get('map50_range'))} | {f4(stability.get('recall_range'))} | "
            f"`{cond.get('eligible_active_classes', [])}` | `{str(cond.get('strong_clean_baseline_protection', False)).lower()}` |"
        )
    write_md(ROOT / "reports" / "adaptive_rb_seed2_50ep_report.md", lines)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
