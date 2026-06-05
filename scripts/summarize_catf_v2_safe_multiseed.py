"""Summarize CATF-v2-Safe multiseed validation.

This script is read-only over completed experiment artifacts except for writing
the final summary reports under the CATF-v2-Safe multiseed report directory.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe")
CLEAN_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
FIXED_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed")
SEED2_SOURCE = Path("outputs/experiments/catf_v2_safe_seed2_50ep")
REPORTS = ROOT / "reports"
SEEDS = (0, 1, 2)
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if "val" in payload and isinstance(payload["val"].get("metrics"), dict):
        metrics = payload["val"]["metrics"]
    elif "metrics" in payload:
        metrics = payload["metrics"]
    else:
        metrics = payload
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: float(a[key]) - float(b[key]) for key in METRIC_KEYS}


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


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def safe_root(seed: int) -> Path:
    if seed == 2 and not (ROOT / "seed_2" / "catf_v2_safe" / "reports" / "final_metrics.json").exists():
        return SEED2_SOURCE
    return ROOT / f"seed_{seed}" / "catf_v2_safe"


def fixed_root(seed: int) -> Path:
    if seed == 1:
        return Path("outputs/experiments/catf_v2_fixed_seed1_50ep")
    return FIXED_ROOT / f"seed_{seed}" / "catf_v2"


def clean_root(seed: int) -> Path:
    return CLEAN_ROOT / f"seed_{seed}" / "clean_native_yolo_default"


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


def load_policy_history(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, dict):
        return list(payload.get("history", []))
    return list(payload)


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


def safe_event_summary(events_path: Path) -> dict[str, Any]:
    events = read_json(events_path).get("events", [])
    reasons = Counter()
    actions = Counter()
    first_noop_epoch = None
    baseline_protection = False
    early_abstention = False
    for event in events:
        actions[str(event.get("action", ""))] += 1
        for reason in event.get("reasons", []) or []:
            reasons[str(reason)] += 1
            if "baseline" in str(reason):
                baseline_protection = True
            if "early_abstention" in str(reason):
                early_abstention = True
        if event.get("action") == "no_op_freeze" and first_noop_epoch is None:
            first_noop_epoch = int(event.get("epoch"))
    return {
        "events": events,
        "event_count": len(events),
        "actions": dict(actions),
        "reasons": dict(reasons),
        "baseline_protection_triggered": baseline_protection,
        "early_abstention_triggered": early_abstention,
        "no_op_fallback_triggered": bool(actions.get("no_op_freeze", 0)),
        "first_no_op_epoch": first_noop_epoch,
    }


def ensure_seed2_link() -> None:
    target = ROOT / "seed_2" / "catf_v2_safe"
    reports = target / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": 2,
        "reuse_source": str(SEED2_SOURCE),
        "note": "CATF-v2-Safe seed2 50ep run was completed before this multiseed summary and is reused here.",
        "source_final_metrics": str(SEED2_SOURCE / "reports" / "final_metrics.json"),
        "source_results_csv": str(SEED2_SOURCE / "train" / "results.csv"),
        "source_final_report": str(SEED2_SOURCE / "reports" / "final_report.md"),
    }
    write_json(reports / "seed2_reuse_link.json", payload)
    write_md(
        reports / "seed2_reuse_link.md",
        [
            "# CATF-v2-Safe Seed2 Reuse Link",
            "",
            "CATF-v2-Safe seed2 was completed before this full multiseed summary and is reused.",
            "",
            f"- Source run: `{SEED2_SOURCE}`",
            f"- Final metrics: `{SEED2_SOURCE / 'reports' / 'final_metrics.json'}`",
            f"- Results CSV: `{SEED2_SOURCE / 'train' / 'results.csv'}`",
            f"- Final report: `{SEED2_SOURCE / 'reports' / 'final_report.md'}`",
        ],
    )
    for rel in [
        "reports/final_metrics.json",
        "reports/constraint_scoring.json",
        "reports/online_aug_stats.json",
        "reports/roi_aug_stats.json",
        "reports/safe_controller_events.json",
        "reports/class_policy_history.json",
        "reports/policy_history.json",
        "reports/final_report.md",
        "train/results.csv",
    ]:
        src = SEED2_SOURCE / rel
        dst = target / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def load_seed(seed: int) -> dict[str, Any]:
    sroot = safe_root(seed)
    croot = clean_root(seed)
    froot = fixed_root(seed)

    safe_payload = read_json(sroot / "reports" / "final_metrics.json")
    clean_payload = read_json(croot / "reports" / "clean_native_yolo_default_metrics.json")
    fixed_payload = read_json(froot / "reports" / "final_metrics.json")
    safe_metrics = compact_metrics(safe_payload)
    clean_metrics = compact_metrics(clean_payload)
    fixed_metrics = compact_metrics(fixed_payload)
    safe_delta_clean = delta(safe_metrics, clean_metrics)
    safe_delta_fixed = delta(safe_metrics, fixed_metrics)
    failed, failure_reasons = constraint_failed(safe_delta_clean)

    online_stats = read_json(sroot / "reports" / "online_aug_stats.json")
    roi_stats = read_json(sroot / "reports" / "roi_aug_stats.json")
    safe_events = safe_event_summary(sroot / "reports" / "safe_controller_events.json")
    active_classes, ok3_active, ok3_active_count = class_policy_counts(sroot / "reports" / "class_policy_history.json")
    history = load_policy_history(sroot / "reports" / "policy_history.json")

    fixed_summary = read_json(FIXED_ROOT / "reports" / "multiseed_catf_v2_fixed_summary.json")["seeds"][str(seed)]
    epoch_info = count_epochs(sroot / "train" / "results.csv")

    return {
        "seed": seed,
        "clean_metrics": clean_metrics,
        "fixed_catf_v2_metrics": fixed_metrics,
        "safe_metrics": safe_metrics,
        "safe_delta_vs_clean": safe_delta_clean,
        "safe_delta_vs_fixed": safe_delta_fixed,
        "safe_constraint_failed": failed,
        "safe_failure_reasons": failure_reasons,
        "fixed_constraint_failed": bool(fixed_summary["fixed_constraint_failed"]),
        "fixed_delta_vs_clean": fixed_summary["delta_fixed_vs_clean"],
        "safe_run_path": str(sroot),
        "clean_run_path": str(croot),
        "fixed_run_path": str(froot),
        "epoch_integrity": epoch_info,
        "summary_integrity": safe_payload.get("summary", {}),
        "train_success": bool(safe_payload.get("train", {}).get("success")),
        "val_success": bool(safe_payload.get("val", {}).get("success")),
        "industrial_samples_augmented": int(online_stats.get("samples_augmented", 0)),
        "router_random_draw_count": int(online_stats.get("router_random_draw_count", 0) or 0),
        "online_aug_ops": online_stats.get("ops", {}),
        "roi_aug_applied": int(roi_stats.get("roi_aug_applied", 0)),
        "roi_affected_classes": roi_stats.get("affected_classes", {}),
        "ok3_active": ok3_active,
        "ok3_active_count": ok3_active_count,
        "ok3_roi_applied": int(roi_stats.get("affected_classes", {}).get("1", 0)),
        "active_classes": active_classes,
        "policy_action_counts": dict(Counter(str(item.get("action", "")) for item in history)),
        "safe_events": {
            key: value for key, value in safe_events.items() if key != "events"
        },
        "safe_event_details": safe_events["events"],
        "baseline_protection_triggered": safe_events["baseline_protection_triggered"],
        "early_abstention_triggered": safe_events["early_abstention_triggered"],
        "no_op_fallback_triggered": safe_events["no_op_fallback_triggered"],
        "train_image_count": int(safe_payload.get("summary", {}).get("train_image_count", 0)),
        "fixed_augmented_dataset_generated": bool(safe_payload.get("summary", {}).get("fixed_augmented_dataset_generated")),
        "bbox_class_valid": bool(safe_payload.get("summary", {}).get("bbox_class_valid")),
        "copy_paste_status": online_stats.get("copy_paste_status"),
        "catf_safe_mode": bool(safe_payload.get("catf_safe_mode")),
        "yolo_default_augmentation_enabled": bool(safe_payload.get("summary", {}).get("yolo_default_augmentation_enabled")),
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    safe_failed_count = sum(1 for row in rows if row["safe_constraint_failed"])
    fixed_failed_count = sum(1 for row in rows if row["fixed_constraint_failed"])
    all_epoch_continuous = all(row["epoch_integrity"]["epoch_continuous"] for row in rows)
    ok3_ever_active = any(row["ok3_active"] for row in rows)
    ok3_roi_total = sum(row["ok3_roi_applied"] for row in rows)
    safe_deltas = {key: [row["safe_delta_vs_clean"][key] for row in rows] for key in METRIC_KEYS}
    seed0 = rows[0]
    seed1 = rows[1]
    seed2 = rows[2]
    seed0_fixed_gain_lost = (
        seed0["fixed_delta_vs_clean"]["map50_95"] > 0.01
        and abs(seed0["safe_delta_vs_clean"]["map50_95"]) < 1e-9
    )
    seed1_fixed_gain_retained = (
        seed1["safe_delta_vs_clean"]["recall"] >= seed1["fixed_delta_vs_clean"]["recall"] - 0.01
        and seed1["safe_delta_vs_clean"]["map50_95"] >= seed1["fixed_delta_vs_clean"]["map50_95"] - 0.01
        and not seed1["safe_constraint_failed"]
    )
    seed2_clean_parity = all(abs(seed2["safe_delta_vs_clean"][key]) < 1e-9 for key in METRIC_KEYS)
    recommend_main = (
        safe_failed_count == 0
        and not ok3_ever_active
        and ok3_roi_total == 0
        and seed2_clean_parity
        and seed1_fixed_gain_retained
        and not seed0_fixed_gain_lost
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_group": ROOT.name,
        "code_commit_used": git_commit(),
        "seeds": [row["seed"] for row in rows],
        "safe_constraint_failed_count": safe_failed_count,
        "safe_constraint_pass_count": len(rows) - safe_failed_count,
        "fixed_constraint_failed_count": fixed_failed_count,
        "fixed_constraint_pass_count": len(rows) - fixed_failed_count,
        "achieved_3_of_3": safe_failed_count == 0,
        "all_epoch_continuous": all_epoch_continuous,
        "ok3_ever_active": ok3_ever_active,
        "ok3_roi_applied_total": ok3_roi_total,
        "safe_delta_vs_clean_mean_std": {key: mean_std(values) for key, values in safe_deltas.items()},
        "seed0_fixed_gain_lost_by_safe_abstention": seed0_fixed_gain_lost,
        "seed1_fixed_gain_retained_by_safe": seed1_fixed_gain_retained,
        "seed2_noop_clean_parity": seed2_clean_parity,
        "safe_recommended_as_paper_main_method": recommend_main,
        "paper_claim_if_3_of_3": (
            "CATF-v2-Safe passes all three seeds under the industrial constraints by using an early-abstention "
            "no-op fallback when the feedback signal shows no recall or mAP gain. However, in this validation it "
            "also abstains on seed0 and seed1, eliminating the gains of fixed CATF-v2. It should be positioned as "
            "a conservative safety/protection variant or fallback layer, not as the sole main augmentation method."
            if safe_failed_count == 0
            else "Do not claim CATF-v2-Safe as the final robust variant until all seeds pass constraints."
        ),
    }


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# CATF-v2-Safe Multiseed Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Code commit used: `{summary['code_commit_used']}`.",
        "",
        "## Per-Seed Metrics",
        "",
        "| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Safe P | Safe R | Safe mAP50 | Safe mAP50-95 | Safe dP vs clean | Safe dR vs clean | Safe dM50 vs clean | Safe dM95 vs clean | Safe dP vs fixed | Safe dR vs fixed | Safe dM50 vs fixed | Safe dM95 vs fixed | constraint_failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        c = row["clean_metrics"]
        f = row["fixed_catf_v2_metrics"]
        s = row["safe_metrics"]
        dc = row["safe_delta_vs_clean"]
        df = row["safe_delta_vs_fixed"]
        lines.append(
            f"| {row['seed']} | {f4(c['precision'])} | {f4(c['recall'])} | {f4(c['map50'])} | {f4(c['map50_95'])} | "
            f"{f4(f['precision'])} | {f4(f['recall'])} | {f4(f['map50'])} | {f4(f['map50_95'])} | "
            f"{f4(s['precision'])} | {f4(s['recall'])} | {f4(s['map50'])} | {f4(s['map50_95'])} | "
            f"{fd(dc['precision'])} | {fd(dc['recall'])} | {fd(dc['map50'])} | {fd(dc['map50_95'])} | "
            f"{fd(df['precision'])} | {fd(df['recall'])} | {fd(df['map50'])} | {fd(df['map50_95'])} | "
            f"{str(row['safe_constraint_failed']).lower()} |"
        )
    lines += [
        "",
        "## Constraint Summary",
        "",
        f"- CATF-v2-Safe constraint_failed count: `{summary['safe_constraint_failed_count']}/3`.",
        f"- Fixed CATF-v2 constraint_failed count: `{summary['fixed_constraint_failed_count']}/3`.",
        f"- CATF-v2-Safe achieved 3/3 pass: `{str(summary['achieved_3_of_3']).lower()}`.",
        f"- All Safe runs have results.csv epoch 1..50 continuous: `{str(summary['all_epoch_continuous']).lower()}`.",
        "",
        "## Safe Controller Behavior",
        "",
        "| Seed | baseline protection | early abstention | no-op fallback | first no-op epoch | industrial samples | router draws | ROI applied | active classes | OK3 active | OK3 ROI |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        safe_events = row["safe_events"]
        lines.append(
            f"| {row['seed']} | {str(row['baseline_protection_triggered']).lower()} | "
            f"{str(row['early_abstention_triggered']).lower()} | {str(row['no_op_fallback_triggered']).lower()} | "
            f"{safe_events.get('first_no_op_epoch')} | {row['industrial_samples_augmented']} | "
            f"{row['router_random_draw_count']} | {row['roi_aug_applied']} | `{row['active_classes']}` | "
            f"{str(row['ok3_active']).lower()} | {row['ok3_roi_applied']} |"
        )
    lines += [
        "",
        "## OK3 / No-Aug Safety",
        "",
        f"- OK3 ever active: `{str(summary['ok3_ever_active']).lower()}`.",
        f"- OK3 ROI applied total: `{summary['ok3_roi_applied_total']}`.",
        "- OK2/OK3 remain no-augmentation classes in the CATF-v2 policy matrix.",
        "",
        "## Required Answers",
        "",
        f"- Seed0 retained fixed CATF-v2 gain: `{str(not summary['seed0_fixed_gain_lost_by_safe_abstention']).lower()}`. Safe passed constraints but no-op abstention removed fixed seed0 mAP gains.",
        f"- Seed1 retained fixed CATF-v2 gain: `{str(summary['seed1_fixed_gain_retained_by_safe']).lower()}`.",
        f"- Seed2 no-op fallback and clean parity: `{str(summary['seed2_noop_clean_parity']).lower()}`.",
        f"- Safe recommended as paper main method: `{str(summary['safe_recommended_as_paper_main_method']).lower()}`.",
        "",
        "## Paper Wording",
        "",
        summary["paper_claim_if_3_of_3"],
        "",
        "## Reported Paths",
        "",
    ]
    for row in rows:
        lines.append(f"- Seed {row['seed']} Safe: `{row['safe_run_path']}`")
    return lines


def main() -> None:
    ensure_seed2_link()
    rows = [load_seed(seed) for seed in SEEDS]
    summary = build_summary(rows)
    payload = {"summary": summary, "rows": rows}
    write_json(REPORTS / "multiseed_catf_v2_safe_summary.json", payload)
    write_md(REPORTS / "multiseed_catf_v2_safe_summary.md", build_markdown(summary, rows))
    print(f"Wrote {REPORTS / 'multiseed_catf_v2_safe_summary.json'}")
    print(f"Wrote {REPORTS / 'multiseed_catf_v2_safe_summary.md'}")


if __name__ == "__main__":
    main()
