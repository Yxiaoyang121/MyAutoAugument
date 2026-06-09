from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT / "outputs/experiments/multiseed_cp_catf_paper_mode"
SMOKE_ROOT = PROJECT_ROOT / "outputs/experiments/cp_catf_paper_mode_10ep_smoke"
DEV_SUMMARY = PROJECT_ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf/reports/multiseed_cp_catf_summary.json"
SPLIT_REPORT = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/reports/probe_split_report.json"


def main() -> None:
    args = parse_args()
    if args.smoke:
        write_smoke_report(Path(args.smoke_root))
    if args.summary:
        write_multiseed_summary(Path(args.paper_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CP-CATF paper-mode experiments.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--smoke-root", default=str(SMOKE_ROOT))
    parser.add_argument("--paper-root", default=str(PAPER_ROOT))
    return parser.parse_args()


def write_smoke_report(smoke_root: Path) -> None:
    payload = read_json(smoke_root / "reports" / "final_metrics.json")
    events = payload.get("causal_probe_events", [])
    first_event = events[0] if events else {}
    audit = payload.get("paper_probe_leakage_audit", {})
    stats = payload.get("online_aug_stats", {})
    roi = payload.get("roi_aug_stats", {})
    report = {
        "paper_mode_enabled": bool(payload.get("paper_probe_mode") or payload.get("summary", {}).get("paper_probe_mode")),
        "train_uses_train_core": int(stats.get("train_image_count", 0) or 0) == int(audit.get("train_core_image_count", -1)),
        "probe_uses_probe_split": first_event.get("policy_selection_source") == "probe_split",
        "final_val_used_for_policy_selection": bool(first_event.get("final_val_used_for_policy_selection", True)),
        "causal_probe_executed": bool(events),
        "candidate_decision": first_event.get("selected_candidate_policy_id"),
        "candidate_action": first_event.get("selected_candidate_action"),
        "strict_image_noop": not bool(first_event.get("image_modification_allowed", False)),
        "bbox_class_valid": bool(payload.get("summary", {}).get("bbox_class_valid", False)),
        "catf_random_draw_count": int(stats.get("router_random_draw_count", 0) or 0),
        "industrial_samples_augmented": int(stats.get("samples_augmented", 0) or 0),
        "roi_applied": int(roi.get("roi_aug_applied", 0) or 0),
        "leakage_audit": audit,
        "recommend_enter_50ep_multiseed": bool(payload.get("train", {}).get("success") and payload.get("val", {}).get("success") and not first_event.get("final_val_used_for_policy_selection", True)),
    }
    reports_dir = smoke_root / "reports"
    write_json(reports_dir / "paper_mode_smoke_report.json", report)
    lines = [
        "# CP-CATF Paper-Mode 10ep Smoke Report",
        "",
        "## Answers",
        "",
        f"- Paper-mode enabled: `{str(report['paper_mode_enabled']).lower()}`",
        f"- Train uses train_core only: `{str(report['train_uses_train_core']).lower()}`",
        f"- Probe uses probe split: `{str(report['probe_uses_probe_split']).lower()}`",
        f"- Final val used for policy selection: `{str(report['final_val_used_for_policy_selection']).lower()}`",
        f"- Causal probe executed: `{str(report['causal_probe_executed']).lower()}`",
        f"- Candidate decision: `{report['candidate_decision']}`",
        f"- Candidate action: `{report['candidate_action']}`",
        f"- Strict image no-op: `{str(report['strict_image_noop']).lower()}`",
        f"- BBox/class legal: `{str(report['bbox_class_valid']).lower()}`",
        f"- CATF random draw count: `{report['catf_random_draw_count']}`",
        f"- Industrial samples augmented: `{report['industrial_samples_augmented']}`",
        f"- ROI applied: `{report['roi_applied']}`",
        f"- Recommend 50ep multiseed: `{str(report['recommend_enter_50ep_multiseed']).lower()}`",
    ]
    (reports_dir / "paper_mode_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_multiseed_summary(root: Path) -> None:
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    split = read_json(SPLIT_REPORT) if SPLIT_REPORT.exists() else {}
    dev = read_json(DEV_SUMMARY) if DEV_SUMMARY.exists() else {}
    seeds = [0, 1, 2]
    rows = {}
    for seed in seeds:
        clean = load_run(root / f"clean_seed_{seed}")
        cp = load_run(root / f"cp_catf_seed_{seed}")
        delta = metric_delta(cp["metrics"], clean["metrics"])
        constraint = build_constraint(cp["metrics"], clean["metrics"])
        rows[str(seed)] = {
            "clean": clean,
            "cp_catf": cp,
            "delta_vs_clean": delta,
            "constraint_failed": constraint["constraint_failed"],
            "constraint_reasons": constraint["failure_reasons"],
        }
    clean_avg = average_metrics([rows[str(seed)]["clean"]["metrics"] for seed in seeds])
    cp_avg = average_metrics([rows[str(seed)]["cp_catf"]["metrics"] for seed in seeds])
    avg_delta = metric_delta(cp_avg, clean_avg)
    leakage_free = all(
        row["cp_catf"].get("leakage_audit", {}).get("probe_final_val_overlap_count", 1) == 0
        and row["cp_catf"].get("leakage_audit", {}).get("train_core_probe_overlap_count", 1) == 0
        and row["cp_catf"].get("leakage_audit", {}).get("train_core_final_val_overlap_count", 1) == 0
        and all(not event.get("final_val_used_for_policy_selection", True) for event in row["cp_catf"].get("causal_probe_events", []))
        for row in rows.values()
    )
    result = {
        "split_summary": split.get("summary", {}),
        "seeds": rows,
        "clean_average": clean_avg,
        "cp_catf_average": cp_avg,
        "average_delta_vs_clean": avg_delta,
        "constraint_failed_count": sum(1 for row in rows.values() if row["constraint_failed"]),
        "pass_count": sum(1 for row in rows.values() if not row["constraint_failed"]),
        "three_of_three_pass": all(not row["constraint_failed"] for row in rows.values()),
        "final_val_leakage_detected": not leakage_free,
        "development_mode_reference": dev.get("average_delta_vs_clean") or dev.get("average_delta") or dev,
        "recommend_as_paper_main_result": leakage_free and all(not row["constraint_failed"] for row in rows.values()) and avg_delta.get("map50_95", 0.0) > 0,
    }
    write_json(reports_dir / "multiseed_cp_catf_paper_mode_summary.json", result)
    (reports_dir / "multiseed_cp_catf_paper_mode_summary.md").write_text(build_summary_markdown(result), encoding="utf-8")
    for seed in seeds:
        write_seed_compare(root / f"cp_catf_seed_{seed}", rows[str(seed)])


def load_run(run_dir: Path) -> dict[str, Any]:
    payload = read_json(run_dir / "reports" / "final_metrics.json")
    stats = read_json(run_dir / "reports" / "online_aug_stats.json") if (run_dir / "reports" / "online_aug_stats.json").exists() else {}
    roi = read_json(run_dir / "reports" / "roi_aug_stats.json") if (run_dir / "reports" / "roi_aug_stats.json").exists() else {}
    events = read_json(run_dir / "reports" / "causal_probe_events.json").get("events", []) if (run_dir / "reports" / "causal_probe_events.json").exists() else []
    audit = read_json(run_dir / "reports" / "paper_probe_leakage_audit.json") if (run_dir / "reports" / "paper_probe_leakage_audit.json").exists() else {}
    decision = read_json(run_dir / "reports" / "causal_probe_decisions_used.json") if (run_dir / "reports" / "causal_probe_decisions_used.json").exists() else {}
    selected = decision.get("selected_candidate") or {}
    return {
        "run_dir": str(run_dir),
        "metrics": compact_metrics(payload.get("val", {}).get("metrics", {})),
        "train_success": payload.get("train", {}).get("success"),
        "val_success": payload.get("val", {}).get("success"),
        "candidate_decisions": [event.get("selected_candidate_policy_id") for event in events],
        "candidate_actions": [event.get("selected_candidate_action") for event in events],
        "causal_scores": [float((selected.get("decision") or {}).get("causal_score", 0.0) or 0.0)] if selected else [],
        "final_selected_candidate": decision.get("selected_candidate_policy_id"),
        "final_selected_action": decision.get("selected_candidate_action"),
        "image_aug_accepted": any(event.get("image_modification_allowed") for event in events),
        "sampler_only": any(event.get("selected_candidate_action") == "sampler_only" for event in events),
        "roi_applied": int(roi.get("roi_aug_applied", 0) or 0),
        "industrial_samples_augmented": int(stats.get("samples_augmented", 0) or 0),
        "router_random_draw_count": int(stats.get("router_random_draw_count", 0) or 0),
        "online_aug_stats": stats,
        "roi_aug_stats": roi,
        "causal_probe_events": events,
        "leakage_audit": audit,
    }


def write_seed_compare(run_dir: Path, row: dict[str, Any]) -> None:
    cp = row["cp_catf"]
    clean = row["clean"]
    lines = [
        "# Compare With Clean Paper Baseline",
        "",
        "| metric | clean paper | CP-CATF paper | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in METRIC_KEYS:
        lines.append(f"| {key} | {fmt(clean['metrics'].get(key))} | {fmt(cp['metrics'].get(key))} | {fmt(row['delta_vs_clean'].get(key))} |")
    lines.extend(
        [
            "",
            f"- Constraint failed: `{str(row['constraint_failed']).lower()}`",
            f"- Candidate decisions: `{cp['candidate_decisions']}`",
            f"- ROI applied: `{cp['roi_applied']}`",
            f"- Industrial samples augmented: `{cp['industrial_samples_augmented']}`",
            f"- Router random draw count: `{cp['router_random_draw_count']}`",
        ]
    )
    (run_dir / "reports" / "compare_with_clean_and_fixed_catf_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Multiseed CP-CATF Paper-Mode Summary",
        "",
        "## Split",
        "",
        f"- Original train images: `{result['split_summary'].get('original_train_images')}`",
        f"- Train core images: `{result['split_summary'].get('train_core_images')}`",
        f"- Probe images: `{result['split_summary'].get('probe_images')}`",
        f"- Final val images: `{result['split_summary'].get('val_images')}`",
        f"- No overlap: `{str(result['split_summary'].get('train_core_probe_val_no_overlap')).lower()}`",
        "",
        "## Metrics",
        "",
        "| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | CP P | CP R | CP mAP50 | CP mAP50-95 | candidate | constraint_failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for seed, row in sorted(result["seeds"].items(), key=lambda item: int(item[0])):
        clean = row["clean"]["metrics"]
        cp = row["cp_catf"]["metrics"]
        candidate = row["cp_catf"].get("candidate_decisions", [])
        lines.append(
            f"| {seed} | {fmt(clean.get('precision'))} | {fmt(clean.get('recall'))} | {fmt(clean.get('map50'))} | {fmt(clean.get('map50_95'))} | "
            f"{fmt(cp.get('precision'))} | {fmt(cp.get('recall'))} | {fmt(cp.get('map50'))} | {fmt(cp.get('map50_95'))} | "
            f"`{candidate}` | `{str(row['constraint_failed']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "## Average",
            "",
            f"- Clean average: `{result['clean_average']}`",
            f"- CP-CATF average: `{result['cp_catf_average']}`",
            f"- Average delta vs clean: `{result['average_delta_vs_clean']}`",
            f"- Constraint failed count: `{result['constraint_failed_count']}/3`",
            f"- 3/3 pass: `{str(result['three_of_three_pass']).lower()}`",
            f"- Final val leakage detected: `{str(result['final_val_leakage_detected']).lower()}`",
            f"- Recommend as paper main result: `{str(result['recommend_as_paper_main_result']).lower()}`",
            "",
            "## Notes",
            "",
            "- Paper-mode uses `train_core` for training and `probe` for policy selection.",
            "- Final validation is kept for final metrics only.",
            "- If gains drop relative to development-mode, the likely causes are smaller `train_core`, smaller probe evidence, and less optimistic probe diagnostics.",
        ]
    )
    return "\n".join(lines) + "\n"


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    return {key: metrics.get(key) for key in ["images", "instances", *METRIC_KEYS]}


def metric_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        ref = baseline.get(key)
        out[key] = float(value) - float(ref) if value is not None and ref is not None else 0.0
    return out


def build_constraint(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    delta = metric_delta(metrics, baseline)
    failures = []
    if delta["precision"] < -0.01:
        failures.append("precision_drop_gt_0.01")
    if delta["map50"] < -0.01:
        failures.append("map50_drop_gt_0.01")
    if delta["map50_95"] < -0.01:
        failures.append("map50_95_drop_gt_0.01")
    return {"constraint_failed": bool(failures), "failure_reasons": failures}


def average_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: mean(float(row.get(key, 0.0) or 0.0) for row in rows) for key in METRIC_KEYS}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
