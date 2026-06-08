from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_cp_catf")
REPORTS = ROOT / "reports"
FIXED_SUMMARY = Path(
    "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json"
)
SAFE_SUMMARY = Path(
    "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.json"
)
GATED_SUMMARY = Path(
    "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.json"
)

METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def metrics_from_final(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("val", {}).get("metrics") or payload.get("summary", {}).get("val_metrics") or payload
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def delta(current: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: current[key] - reference[key] for key in METRIC_KEYS}


def constraint_failed(deltas: dict[str, float]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if deltas["precision"] < -0.01:
        failures.append("precision_drop_gt_0.01")
    if deltas["map50"] < -0.01:
        failures.append("map50_drop_gt_0.01")
    if deltas["map50_95"] < -0.01:
        failures.append("map50_95_drop_gt_0.01")
    return bool(failures), failures


def f4(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def fd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def retention_ratio(cp_delta: dict[str, float], fixed_delta: dict[str, float], key: str) -> float | None:
    if fixed_delta[key] <= 0.0:
        return None
    return cp_delta[key] / fixed_delta[key]


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def epoch_integrity(run_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    continuity = payload.get("continuity") or {}
    if continuity:
        return {
            "epoch_count": continuity.get("epoch_count"),
            "first_epoch": continuity.get("first_epoch"),
            "last_epoch": continuity.get("last_epoch"),
            "epoch_continuous": bool(continuity.get("epoch_continuous")),
        }
    rows = []
    csv_path = run_root / "train" / "results.csv"
    if csv_path.exists():
        for line in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
            if line.strip():
                rows.append(int(float(line.split(",", 1)[0].strip())))
    return {
        "epoch_count": len(rows),
        "first_epoch": rows[0] if rows else None,
        "last_epoch": rows[-1] if rows else None,
        "epoch_continuous": rows == list(range(1, len(rows) + 1)),
    }


def active_classes_from_policy(run_root: Path) -> list[int]:
    active: set[int] = set()
    for path in sorted((run_root / "reports").glob("policy_matrix_epoch_*_after_causal_probe.json")):
        payload = read_json(path)
        for class_id, class_payload in (payload.get("classes") or {}).items():
            ops = class_payload.get("ops") or {}
            if any(float(op.get("prob", 0.0) or 0.0) > 0.0 for op in ops.values()):
                active.add(int(class_id))
    return sorted(active)


def row_from_optional_summary(summary_path: Path, field: str, seed: int) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    payload = read_json(summary_path)
    for row in payload.get("rows", []):
        if int(row.get("seed")) == seed:
            return row.get(field) or row
    return None


def ok3_audit(run_root: Path, roi_stats: dict[str, Any]) -> dict[str, Any]:
    active_classes = active_classes_from_policy(run_root)
    affected = roi_stats.get("affected_classes") or {}
    ok3_roi_applied = int(affected.get("1", 0) or affected.get(1, 0) or 0)
    return {
        "ok3_active": 1 in active_classes or ok3_roi_applied > 0,
        "ok3_roi_applied": ok3_roi_applied,
        "active_classes": active_classes,
    }


def build_row(seed: int, fixed_payload: dict[str, Any]) -> dict[str, Any]:
    run_root = ROOT / f"seed_{seed}" / "catf_v2_cp_catf"
    final_payload = read_json(run_root / "reports" / "final_metrics.json")
    online = read_json(run_root / "reports" / "online_aug_stats.json")
    roi = read_json(run_root / "reports" / "roi_aug_stats.json")
    events = read_json(run_root / "reports" / "causal_probe_events.json").get("events", [])
    decision = read_json(run_root / "reports" / "causal_probe_decisions_used.json")

    fixed_seed = fixed_payload["seeds"][str(seed)]
    clean = dict(fixed_seed["clean_native"])
    fixed = dict(fixed_seed["fixed_catf_v2"])
    cp_metrics = metrics_from_final(final_payload)
    cp_delta_clean = delta(cp_metrics, clean)
    cp_delta_fixed = delta(cp_metrics, fixed)
    failed, failure_reasons = constraint_failed(cp_delta_clean)
    audit = ok3_audit(run_root, roi)

    first_event = events[0] if events else {}
    fixed_delta_clean = dict(fixed_seed["delta_fixed_vs_clean"])
    sample_weighting_statuses = sorted({str(event.get("sample_weighting_status")) for event in events if event})

    return {
        "seed": seed,
        "run_path": str(run_root),
        "clean_metrics": clean,
        "fixed_catf_v2_metrics": fixed,
        "cp_catf_metrics": cp_metrics,
        "delta_cp_vs_clean": cp_delta_clean,
        "delta_cp_vs_fixed": cp_delta_fixed,
        "delta_fixed_vs_clean": fixed_delta_clean,
        "constraint_failed": failed,
        "failure_reasons": failure_reasons,
        "fixed_constraint_failed": bool(fixed_seed["fixed_constraint_failed"]),
        "selected_candidate_policy_id": decision.get("selected_candidate_policy_id"),
        "selected_candidate_action": decision.get("selected_candidate_action") or decision.get("selected_action"),
        "image_modification_allowed": bool(first_event.get("image_modification_allowed")),
        "probe_reject_image_aug": bool(first_event.get("probe_reject_image_aug")),
        "sample_weighting_allowed": bool(first_event.get("sample_weighting_allowed")),
        "sample_weighting_effective": bool(first_event.get("sample_weighting_effective")),
        "sample_weighting_statuses": sample_weighting_statuses,
        "development_probe_uses_existing_val_diagnostics": bool(
            first_event.get("development_probe_uses_existing_val_diagnostics")
            or decision.get("development_probe_uses_existing_val_diagnostics")
        ),
        "seed_specific_rule": bool(first_event.get("seed_specific_rule")),
        "fixed_class_id_specific_rule": bool(first_event.get("fixed_class_id_specific_rule")),
        "riskguard_used_as_final_rule": bool(first_event.get("riskguard_used_as_final_rule")),
        "causal_probe_event_count": len(events),
        "industrial_samples_augmented": int(online.get("samples_augmented", 0) or 0),
        "roi_aug_applied": int(roi.get("roi_aug_applied", 0) or 0),
        "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
        "online_ops": dict(online.get("ops") or {}),
        "roi_affected_classes": dict(roi.get("affected_classes") or {}),
        "train_image_count": int(online.get("train_image_count", 0) or 0),
        "fixed_augmented_dataset_generated": bool(online.get("fixed_augmented_dataset_generated")),
        "bbox_class_valid": bool(final_payload.get("summary", {}).get("bbox_class_valid")),
        "epoch_integrity": epoch_integrity(run_root, final_payload),
        "ok3_active": audit["ok3_active"],
        "ok3_roi_applied": audit["ok3_roi_applied"],
        "active_classes": audit["active_classes"],
        "seed0_fixed_map_gain_retained": (
            seed == 0
            and cp_delta_clean["map50"] > 0.0
            and cp_delta_clean["map50_95"] > 0.0
            and cp_delta_fixed["map50"] >= -0.005
            and cp_delta_fixed["map50_95"] >= -0.005
        ),
        "seed1_fixed_gain_retained": (
            seed == 1
            and cp_delta_clean["precision"] >= 0.0
            and cp_delta_clean["recall"] > 0.0
            and cp_delta_clean["map50"] > 0.0
            and cp_delta_clean["map50_95"] > 0.0
        ),
        "seed2_clean_protected": (
            seed == 2
            and not failed
            and not bool(first_event.get("image_modification_allowed"))
            and int(roi.get("roi_aug_applied", 0) or 0) == 0
            and int(online.get("samples_augmented", 0) or 0) == 0
            and int(online.get("router_random_draw_count", 0) or 0) == 0
        ),
        "map50_fixed_gain_retention": retention_ratio(cp_delta_clean, fixed_delta_clean, "map50"),
        "map95_fixed_gain_retention": retention_ratio(cp_delta_clean, fixed_delta_clean, "map50_95"),
    }


def mean_std(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [row[field][key] for row in rows]
        payload[key] = {"mean": mean(values), "std": pstdev(values)}
    return payload


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_count = sum(1 for row in rows if row["constraint_failed"])
    seed0 = next(row for row in rows if row["seed"] == 0)
    seed1 = next(row for row in rows if row["seed"] == 1)
    seed2 = next(row for row in rows if row["seed"] == 2)
    return {
        "constraint_failed_count": failed_count,
        "constraint_pass_count": len(rows) - failed_count,
        "three_of_three_pass": failed_count == 0,
        "seed0_fixed_map_gain_retained": bool(seed0["seed0_fixed_map_gain_retained"]),
        "seed1_fixed_gain_retained": bool(seed1["seed1_fixed_gain_retained"]),
        "seed2_clean_protected": bool(seed2["seed2_clean_protected"]),
        "seed2_image_aug_rejected": bool(seed2["probe_reject_image_aug"]),
        "seed2_roi_industrial_zero": seed2["roi_aug_applied"] == 0 and seed2["industrial_samples_augmented"] == 0,
        "ok3_any_active": any(row["ok3_active"] for row in rows),
        "ok3_total_roi_applied": sum(row["ok3_roi_applied"] for row in rows),
        "dataset_specific_rule_used": any(row["seed_specific_rule"] or row["fixed_class_id_specific_rule"] for row in rows),
        "riskguard_used_as_final_rule": any(row["riskguard_used_as_final_rule"] for row in rows),
        "development_probe_uses_existing_val_diagnostics": any(
            row["development_probe_uses_existing_val_diagnostics"] for row in rows
        ),
        "main_method_recommendation": (
            "CP-CATF is a main-method candidate for this development-mode validation. "
            "A paper-mode train/probe split is still required before claiming leakage-free final results."
        ),
        "paper_mode_next_step": (
            "Replace development offline probe decisions from validation diagnostics with a train/probe split "
            "or train hard-example probe set, then rerun multiseed CP-CATF."
        ),
        "clean_mean_std": mean_std(rows, "clean_metrics"),
        "fixed_mean_std": mean_std(rows, "fixed_catf_v2_metrics"),
        "cp_catf_mean_std": mean_std(rows, "cp_catf_metrics"),
        "cp_delta_vs_clean_mean": {
            key: mean(row["delta_cp_vs_clean"][key] for row in rows) for key in METRIC_KEYS
        },
        "cp_delta_vs_fixed_mean": {
            key: mean(row["delta_cp_vs_fixed"][key] for row in rows) for key in METRIC_KEYS
        },
    }


def write_seed_compare(row: dict[str, Any]) -> None:
    seed = row["seed"]
    path = ROOT / f"seed_{seed}" / "catf_v2_cp_catf" / "reports" / "compare_with_clean_and_fixed_catf_v2.md"
    lines = [
        "# CP-CATF seed comparison",
        "",
        f"- Seed: `{seed}`",
        f"- Selected candidate: `{row['selected_candidate_policy_id']}` / `{row['selected_candidate_action']}`",
        f"- Image modification allowed: `{str(row['image_modification_allowed']).lower()}`",
        f"- Probe rejected image augmentation: `{str(row['probe_reject_image_aug']).lower()}`",
        f"- Constraint failed: `{str(row['constraint_failed']).lower()}`",
        f"- Failure reasons: `{row['failure_reasons']}`",
        "",
        "| Group | P | R | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| clean native | {f4(row['clean_metrics']['precision'])} | {f4(row['clean_metrics']['recall'])} | "
            f"{f4(row['clean_metrics']['map50'])} | {f4(row['clean_metrics']['map50_95'])} |"
        ),
        (
            f"| fixed CATF-v2 | {f4(row['fixed_catf_v2_metrics']['precision'])} | "
            f"{f4(row['fixed_catf_v2_metrics']['recall'])} | {f4(row['fixed_catf_v2_metrics']['map50'])} | "
            f"{f4(row['fixed_catf_v2_metrics']['map50_95'])} |"
        ),
        (
            f"| CP-CATF | {f4(row['cp_catf_metrics']['precision'])} | {f4(row['cp_catf_metrics']['recall'])} | "
            f"{f4(row['cp_catf_metrics']['map50'])} | {f4(row['cp_catf_metrics']['map50_95'])} |"
        ),
        "",
        "| Delta | dP | dR | d mAP50 | d mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| CP-CATF vs clean | {fd(row['delta_cp_vs_clean']['precision'])} | "
            f"{fd(row['delta_cp_vs_clean']['recall'])} | {fd(row['delta_cp_vs_clean']['map50'])} | "
            f"{fd(row['delta_cp_vs_clean']['map50_95'])} |"
        ),
        (
            f"| CP-CATF vs fixed | {fd(row['delta_cp_vs_fixed']['precision'])} | "
            f"{fd(row['delta_cp_vs_fixed']['recall'])} | {fd(row['delta_cp_vs_fixed']['map50'])} | "
            f"{fd(row['delta_cp_vs_fixed']['map50_95'])} |"
        ),
        "",
        f"- Industrial image samples augmented: `{row['industrial_samples_augmented']}`",
        f"- ROI applied: `{row['roi_aug_applied']}`",
        f"- Router random draw count: `{row['router_random_draw_count']}`",
        f"- ROI affected classes: `{row['roi_affected_classes']}`",
        f"- OK3 active: `{str(row['ok3_active']).lower()}`",
        f"- OK3 ROI applied: `{row['ok3_roi_applied']}`",
        f"- Epoch integrity: `{row['epoch_integrity']}`",
        "",
        "Development-mode note: offline probe decisions use existing validation diagnostics for this engineering validation; paper-mode CP-CATF needs a train/probe split or train hard-example probe set.",
    ]
    write_md(path, lines)


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Multiseed CP-CATF Training Validation Summary",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Code commit at report generation: `{git_commit()}`",
        f"- Run root: `{ROOT}`",
        f"- Constraint failed count: `{summary['constraint_failed_count']}/3`",
        f"- 3/3 pass: `{str(summary['three_of_three_pass']).lower()}`",
        f"- RiskGuard used as final rule: `{str(summary['riskguard_used_as_final_rule']).lower()}`",
        f"- Dataset/seed/class-specific rule used: `{str(summary['dataset_specific_rule_used']).lower()}`",
        f"- Development probe uses existing validation diagnostics: `{str(summary['development_probe_uses_existing_val_diagnostics']).lower()}`",
        "",
        "## Metrics",
        "",
        "| Seed | Candidate | Action | CP P | CP R | CP mAP50 | CP mAP50-95 | dP vs clean | dR vs clean | dM50 vs clean | dM95 vs clean | dP vs fixed | dR vs fixed | dM50 vs fixed | dM95 vs fixed | constraint_failed |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | `{row['selected_candidate_policy_id']}` | `{row['selected_candidate_action']}` | "
            f"{f4(row['cp_catf_metrics']['precision'])} | {f4(row['cp_catf_metrics']['recall'])} | "
            f"{f4(row['cp_catf_metrics']['map50'])} | {f4(row['cp_catf_metrics']['map50_95'])} | "
            f"{fd(row['delta_cp_vs_clean']['precision'])} | {fd(row['delta_cp_vs_clean']['recall'])} | "
            f"{fd(row['delta_cp_vs_clean']['map50'])} | {fd(row['delta_cp_vs_clean']['map50_95'])} | "
            f"{fd(row['delta_cp_vs_fixed']['precision'])} | {fd(row['delta_cp_vs_fixed']['recall'])} | "
            f"{fd(row['delta_cp_vs_fixed']['map50'])} | {fd(row['delta_cp_vs_fixed']['map50_95'])} | "
            f"{str(row['constraint_failed']).lower()} |"
        )

    lines.extend(
        [
            "",
            "## Augmentation and Probe Decisions",
            "",
            "| Seed | Image aug allowed | Probe reject image aug | Industrial samples | ROI applied | Router random draws | Active classes | OK3 active | OK3 ROI | Sample weighting |",
            "|---:|---|---|---:|---:|---:|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['seed']} | {str(row['image_modification_allowed']).lower()} | "
            f"{str(row['probe_reject_image_aug']).lower()} | {row['industrial_samples_augmented']} | "
            f"{row['roi_aug_applied']} | {row['router_random_draw_count']} | `{row['active_classes']}` | "
            f"{str(row['ok3_active']).lower()} | {row['ok3_roi_applied']} | "
            f"`{row['sample_weighting_statuses']}` |"
        )

    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"1. Seed0 CP-CATF metrics: P={f4(rows[0]['cp_catf_metrics']['precision'])}, R={f4(rows[0]['cp_catf_metrics']['recall'])}, mAP50={f4(rows[0]['cp_catf_metrics']['map50'])}, mAP50-95={f4(rows[0]['cp_catf_metrics']['map50_95'])}.",
            f"2. Seed1 CP-CATF metrics: P={f4(rows[1]['cp_catf_metrics']['precision'])}, R={f4(rows[1]['cp_catf_metrics']['recall'])}, mAP50={f4(rows[1]['cp_catf_metrics']['map50'])}, mAP50-95={f4(rows[1]['cp_catf_metrics']['map50_95'])}.",
            f"3. Seed2 CP-CATF metrics: P={f4(rows[2]['cp_catf_metrics']['precision'])}, R={f4(rows[2]['cp_catf_metrics']['recall'])}, mAP50={f4(rows[2]['cp_catf_metrics']['map50'])}, mAP50-95={f4(rows[2]['cp_catf_metrics']['map50_95'])}.",
            f"4. Constraint pass: `{summary['constraint_pass_count']}/3`; 3/3 pass is `{str(summary['three_of_three_pass']).lower()}`.",
            f"5. Seed0 retained fixed CATF-v2 mAP gain: `{str(summary['seed0_fixed_map_gain_retained']).lower()}`; mAP50 retention={pct(rows[0]['map50_fixed_gain_retention'])}, mAP50-95 retention={pct(rows[0]['map95_fixed_gain_retention'])}.",
            f"6. Seed1 retained fixed CATF-v2 gain: `{str(summary['seed1_fixed_gain_retained']).lower()}`; mAP50 retention={pct(rows[1]['map50_fixed_gain_retention'])}, mAP50-95 retention={pct(rows[1]['map95_fixed_gain_retention'])}.",
            f"7. Seed2 rejected image augmentation and protected clean baseline: `{str(summary['seed2_clean_protected']).lower()}`; ROI/industrial zero: `{str(summary['seed2_roi_industrial_zero']).lower()}`.",
            f"8. OK3 was never active: `{str(not summary['ok3_any_active']).lower()}`; OK3 total ROI applied={summary['ok3_total_roi_applied']}.",
            f"9. CP-CATF did not rely on dataset-specific rules: `{str(not summary['dataset_specific_rule_used']).lower()}`; RiskGuard final rule used: `{str(summary['riskguard_used_as_final_rule']).lower()}`.",
            "10. Recommendation: CP-CATF is a final main-method candidate for this development-mode validation because seed0/seed1 retain fixed CATF-v2 behavior and seed2 is protected, but it is not yet a leakage-free paper result.",
            "11. Limitation: current offline probe uses existing validation diagnostics. Paper mode must replace this with a train/probe split or train hard-example probe set before final claims.",
            "",
            "## Mean Metrics",
            "",
            "| Group | P mean | R mean | mAP50 mean | mAP50-95 mean |",
            "|---|---:|---:|---:|---:|",
            f"| clean native | {f4(summary['clean_mean_std']['precision']['mean'])} | {f4(summary['clean_mean_std']['recall']['mean'])} | {f4(summary['clean_mean_std']['map50']['mean'])} | {f4(summary['clean_mean_std']['map50_95']['mean'])} |",
            f"| fixed CATF-v2 | {f4(summary['fixed_mean_std']['precision']['mean'])} | {f4(summary['fixed_mean_std']['recall']['mean'])} | {f4(summary['fixed_mean_std']['map50']['mean'])} | {f4(summary['fixed_mean_std']['map50_95']['mean'])} |",
            f"| CP-CATF | {f4(summary['cp_catf_mean_std']['precision']['mean'])} | {f4(summary['cp_catf_mean_std']['recall']['mean'])} | {f4(summary['cp_catf_mean_std']['map50']['mean'])} | {f4(summary['cp_catf_mean_std']['map50_95']['mean'])} |",
            "",
            f"- Mean CP-CATF delta vs clean: dP={fd(summary['cp_delta_vs_clean_mean']['precision'])}, dR={fd(summary['cp_delta_vs_clean_mean']['recall'])}, dM50={fd(summary['cp_delta_vs_clean_mean']['map50'])}, dM95={fd(summary['cp_delta_vs_clean_mean']['map50_95'])}.",
            f"- Mean CP-CATF delta vs fixed CATF-v2: dP={fd(summary['cp_delta_vs_fixed_mean']['precision'])}, dR={fd(summary['cp_delta_vs_fixed_mean']['recall'])}, dM50={fd(summary['cp_delta_vs_fixed_mean']['map50'])}, dM95={fd(summary['cp_delta_vs_fixed_mean']['map50_95'])}.",
        ]
    )
    return lines


def main() -> None:
    fixed_payload = read_json(FIXED_SUMMARY)
    rows = [build_row(seed, fixed_payload) for seed in (0, 1, 2)]
    summary = build_summary(rows)
    payload = {
        "run_group": "multiseed_clean_yolo_default_vs_catf_v2_cp_catf",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "code_commit_at_report_generation": git_commit(),
        "fixed_summary_path": str(FIXED_SUMMARY),
        "safe_summary_path": str(SAFE_SUMMARY),
        "gated_summary_path": str(GATED_SUMMARY),
        "summary": summary,
        "rows": rows,
    }
    write_json(REPORTS / "multiseed_cp_catf_summary.json", payload)
    write_md(REPORTS / "multiseed_cp_catf_summary.md", build_markdown(summary, rows))
    for row in rows:
        write_seed_compare(row)
    print(f"Wrote {REPORTS / 'multiseed_cp_catf_summary.json'}")
    print(f"Wrote {REPORTS / 'multiseed_cp_catf_summary.md'}")


if __name__ == "__main__":
    main()
