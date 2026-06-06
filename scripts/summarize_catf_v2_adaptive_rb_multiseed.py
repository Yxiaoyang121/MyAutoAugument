from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_adaptive_rb")
REPORTS = ROOT / "reports"
SEED2_SOURCE = Path("outputs/experiments/catf_v2_adaptive_rb_seed2_50ep")
FIXED_SUMMARY = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json")
SAFE_SUMMARY = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_safe/reports/multiseed_catf_v2_safe_summary.json")
GATED_SUMMARY = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated/reports/multiseed_catf_v2_gated_summary.json")

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


def metrics(source: dict[str, Any]) -> dict[str, float]:
    payload = source.get("metrics") or source.get("val", {}).get("metrics") or source
    return {key: float(payload[key]) for key in METRIC_KEYS}


def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: a[key] - b[key] for key in METRIC_KEYS}


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


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def row_by_seed(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return next(row for row in rows if int(row["seed"]) == int(seed))


def adaptive_root(seed: int) -> Path:
    if seed == 2:
        return SEED2_SOURCE
    return ROOT / f"seed_{seed}" / "catf_v2_adaptive_rb"


def first_event(events: list[dict[str, Any]], action: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("action") == action), None)


def candidate_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((event for event in events if event.get("candidate_branch_started")), None)


def rollback_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((event for event in events if event.get("action") == "rollback"), None)


def accept_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((event for event in events if event.get("action") == "accept_candidate"), None)


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
    results_csv = run_root / "train" / "results.csv"
    if results_csv.exists():
        for line in results_csv.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
            if line.strip():
                rows.append(line.split(",", 1)[0].strip())
    epochs = [int(float(value)) for value in rows if value]
    return {
        "epoch_count": len(epochs),
        "first_epoch": epochs[0] if epochs else None,
        "last_epoch": epochs[-1] if epochs else None,
        "epoch_continuous": epochs == list(range(1, len(epochs) + 1)),
    }


def active_classes_from_policy_matrices(run_root: Path) -> list[int]:
    active: set[int] = set()
    for path in sorted((run_root / "reports").glob("policy_matrix_epoch_*_after.json")):
        payload = read_json(path)
        for cid, class_payload in (payload.get("classes") or {}).items():
            ops = class_payload.get("ops") or {}
            if any(float(op.get("prob", 0.0) or 0.0) > 0.0 for op in ops.values()):
                active.add(int(cid))
    return sorted(active)


def class_ids_from_mapping(mapping: dict[str, Any]) -> list[int]:
    ids: list[int] = []
    for key, value in mapping.items():
        if int(value or 0) > 0:
            ids.append(int(key))
    return sorted(ids)


def ok3_audit(run_root: Path, roi_stats: dict[str, Any], *, candidate_started: bool) -> dict[str, Any]:
    policy_nonzero = active_classes_from_policy_matrices(run_root)
    affected = roi_stats.get("affected_classes") or {}
    applied = class_ids_from_mapping(affected)
    return {
        "ok3_active": 1 in applied or (candidate_started and 1 in policy_nonzero),
        "ok3_roi_applied": int(affected.get("1", 0) or affected.get(1, 0) or 0),
        "active_classes": applied,
        "policy_nonzero_classes": policy_nonzero,
    }


def retention_ratio(adaptive_delta: dict[str, float], fixed_delta: dict[str, float], key: str) -> float | None:
    if fixed_delta[key] <= 0.0:
        return None
    return adaptive_delta[key] / fixed_delta[key]


def build_row(seed: int, fixed_payload: dict[str, Any], safe_rows: list[dict[str, Any]], gated_rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_root = adaptive_root(seed)
    final_payload = read_json(run_root / "reports" / "final_metrics.json")
    events = read_json(run_root / "reports" / "adaptive_burnin_events.json").get("events", [])
    rb_events = read_json(run_root / "reports" / "rollback_controller_events.json").get("events", [])
    online = read_json(run_root / "reports" / "online_aug_stats.json")
    roi = read_json(run_root / "reports" / "roi_aug_stats.json")
    fixed_seed = fixed_payload["seeds"][str(seed)]
    safe_row = row_by_seed(safe_rows, seed)
    gated_row = row_by_seed(gated_rows, seed)

    clean = dict(fixed_seed["clean_native"])
    fixed = dict(fixed_seed["fixed_catf_v2"])
    safe = dict(safe_row["safe_metrics"])
    gated = dict(gated_row["gated_metrics"])
    adaptive = metrics(final_payload)
    adaptive_delta_clean = delta(adaptive, clean)
    adaptive_delta_fixed = delta(adaptive, fixed)
    failed, failure_reasons = constraint_failed(adaptive_delta_clean)
    noop = first_event(events, "no_op_fallback")
    start = candidate_event(events)
    rb_rollback = rollback_event(rb_events)
    rb_accept = accept_event(rb_events)
    audit = ok3_audit(run_root, roi, candidate_started=bool(start))
    fixed_delta_clean = dict(fixed_seed["delta_fixed_vs_clean"])

    return {
        "seed": seed,
        "adaptive_run_path": str(run_root),
        "seed2_reused_from": str(SEED2_SOURCE) if seed == 2 else None,
        "clean_metrics": clean,
        "fixed_catf_v2_metrics": fixed,
        "safe_metrics": safe,
        "gated_metrics": gated,
        "adaptive_rb_metrics": adaptive,
        "fixed_delta_vs_clean": fixed_delta_clean,
        "safe_delta_vs_clean": delta(safe, clean),
        "gated_delta_vs_clean": delta(gated, clean),
        "adaptive_delta_vs_clean": adaptive_delta_clean,
        "adaptive_delta_vs_fixed": adaptive_delta_fixed,
        "adaptive_delta_vs_safe": delta(adaptive, safe),
        "adaptive_delta_vs_gated": delta(adaptive, gated),
        "adaptive_constraint_failed": failed,
        "adaptive_failure_reasons": failure_reasons,
        "fixed_constraint_failed": bool(fixed_seed["fixed_constraint_failed"]),
        "safe_constraint_failed": bool(safe_row["safe_constraint_failed"]),
        "gated_constraint_failed": bool(gated_row["gated_constraint_failed"]),
        "adaptive_start_epoch": final_payload.get("adaptive_start_epoch"),
        "candidate_started": bool(start),
        "candidate_start_epoch": start.get("epoch") if start else None,
        "candidate_action": start.get("action") if start else None,
        "rollback_triggered": bool(rb_rollback),
        "rollback_epoch": rb_rollback.get("epoch") if rb_rollback else None,
        "candidate_accepted": bool(rb_accept),
        "candidate_accept_epoch": rb_accept.get("epoch") if rb_accept else None,
        "noop_fallback": bool(noop),
        "noop_fallback_epoch": noop.get("epoch") if noop else None,
        "noop_reasons": noop.get("reasons") if noop else [],
        "industrial_samples_augmented": int(online.get("samples_augmented", 0) or 0),
        "roi_aug_applied": int(roi.get("roi_aug_applied", 0) or 0),
        "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
        "ops": dict(online.get("ops") or {}),
        "roi_affected_classes": dict(roi.get("affected_classes") or {}),
        "train_image_count": int(online.get("train_image_count", 0) or 0),
        "bbox_class_valid": bool(final_payload.get("summary", {}).get("bbox_class_valid")),
        "epoch_integrity": epoch_integrity(run_root, final_payload),
        "ok3_active": audit["ok3_active"],
        "ok3_roi_applied": audit["ok3_roi_applied"],
        "active_classes": audit["active_classes"],
        "policy_nonzero_classes": audit["policy_nonzero_classes"],
        "seed0_fixed_map_gain_retained": (
            seed == 0
            and adaptive_delta_clean["map50"] > 0
            and adaptive_delta_clean["map50_95"] > 0
            and adaptive_delta_fixed["map50"] >= -0.005
            and adaptive_delta_fixed["map50_95"] >= -0.005
        ),
        "seed1_fixed_gain_retained": (
            seed == 1
            and adaptive_delta_clean["recall"] > 0
            and adaptive_delta_clean["map50"] > 0
            and adaptive_delta_clean["map50_95"] > 0
            and not failed
        ),
        "map50_fixed_gain_retention": retention_ratio(adaptive_delta_clean, fixed_delta_clean, "map50"),
        "map95_fixed_gain_retention": retention_ratio(adaptive_delta_clean, fixed_delta_clean, "map50_95"),
    }


def mean_std(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [float(row[field][key]) for row in rows]
        payload[key] = {"mean": mean(values), "std": pstdev(values)}
    return payload


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed_count = sum(1 for row in rows if row["adaptive_constraint_failed"])
    pass_count = len(rows) - failed_count
    seed0 = row_by_seed(rows, 0)
    seed1 = row_by_seed(rows, 1)
    seed2 = row_by_seed(rows, 2)
    seed0_retained = bool(seed0["seed0_fixed_map_gain_retained"])
    seed1_retained = bool(seed1["seed1_fixed_gain_retained"])
    seed2_clean = (
        seed2["noop_fallback"]
        and not seed2["candidate_started"]
        and not seed2["adaptive_constraint_failed"]
        and all(abs(seed2["adaptive_delta_vs_clean"][key]) < 1e-9 for key in METRIC_KEYS)
    )
    main_method = failed_count == 0 and seed0_retained and seed1_retained and seed2_clean
    if main_method:
        paper_position = "Adaptive-RB is a final paper main-method candidate."
    else:
        paper_position = (
            "Adaptive-RB is not yet a final paper main method: it protects seed2, but seed0 falls back to clean "
            "and seed1 violates the precision-drop constraint despite recall/mAP gains."
        )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_group": ROOT.name,
        "code_commit_used": git_commit(),
        "seeds": [row["seed"] for row in rows],
        "adaptive_constraint_failed_count": failed_count,
        "adaptive_constraint_pass_count": pass_count,
        "achieved_3_of_3": failed_count == 0,
        "all_epoch_continuous": all(bool(row["epoch_integrity"].get("epoch_continuous")) for row in rows),
        "all_train_images_2301": all(row["train_image_count"] == 2301 for row in rows),
        "all_bbox_class_valid": all(row["bbox_class_valid"] for row in rows),
        "ok3_ever_active": any(row["ok3_active"] for row in rows),
        "ok3_roi_applied_total": sum(row["ok3_roi_applied"] for row in rows),
        "adaptive_delta_vs_clean_mean_std": mean_std(rows, "adaptive_delta_vs_clean"),
        "seed0_fixed_map_gain_retained": seed0_retained,
        "seed1_fixed_gain_retained": seed1_retained,
        "seed2_clean_noop_fallback": seed2_clean,
        "seed2_noop_epoch": seed2["noop_fallback_epoch"],
        "adaptive_rb_recommended_as_paper_main_method": main_method,
        "paper_position": paper_position,
        "next_parameter_adjustments_if_needed": [
            "Use cumulative burn-in evidence across epoch 5/10/15 instead of only the current diagnosis so seed0 is not lost when epoch15 evidence thins out.",
            "Make the RB probe gate compare against clean/reference constraints, not only the immediate probe reference, so seed1 precision_drop_gt_0.01 triggers rollback or shrink.",
            "Require a precision floor or threshold-calibration step before accepting a low-risk candidate with recall/mAP gains.",
            "Keep strong clean baseline protection for seed2 unchanged unless later seeds show false abstention under clear high-confidence issues.",
        ],
    }


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Full Multiseed CATF-v2 Adaptive Burn-in + RB Summary",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Code commit used: `{summary['code_commit_used']}`",
        "",
        "## Metrics",
        "",
        "| Seed | Clean P | Clean R | Clean mAP50 | Clean mAP50-95 | Fixed P | Fixed R | Fixed mAP50 | Fixed mAP50-95 | Adaptive-RB P | Adaptive-RB R | Adaptive-RB mAP50 | Adaptive-RB mAP50-95 | dP vs clean | dR vs clean | dM50 vs clean | dM95 vs clean | constraint_failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        clean = row["clean_metrics"]
        fixed = row["fixed_catf_v2_metrics"]
        adaptive = row["adaptive_rb_metrics"]
        d = row["adaptive_delta_vs_clean"]
        lines.append(
            f"| {row['seed']} | {f4(clean['precision'])} | {f4(clean['recall'])} | {f4(clean['map50'])} | {f4(clean['map50_95'])} | "
            f"{f4(fixed['precision'])} | {f4(fixed['recall'])} | {f4(fixed['map50'])} | {f4(fixed['map50_95'])} | "
            f"{f4(adaptive['precision'])} | {f4(adaptive['recall'])} | {f4(adaptive['map50'])} | {f4(adaptive['map50_95'])} | "
            f"{fd(d['precision'])} | {fd(d['recall'])} | {fd(d['map50'])} | {fd(d['map50_95'])} | "
            f"{str(row['adaptive_constraint_failed']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Gate And Augmentation Audit",
            "",
            "| Seed | adaptive start | candidate | rollback | accepted | no-op fallback | no-op epoch | industrial samples | ROI applied | router draws | active classes | OK3 active | OK3 ROI |",
            "|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['adaptive_start_epoch']} | {str(row['candidate_started']).lower()} | "
            f"{str(row['rollback_triggered']).lower()} | {str(row['candidate_accepted']).lower()} | "
            f"{str(row['noop_fallback']).lower()} | {row['noop_fallback_epoch']} | "
            f"{row['industrial_samples_augmented']} | {row['roi_aug_applied']} | {row['router_random_draw_count']} | "
            f"`{row['active_classes']}` | {str(row['ok3_active']).lower()} | {row['ok3_roi_applied']} |"
        )
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            f"- Constraint pass count: `{summary['adaptive_constraint_pass_count']}/3`; achieved 3/3: `{str(summary['achieved_3_of_3']).lower()}`.",
            f"- Seed0 retained fixed mAP gains: `{str(summary['seed0_fixed_map_gain_retained']).lower()}`. It fell back to clean/no-op and lost fixed CATF-v2 mAP50/mAP50-95 gains.",
            f"- Seed1 retained fixed gains: `{str(summary['seed1_fixed_gain_retained']).lower()}`. It retained part of the recall/mAP lift, but failed the precision constraint.",
            f"- Seed2 clean no-op fallback: `{str(summary['seed2_clean_noop_fallback']).lower()}` at epoch `{summary['seed2_noop_epoch']}`.",
            f"- OK3 ever active: `{str(summary['ok3_ever_active']).lower()}`; OK3 ROI applied total: `{summary['ok3_roi_applied_total']}`.",
            f"- Recommended as paper main method: `{str(summary['adaptive_rb_recommended_as_paper_main_method']).lower()}`.",
            f"- Position: {summary['paper_position']}",
            "",
            "## Comparison With Fixed / Safe / Gated",
            "",
            "- Versus fixed CATF-v2: adaptive-RB protects seed2 through strict no-op, but does not preserve seed0 gains and seed1 fails the precision constraint.",
            "- Versus Safe: adaptive-RB is less conservative on seed1 and recovers recall/mAP gains, but Safe remains better on constraint pass count in this run.",
            "- Versus Gated: adaptive-RB fixes the seed2 non-rollback problem by preventing candidate start, but its low-risk seed1 candidate still needs a stricter precision-aware accept gate.",
            "",
            "## Gain Retention",
            "",
            "| Seed | dM50 fixed vs clean | dM95 fixed vs clean | dM50 adaptive vs clean | dM95 adaptive vs clean | mAP50 retention | mAP95 retention |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        fd_clean = row["fixed_delta_vs_clean"]
        ad_clean = row["adaptive_delta_vs_clean"]
        lines.append(
            f"| {row['seed']} | {fd(fd_clean['map50'])} | {fd(fd_clean['map50_95'])} | "
            f"{fd(ad_clean['map50'])} | {fd(ad_clean['map50_95'])} | "
            f"{pct(row['map50_fixed_gain_retention'])} | {pct(row['map95_fixed_gain_retention'])} |"
        )
    lines.extend(
        [
            "",
            "## Next Parameter Adjustments",
            "",
        ]
    )
    for item in summary["next_parameter_adjustments_if_needed"]:
        lines.append(f"- {item}")
    return lines


def write_seed2_reference(row: dict[str, Any]) -> None:
    reference_dir = ROOT / "seed_2" / "catf_v2_adaptive_rb" / "reports"
    payload = {
        "note": "Seed2 adaptive-RB 50ep was completed before this full multiseed summary and is reused here.",
        "source_run": str(SEED2_SOURCE),
        "source_final_metrics": str(SEED2_SOURCE / "reports" / "final_metrics.json"),
        "source_summary": str(SEED2_SOURCE / "reports" / "adaptive_rb_seed2_50ep_summary.json"),
        "adaptive_rb_metrics": row["adaptive_rb_metrics"],
        "adaptive_start_epoch": row["adaptive_start_epoch"],
        "candidate_started": row["candidate_started"],
        "rollback_triggered": row["rollback_triggered"],
        "noop_fallback": row["noop_fallback"],
        "noop_fallback_epoch": row["noop_fallback_epoch"],
        "constraint_failed": row["adaptive_constraint_failed"],
        "industrial_samples_augmented": row["industrial_samples_augmented"],
        "roi_aug_applied": row["roi_aug_applied"],
        "router_random_draw_count": row["router_random_draw_count"],
    }
    write_json(reference_dir / "adaptive_rb_seed2_reuse_reference.json", payload)
    write_md(
        reference_dir / "README.md",
        [
            "# Seed2 Adaptive-RB Reuse Reference",
            "",
            "Seed2 adaptive-RB 50ep was completed before this full multiseed summary and is reused here.",
            "",
            f"- Source run: `{SEED2_SOURCE}`",
            f"- Final metrics JSON: `{SEED2_SOURCE / 'reports' / 'final_metrics.json'}`",
            f"- Summary JSON: `{SEED2_SOURCE / 'reports' / 'adaptive_rb_seed2_50ep_summary.json'}`",
            f"- adaptive start epoch: `{row['adaptive_start_epoch']}`",
            f"- candidate started: `{str(row['candidate_started']).lower()}`",
            f"- rollback triggered: `{str(row['rollback_triggered']).lower()}`",
            f"- no-op fallback: `{str(row['noop_fallback']).lower()}` at epoch `{row['noop_fallback_epoch']}`",
            f"- constraint_failed: `{str(row['adaptive_constraint_failed']).lower()}`",
        ],
    )


def main() -> None:
    fixed_payload = read_json(FIXED_SUMMARY)
    safe_payload = read_json(SAFE_SUMMARY)
    gated_payload = read_json(GATED_SUMMARY)
    rows = [build_row(seed, fixed_payload, safe_payload["rows"], gated_payload["rows"]) for seed in (0, 1, 2)]
    summary = build_summary(rows)
    payload = {"summary": summary, "rows": rows}
    write_json(REPORTS / "multiseed_adaptive_rb_summary.json", payload)
    write_md(REPORTS / "multiseed_adaptive_rb_summary.md", build_markdown(summary, rows))
    write_seed2_reference(row_by_seed(rows, 2))
    print(f"Wrote {REPORTS / 'multiseed_adaptive_rb_summary.json'}")
    print(f"Wrote {REPORTS / 'multiseed_adaptive_rb_summary.md'}")


if __name__ == "__main__":
    main()
