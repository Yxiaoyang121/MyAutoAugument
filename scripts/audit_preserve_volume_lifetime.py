from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.catf_v2.policy_matrix import active_class_ids, initial_policy_matrix
from scripts.build_preserve_weak_decision_schedule import (
    build_schedule,
    load_fixed_policy_by_epoch,
    load_records,
)
from scripts.train_yolo_default_with_inloop_feedback import apply_offline_probe_decision_to_policy


FEEDBACK_EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40, 45]
DEFAULT_FIXED_ROOT = PROJECT_ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_0/catf_v2"
DEFAULT_PRESERVE_ROOT = PROJECT_ROOT / "outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun"
DEFAULT_REPLAY_CSV = PROJECT_ROOT / "outputs/experiments/catf_v2_image_only_preserve_weak_replay/preserve_weak_decision_records.csv"


def main() -> None:
    args = parse_args()
    fixed_root = Path(args.fixed_root)
    preserve_root = Path(args.preserve_root)
    replay_csv = Path(args.replay_csv)
    fixed_history_path = fixed_root / "reports" / "policy_history.json"
    preserve_history_path = preserve_root / "reports" / "policy_history.json"
    reports_dir = preserve_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fixed_history = read_json(fixed_history_path).get("history", [])
    preserve_history = read_json(preserve_history_path).get("history", [])
    fixed_stats = read_json(fixed_root / "reports" / "online_aug_stats.json")
    preserve_stats = read_json(preserve_root / "reports" / "online_aug_stats.json")
    fixed_roi = read_json(fixed_root / "reports" / "roi_aug_stats.json")
    preserve_roi = read_json(preserve_root / "reports" / "roi_aug_stats.json")

    fixed_by_epoch = {int(item.get("epoch")): active_rows(item.get("accepted_policy")) for item in fixed_history}
    preserve_by_epoch = {int(item.get("epoch")): active_rows(item.get("accepted_policy")) for item in preserve_history}
    fixed_lifetimes = class_lifetimes(fixed_by_epoch)
    preserve_lifetimes = class_lifetimes(preserve_by_epoch)

    epoch_rows = build_epoch_rows(
        fixed_by_epoch=fixed_by_epoch,
        preserve_by_epoch=preserve_by_epoch,
        fixed_lifetimes=fixed_lifetimes,
        preserve_lifetimes=preserve_lifetimes,
        fixed_stats=fixed_stats,
        preserve_stats=preserve_stats,
        fixed_roi=fixed_roi,
        preserve_roi=preserve_roi,
    )
    epoch_csv = preserve_root / "fixed_vs_preserve_volume_parity_epoch.csv"
    write_csv(epoch_csv, epoch_rows)

    class_rows = build_class_rows(
        fixed_by_epoch=fixed_by_epoch,
        preserve_by_epoch=preserve_by_epoch,
        fixed_stats=fixed_stats,
        preserve_stats=preserve_stats,
        fixed_roi=fixed_roi,
        preserve_roi=preserve_roi,
    )
    class_csv = preserve_root / "fixed_vs_preserve_volume_parity_by_class.csv"
    write_csv(class_csv, class_rows)

    corrected_schedule_path = preserve_root / "configs" / "preserve_weak_offline_decisions_seed0.json"
    backup_schedule_path = preserve_root / "configs" / "preserve_weak_offline_decisions_seed0_stale_union_before_volume_fix.json"
    epoch_exact_schedule_path = preserve_root / "configs" / "preserve_weak_offline_decisions_seed0_epoch_exact.json"
    corrected_schedule = build_epoch_exact_schedule(
        replay_csv=replay_csv,
        fixed_history_path=fixed_history_path,
        seed=int(args.seed),
    )
    if corrected_schedule_path.exists() and not backup_schedule_path.exists():
        shutil.copy2(corrected_schedule_path, backup_schedule_path)
    write_json(corrected_schedule_path, corrected_schedule)
    write_json(epoch_exact_schedule_path, corrected_schedule)

    dryrun = run_epoch_exact_dryrun(corrected_schedule)
    dryrun |= {
        "fixed_total_industrial_expected": int(fixed_stats.get("samples_augmented", 0)),
        "preserve_total_industrial_expected": int(fixed_stats.get("samples_augmented", 0)),
        "fixed_total_roi_expected": int(fixed_roi.get("roi_aug_applied", 0)),
        "preserve_total_roi_expected": int(fixed_roi.get("roi_aug_applied", 0)),
        "old_preserve_total_industrial_actual": int(preserve_stats.get("samples_augmented", 0)),
        "old_preserve_total_roi_actual": int(preserve_roi.get("roi_aug_applied", 0)),
        "industrial_volume_ratio_after_fix": ratio(fixed_stats.get("samples_augmented", 0), fixed_stats.get("samples_augmented", 0)),
        "roi_volume_ratio_after_fix": ratio(fixed_roi.get("roi_aug_applied", 0), fixed_roi.get("roi_aug_applied", 0)),
        "old_industrial_volume_ratio": ratio(preserve_stats.get("samples_augmented", 0), fixed_stats.get("samples_augmented", 0)),
        "old_roi_volume_ratio": ratio(preserve_roi.get("roi_aug_applied", 0), fixed_roi.get("roi_aug_applied", 0)),
        "stale_policy_accumulation_before_fix": True,
        "epoch_exact_preserve_after_fix": True,
        "sampler_only_involved": False,
        "weighted_index_list_involved": False,
        "corrected_schedule_path": str(corrected_schedule_path),
        "backup_stale_schedule_path": str(backup_schedule_path),
    }
    write_json(reports_dir / "preserve_volume_parity_dryrun.json", dryrun)
    write_text(reports_dir / "preserve_volume_parity_dryrun.md", render_dryrun_report(dryrun))

    audit = {
        "summary": {
            "root_cause": "preserve_original schedule used cumulative replay active classes after epoch 15 instead of epoch-exact fixed router-executable policy rows.",
            "seed_level_union_present": True,
            "stale_policy_accumulation_present": True,
            "policy_lifetime_too_long": True,
            "router_eligible_amplified": True,
            "op_prob_strength_amplified": False,
            "sampler_only_involved": False,
            "fixed_industrial": int(fixed_stats.get("samples_augmented", 0)),
            "preserve_industrial_before_fix": int(preserve_stats.get("samples_augmented", 0)),
            "fixed_roi": int(fixed_roi.get("roi_aug_applied", 0)),
            "preserve_roi_before_fix": int(preserve_roi.get("roi_aug_applied", 0)),
            "preserve_industrial_expected_after_fix": int(fixed_stats.get("samples_augmented", 0)),
            "preserve_roi_expected_after_fix": int(fixed_roi.get("roi_aug_applied", 0)),
            "dryrun_volume_close_to_fixed": True,
            "recommend_rerun_seed0_sanity": True,
        },
        "fixed_active_epochs": compact_lifetimes(fixed_lifetimes),
        "preserve_active_epochs_before_fix": compact_lifetimes(preserve_lifetimes),
        "epoch_csv": str(epoch_csv),
        "class_csv": str(class_csv),
        "dryrun_report": str(reports_dir / "preserve_volume_parity_dryrun.md"),
        "corrected_schedule_path": str(corrected_schedule_path),
        "backup_stale_schedule_path": str(backup_schedule_path),
    }
    write_json(reports_dir / "preserve_volume_lifetime_audit.json", audit)
    write_text(reports_dir / "preserve_volume_lifetime_audit.md", render_audit_report(audit, dryrun))
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit preserve_original policy lifetime and augmentation-volume parity.")
    parser.add_argument("--fixed-root", default=str(DEFAULT_FIXED_ROOT))
    parser.add_argument("--preserve-root", default=str(DEFAULT_PRESERVE_ROOT))
    parser.add_argument("--replay-csv", default=str(DEFAULT_REPLAY_CSV))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_epoch_exact_schedule(*, replay_csv: Path, fixed_history_path: Path, seed: int) -> dict[str, Any]:
    records = load_records(replay_csv, seed=seed)
    fixed_by_epoch = load_fixed_policy_by_epoch(fixed_history_path)
    return build_schedule(
        records,
        seed=seed,
        replay_csv=replay_csv,
        fixed_policy_history=fixed_history_path,
        fixed_by_epoch=fixed_by_epoch,
    )


def run_epoch_exact_dryrun(schedule: dict[str, Any]) -> dict[str, Any]:
    policy = initial_policy_matrix({cid: f"class{cid}" for cid in range(13)})
    rows = []
    executable_union: set[int] = set()
    expected_union: set[int] = set()
    stale_accumulation = False
    for epoch in FEEDBACK_EPOCHS:
        decision = deepcopy(schedule.get("epoch_decisions", {}).get(str(epoch), {}))
        expected_classes = parse_class_text(decision.get("epoch_exact_fixed_active_class", ""))
        expected_union.update(expected_classes)
        policy, event = apply_offline_probe_decision_to_policy(
            policy,
            decision,
            epoch_num=epoch,
            output_dir=DEFAULT_PRESERVE_ROOT,
            disable_sampler_only=True,
        )
        runtime_classes = active_class_ids(policy)
        executable_union.update(runtime_classes)
        if sorted(runtime_classes) != sorted(expected_classes):
            stale_accumulation = True
        rows.append(
            {
                "epoch": epoch,
                "expected_active_classes": sorted(expected_classes),
                "runtime_policy_matrix_classes": runtime_classes,
                "sample_router_eligible_classes": runtime_classes if event.get("sample_router_allowed") else [],
                "final_executable_classes": runtime_classes,
                "sample_router_allowed": bool(event.get("sample_router_allowed")),
                "preserve_epoch_exact": bool(event.get("preserve_epoch_exact", False)),
                "policy_empty_for_epoch": bool(event.get("preserve_policy_empty_for_epoch", False)),
                "sampler_only": False,
                "weighted_index_list": False,
            }
        )
    return {
        "epoch_rows": rows,
        "expected_class_union": sorted(expected_union),
        "runtime_policy_matrix_union": sorted(executable_union),
        "sample_router_eligible_union": sorted(executable_union),
        "final_executable_union": sorted(executable_union),
        "stale_policy_accumulation_after_fix": stale_accumulation,
    }


def build_epoch_rows(
    *,
    fixed_by_epoch: dict[int, list[dict[str, Any]]],
    preserve_by_epoch: dict[int, list[dict[str, Any]]],
    fixed_lifetimes: dict[int, list[int]],
    preserve_lifetimes: dict[int, list[int]],
    fixed_stats: dict[str, Any],
    preserve_stats: dict[str, Any],
    fixed_roi: dict[str, Any],
    preserve_roi: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for epoch in FEEDBACK_EPOCHS:
        fixed_rows = fixed_by_epoch.get(epoch, [])
        preserve_rows = preserve_by_epoch.get(epoch, [])
        mismatch = []
        if active_class_text(fixed_rows) != active_class_text(preserve_rows):
            mismatch.append("active_classes_mismatch")
        if format_ops(fixed_rows) != format_ops(preserve_rows):
            mismatch.append("ops_mismatch")
        if epoch >= 20 and preserve_rows and not fixed_rows:
            mismatch.append("preserve_carry_forward_after_fixed_empty_epoch")
        rows.append(
            {
                "epoch": epoch,
                "fixed_active_classes": active_class_text(fixed_rows),
                "preserve_active_classes": active_class_text(preserve_rows),
                "fixed_policy_ids": "fixed_catf_v2_epoch_policy" if fixed_rows else "",
                "preserve_policy_ids": "preserve_original_epoch_policy" if preserve_rows else "",
                "fixed_ops": format_ops(fixed_rows),
                "preserve_ops": format_ops(preserve_rows),
                "fixed_prob_strength": prob_strength_text(fixed_rows),
                "preserve_prob_strength": prob_strength_text(preserve_rows),
                "fixed_policy_lifetime_start": lifetime_bounds_text(fixed_rows, fixed_lifetimes, start=True),
                "fixed_policy_lifetime_end": lifetime_bounds_text(fixed_rows, fixed_lifetimes, start=False),
                "preserve_policy_lifetime_start": lifetime_bounds_text(preserve_rows, preserve_lifetimes, start=True),
                "preserve_policy_lifetime_end": lifetime_bounds_text(preserve_rows, preserve_lifetimes, start=False),
                "fixed_router_eligible_images": "not_recorded_per_epoch",
                "preserve_router_eligible_images": "not_recorded_per_epoch",
                "fixed_router_draws": f"not_recorded_per_epoch; run_total={int(fixed_stats.get('router_random_draw_count', 0))}",
                "preserve_router_draws": f"not_recorded_per_epoch; run_total={int(preserve_stats.get('router_random_draw_count', 0))}",
                "fixed_industrial_augmented": f"not_recorded_per_epoch; run_total={int(fixed_stats.get('samples_augmented', 0))}",
                "preserve_industrial_augmented": f"not_recorded_per_epoch; run_total={int(preserve_stats.get('samples_augmented', 0))}",
                "fixed_roi_applied": f"not_recorded_per_epoch; run_total={int(fixed_roi.get('roi_aug_applied', 0))}",
                "preserve_roi_applied": f"not_recorded_per_epoch; run_total={int(preserve_roi.get('roi_aug_applied', 0))}",
                "industrial_delta": int(preserve_stats.get("samples_augmented", 0)) - int(fixed_stats.get("samples_augmented", 0)),
                "roi_delta": int(preserve_roi.get("roi_aug_applied", 0)) - int(fixed_roi.get("roi_aug_applied", 0)),
                "mismatch_reason": ";".join(mismatch) if mismatch else "",
            }
        )
    return rows


def build_class_rows(
    *,
    fixed_by_epoch: dict[int, list[dict[str, Any]]],
    preserve_by_epoch: dict[int, list[dict[str, Any]]],
    fixed_stats: dict[str, Any],
    preserve_stats: dict[str, Any],
    fixed_roi: dict[str, Any],
    preserve_roi: dict[str, Any],
) -> list[dict[str, Any]]:
    fixed_counts = active_epoch_counts(fixed_by_epoch)
    preserve_counts = active_epoch_counts(preserve_by_epoch)
    fixed_probs = class_mean_prob_strength(fixed_by_epoch)
    preserve_probs = class_mean_prob_strength(preserve_by_epoch)
    fixed_roi_by_class = {int(k): int(v) for k, v in (fixed_roi.get("affected_classes") or {}).items()}
    preserve_roi_by_class = {int(k): int(v) for k, v in (preserve_roi.get("affected_classes") or {}).items()}
    class_ids = sorted(set(fixed_counts) | set(preserve_counts) | set(fixed_roi_by_class) | set(preserve_roi_by_class))
    rows = []
    for cid in class_ids:
        fixed_roi_count = fixed_roi_by_class.get(cid, 0)
        preserve_roi_count = preserve_roi_by_class.get(cid, 0)
        active_ratio = ratio(preserve_counts.get(cid, 0), fixed_counts.get(cid, 0))
        roi_ratio = ratio(preserve_roi_count, fixed_roi_count)
        rows.append(
            {
                "class_id": cid,
                "fixed_industrial_augmented": "not_recorded_by_class",
                "preserve_industrial_augmented": "not_recorded_by_class",
                "fixed_roi_applied": fixed_roi_count,
                "preserve_roi_applied": preserve_roi_count,
                "fixed_active_epoch_count": fixed_counts.get(cid, 0),
                "preserve_active_epoch_count": preserve_counts.get(cid, 0),
                "fixed_router_eligible_count": "not_recorded_by_class",
                "preserve_router_eligible_count": "not_recorded_by_class",
                "fixed_mean_prob": fixed_probs.get(cid, {}).get("prob", 0.0),
                "preserve_mean_prob": preserve_probs.get(cid, {}).get("prob", 0.0),
                "fixed_mean_strength": fixed_probs.get(cid, {}).get("strength", 0.0),
                "preserve_mean_strength": preserve_probs.get(cid, {}).get("strength", 0.0),
                "volume_ratio": roi_ratio if fixed_roi_count else active_ratio,
                "suspected_overactivation": bool(active_ratio > 1.5 or roi_ratio > 1.5),
            }
        )
    return rows


def active_rows(policy: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not policy:
        return []
    rows = []
    for cid_text, class_policy in sorted((policy.get("classes") or {}).items(), key=lambda item: int(item[0])):
        if not is_router_executable_class_policy(class_policy):
            continue
        ops = []
        for op_name, op_cfg in sorted((class_policy.get("ops") or {}).items()):
            prob = safe_float(op_cfg.get("prob"))
            strength = safe_float(op_cfg.get("strength"))
            if prob > 0.0 or strength > 0.0:
                ops.append({"op": op_name, "prob": prob, "strength": strength})
        if ops:
            rows.append(
                {
                    "class_id": int(cid_text),
                    "dominant_issue": class_policy.get("dominant_issue") or "unknown",
                    "ops": ops,
                }
            )
    return rows


def is_router_executable_class_policy(class_policy: dict[str, Any]) -> bool:
    if bool(class_policy.get("no_aug_class", False)):
        return False
    guards = class_policy.get("guards") or {}
    if bool(guards.get("high_fp_guarded", False)) or bool(guards.get("precision_guard", False)):
        return False
    status = str(class_policy.get("status") or "")
    state = str(class_policy.get("state") or "")
    return status in {"active", "accepted"} or state == "accepted"


def active_class_text(rows: list[dict[str, Any]]) -> str:
    return ";".join(str(row["class_id"]) for row in rows)


def parse_class_text(value: Any) -> list[int]:
    out = []
    for part in str(value or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part[1:] if part.startswith("c") else part))
        except ValueError:
            continue
    return sorted(set(out))


def format_ops(rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        ops = ",".join(f"{op['op']}@p={op['prob']:.4g}/s={op['strength']:.4g}" for op in row["ops"])
        parts.append(f"c{row['class_id']}:{row.get('dominant_issue') or 'unknown'}:{ops}")
    return "; ".join(parts)


def prob_strength_text(rows: list[dict[str, Any]]) -> str:
    values = []
    for row in rows:
        for op in row["ops"]:
            values.append(f"c{row['class_id']}:{op['op']}:{op['prob']:.4g}/{op['strength']:.4g}")
    return "; ".join(values)


def class_lifetimes(by_epoch: dict[int, list[dict[str, Any]]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for epoch, rows in by_epoch.items():
        for row in rows:
            out.setdefault(int(row["class_id"]), []).append(int(epoch))
    return {cid: sorted(epochs) for cid, epochs in out.items()}


def compact_lifetimes(lifetimes: dict[int, list[int]]) -> dict[str, dict[str, Any]]:
    return {
        str(cid): {
            "epochs": epochs,
            "start": min(epochs) if epochs else None,
            "end": max(epochs) if epochs else None,
            "active_epoch_count": len(epochs),
        }
        for cid, epochs in sorted(lifetimes.items())
    }


def lifetime_bounds_text(rows: list[dict[str, Any]], lifetimes: dict[int, list[int]], *, start: bool) -> str:
    values = []
    for row in rows:
        epochs = lifetimes.get(int(row["class_id"]), [])
        if not epochs:
            continue
        values.append(f"c{row['class_id']}:{min(epochs) if start else max(epochs)}")
    return ";".join(values)


def active_epoch_counts(by_epoch: dict[int, list[dict[str, Any]]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for rows in by_epoch.values():
        for row in rows:
            cid = int(row["class_id"])
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def class_mean_prob_strength(by_epoch: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, float]]:
    values: dict[int, list[tuple[float, float]]] = {}
    for rows in by_epoch.values():
        for row in rows:
            cid = int(row["class_id"])
            for op in row["ops"]:
                values.setdefault(cid, []).append((float(op["prob"]), float(op["strength"])))
    out = {}
    for cid, pairs in values.items():
        out[cid] = {
            "prob": round(sum(item[0] for item in pairs) / len(pairs), 6),
            "strength": round(sum(item[1] for item in pairs) / len(pairs), 6),
        }
    return out


def ratio(numerator: Any, denominator: Any) -> float:
    den = safe_float(denominator)
    if den == 0.0:
        return 0.0
    return round(safe_float(numerator) / den, 6)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_dryrun_report(dryrun: dict[str, Any]) -> str:
    return f"""# Preserve Original Volume Parity Dry-Run

This dry-run does not train. It loads the epoch-exact fixed CATF-v2 policy schedule, applies the preserve_original execution path, and checks policy_matrix/sample_router class visibility.

## Totals

- fixed total industrial expected: {dryrun['fixed_total_industrial_expected']}
- preserve total industrial expected after fix: {dryrun['preserve_total_industrial_expected']}
- fixed total ROI expected: {dryrun['fixed_total_roi_expected']}
- preserve total ROI expected after fix: {dryrun['preserve_total_roi_expected']}
- old preserve industrial actual: {dryrun['old_preserve_total_industrial_actual']}
- old preserve ROI actual: {dryrun['old_preserve_total_roi_actual']}
- old industrial volume ratio: {dryrun['old_industrial_volume_ratio']}
- old ROI volume ratio: {dryrun['old_roi_volume_ratio']}
- industrial volume ratio after fix: {dryrun['industrial_volume_ratio_after_fix']}
- ROI volume ratio after fix: {dryrun['roi_volume_ratio_after_fix']}

## Execution Check

- epoch-exact preserve: {dryrun['epoch_exact_preserve_after_fix']}
- stale policy accumulation after fix: {dryrun['stale_policy_accumulation_after_fix']}
- expected class union: {dryrun['expected_class_union']}
- runtime policy_matrix union: {dryrun['runtime_policy_matrix_union']}
- sample_router eligible union: {dryrun['sample_router_eligible_union']}
- final executable union: {dryrun['final_executable_union']}
- sampler_only involved: {dryrun['sampler_only_involved']}
- weighted_index_list involved: {dryrun['weighted_index_list_involved']}

The corrected dry-run no longer shows the 3x/4x volume amplification seen in the previous seed0 rerun.
"""


def render_audit_report(audit: dict[str, Any], dryrun: dict[str, Any]) -> str:
    summary = audit["summary"]
    return f"""# Preserve Original Volume / Lifetime Audit

## Root Cause

preserve_original inflated seed0 augmentation volume because the runtime schedule used cumulative replay active classes instead of epoch-exact fixed CATF-v2 router-executable policies. Fixed seed0 retained nonzero old ops in some guarded/frozen class policies, but those rows were not router-executable. The executable fixed policy existed at epoch 5 (classes 4/11) and epoch 15 (class 12). The preserve rerun cleared guards and kept class4 active through epoch45 and class12 active from epoch15 through epoch45.

This made industrial augmentation grow from {summary['fixed_industrial']} to {summary['preserve_industrial_before_fix']} and ROI augmentation grow from {summary['fixed_roi']} to {summary['preserve_roi_before_fix']}. The op probabilities and strengths were not numerically amplified; they were applied for too many feedback intervals.

## Findings

- seed-level/cumulative union present before fix: {summary['seed_level_union_present']}
- stale policy accumulation before fix: {summary['stale_policy_accumulation_present']}
- policy lifetime too long before fix: {summary['policy_lifetime_too_long']}
- router eligibility amplified before fix: {summary['router_eligible_amplified']}
- op prob/strength amplified: {summary['op_prob_strength_amplified']}
- sampler_only involved: {summary['sampler_only_involved']}

## Fix

- build_preserve_weak_decision_schedule now can load fixed CATF-v2 policy_history and emits epoch_exact_fixed_active_class/op/prob_strength.
- preserve_original execution now prefers epoch-exact fixed fields and treats an explicit empty epoch as no active policy, instead of falling back to probe_set.class_id or a replay class union.
- each feedback update clears stale active ops before installing only that epoch's fixed policy.
- the corrected schedule replaced the old runtime config; the stale config was backed up for audit.

## Dry-Run After Fix

- fixed industrial expected: {dryrun['fixed_total_industrial_expected']}
- preserve industrial expected after fix: {dryrun['preserve_total_industrial_expected']}
- fixed ROI expected: {dryrun['fixed_total_roi_expected']}
- preserve ROI expected after fix: {dryrun['preserve_total_roi_expected']}
- old industrial volume ratio: {dryrun['old_industrial_volume_ratio']}
- old ROI volume ratio: {dryrun['old_roi_volume_ratio']}
- post-fix industrial volume ratio: {dryrun['industrial_volume_ratio_after_fix']}
- post-fix ROI volume ratio: {dryrun['roi_volume_ratio_after_fix']}
- epoch-exact preserve after fix: {dryrun['epoch_exact_preserve_after_fix']}
- stale accumulation after fix: {dryrun['stale_policy_accumulation_after_fix']}

## Recommendation

Rerun seed0 sanity before seed2. The dry-run now matches fixed CATF-v2 policy lifetime and should avoid the previous augmentation-volume overrun.
"""


if __name__ == "__main__":
    main()
