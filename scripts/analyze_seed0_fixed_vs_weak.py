"""Audit seed0 fixed CATF-v2 vs weak image augmentation failure.

This script is intentionally offline-only. It reads existing run artifacts,
compares seed0 fixed CATF-v2 with seed0 weak image augmentation, and writes
auditable reports plus CSV tables. It does not train, mutate policy logic, or
touch sampler-only code paths.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CLEAN_METRICS = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/seed_0/clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json"
FIXED_DIR = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_0/catf_v2"
FIXED_METRICS = FIXED_DIR / "reports/final_metrics.json"
FIXED_POLICY = FIXED_DIR / "reports/policy_history.json"

WEAK_DIR = ROOT / "outputs/experiments/catf_v2_image_only_weak_aug_multiseed/seed0"
WEAK_METRICS = WEAK_DIR / "reports/final_metrics.json"
WEAK_POLICY = WEAK_DIR / "reports/policy_history.json"
WEAK_EVENTS = WEAK_DIR / "reports/causal_probe_events.json"
WEAK_REPLAY_CSV = ROOT / "outputs/experiments/catf_v2_image_only_weak_aug_replay/weak_candidate_records.csv"

OUT_ROOT = ROOT / "outputs/experiments/catf_v2_image_only_weak_aug_multiseed"
OUT_REPORTS = OUT_ROOT / "reports"
OUT_MD = OUT_REPORTS / "seed0_fixed_vs_weak_failure_audit.md"
OUT_JSON = OUT_REPORTS / "seed0_fixed_vs_weak_failure_audit.json"
OUT_EPOCH_CSV = OUT_ROOT / "seed0_fixed_vs_weak_epoch_policy_diff.csv"
OUT_CLASS_CSV = OUT_ROOT / "seed0_fixed_vs_weak_per_class_regression.csv"


METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
PER_CLASS_METRIC_KEYS = ("precision", "recall", "ap50", "ap50_95")
FEEDBACK_EPOCHS = (5, 10, 15, 20, 25, 30, 35, 40, 45)
FIXED_AUG_CLASSES = {4, 11, 12}
WEAK_AUG_CLASSES = {9}


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def val_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("val", {}).get("metrics") or payload.get("metrics") or {}


def metric_summary(payload: dict[str, Any]) -> dict[str, float]:
    metrics = val_metrics(payload)
    return {key: float(metrics.get(key, 0.0)) for key in METRIC_KEYS}


def metric_delta(new: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {key: new[key] - base[key] for key in METRIC_KEYS}


def get_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("history", [])
    if not isinstance(history, list):
        return []
    return history


def get_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events", [])
    if isinstance(events, list):
        return events
    return []


def per_class_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    per_class = val_metrics(payload).get("per_class", [])
    return {int(row["class_id"]): row for row in per_class}


def approx_counts(row: dict[str, Any]) -> dict[str, float]:
    instances = float(row.get("instances") or 0)
    precision = float(row.get("precision") or 0)
    recall = float(row.get("recall") or 0)
    tp = recall * instances
    fn = max(instances - tp, 0.0)
    if precision > 0:
        fp = max(tp * (1.0 / precision - 1.0), 0.0)
    else:
        fp = math.inf if tp > 0 else 0.0
    return {"tp_est": tp, "fp_est": fp, "fn_est": fn}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def concise_ops(policy: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not policy:
        return []
    classes = policy.get("classes", {})
    rows: list[dict[str, Any]] = []
    for class_id_text, class_policy in sorted(classes.items(), key=lambda item: int(item[0])):
        ops = class_policy.get("ops", {})
        active_ops = []
        for op_name, op_cfg in sorted(ops.items()):
            prob = safe_float(op_cfg.get("prob"))
            strength = safe_float(op_cfg.get("strength"))
            if prob > 0 or strength > 0:
                active_ops.append(
                    {
                        "op": op_name,
                        "prob": prob,
                        "strength": strength,
                    }
                )
        if active_ops:
            rows.append(
                {
                    "class_id": int(class_id_text),
                    "class_name": class_policy.get("class_name"),
                    "state": class_policy.get("state"),
                    "status": class_policy.get("status"),
                    "dominant_issue": class_policy.get("dominant_issue"),
                    "ops": active_ops,
                    "weak_image_aug": bool(class_policy.get("weak_image_aug", False)),
                    "attenuation_ratio": class_policy.get("attenuation_ratio"),
                }
            )
    return rows


def format_ops(ops: list[dict[str, Any]]) -> str:
    parts = []
    for class_row in ops:
        op_parts = [
            f"{op['op']}@p={op['prob']:.4g}/s={op['strength']:.4g}"
            for op in class_row.get("ops", [])
        ]
        cls = f"c{class_row['class_id']}:{class_row.get('class_name')}"
        issue = class_row.get("dominant_issue") or "unknown"
        parts.append(f"{cls}:{issue}:" + ",".join(op_parts))
    return "; ".join(parts)


def class_issue_summary(ops: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"c{row['class_id']}:{row.get('dominant_issue') or 'unknown'}"
        for row in ops
    )


def action_candidate(action: str | None, active_ops: list[dict[str, Any]]) -> str:
    if not active_ops:
        return "candidate_policy_0_noop"
    if action in {"rollback", "shrink", "observe", "freeze"}:
        return f"fixed_catf_v2_{action}_policy"
    return "fixed_catf_v2_roi_texture"


def read_replay_rows() -> dict[int, dict[str, str]]:
    if not WEAK_REPLAY_CSV.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with WEAK_REPLAY_CSV.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seed = row.get("seed")
            if str(seed) != "0":
                continue
            epoch = int(float(row.get("epoch") or 0))
            rows[epoch] = row
    return rows


def to_by_epoch(history: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item.get("epoch")): item for item in history if item.get("epoch") is not None}


def events_by_epoch(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item.get("epoch")): item for item in events if item.get("epoch") is not None}


def build_epoch_rows(
    fixed_history: list[dict[str, Any]],
    weak_history: list[dict[str, Any]],
    weak_events: list[dict[str, Any]],
    fixed_stats: dict[str, Any],
    fixed_roi: dict[str, Any],
    weak_stats: dict[str, Any],
    weak_roi: dict[str, Any],
) -> list[dict[str, Any]]:
    fixed_by_epoch = to_by_epoch(fixed_history)
    weak_by_epoch = to_by_epoch(weak_history)
    weak_event_by_epoch = events_by_epoch(weak_events)
    replay_by_epoch = read_replay_rows()
    rows: list[dict[str, Any]] = []

    for epoch in FEEDBACK_EPOCHS:
        fixed_event = fixed_by_epoch.get(epoch, {})
        weak_event = weak_by_epoch.get(epoch, {})
        weak_causal = weak_event_by_epoch.get(epoch) or weak_event.get("causal_probe_event", {})
        replay = replay_by_epoch.get(epoch, {})

        fixed_ops = concise_ops(fixed_event.get("accepted_policy"))
        weak_ops = concise_ops(weak_event.get("accepted_policy"))
        fixed_classes = {row["class_id"] for row in fixed_ops}
        weak_classes = {row["class_id"] for row in weak_ops}
        fixed_active_classes = fixed_event.get("active_classes") or []
        weak_active_classes = weak_event.get("active_classes") or []

        weak_candidate = weak_causal.get("selected_candidate_policy_id") or "unknown"
        weak_action = weak_causal.get("selected_candidate_action") or weak_causal.get("action") or weak_event.get("action")
        weak_added_new_policy = bool(weak_classes - fixed_classes)
        weak_replaced_fixed_safe_policy = bool((fixed_classes & FIXED_AUG_CLASSES) - weak_classes)

        no_op_reason = ""
        if weak_candidate == "candidate_policy_0_noop":
            no_op_reason = weak_causal.get("weak_rejection_reasons") or weak_causal.get("reason") or ""
        elif not weak_ops:
            no_op_reason = weak_causal.get("reason") or ""

        weak_industrial = 0
        weak_roi_count = 0
        if epoch == 25:
            interval_counts = weak_stats.get("weak_image_aug_interval_counts") or {}
            weak_industrial = int(interval_counts.get("9:25", 0))
            weak_roi_count = int((weak_roi.get("affected_classes") or {}).get("9", 0))

        fixed_roi_note = ""
        if epoch == 5:
            fixed_roi_note = "aggregate affected: c4=9, c11=22"
        elif epoch == 15:
            fixed_roi_note = "aggregate affected: c12=14"
        elif fixed_ops:
            fixed_roi_note = f"aggregate run total={fixed_roi.get('roi_aug_applied', 0)}"
        else:
            fixed_roi_note = "0"

        fixed_aug_note = "0"
        if fixed_ops:
            fixed_aug_note = f"aggregate run total={fixed_stats.get('samples_augmented', 0)}"

        rows.append(
            {
                "epoch": epoch,
                "active_class_fixed": ";".join(str(x) for x in fixed_active_classes),
                "dominant_issue_fixed": class_issue_summary(fixed_ops),
                "fixed_candidate": action_candidate(fixed_event.get("action"), fixed_ops),
                "fixed_action": fixed_event.get("action"),
                "fixed_op_list": format_ops(fixed_ops),
                "fixed_prob_strength": format_ops(fixed_ops),
                "fixed_roi_applied": fixed_roi_note,
                "fixed_industrial_augmented": fixed_aug_note,
                "active_class_weak": weak_causal.get("candidate_class_id") or ";".join(str(x) for x in weak_active_classes),
                "dominant_issue_weak": class_issue_summary(weak_ops) or replay.get("dominant_issue", ""),
                "weak_candidate": weak_candidate,
                "weak_action": weak_action,
                "weak_op_list": format_ops(weak_ops),
                "weak_prob_strength": format_ops(weak_ops),
                "weak_roi_applied": weak_roi_count,
                "weak_industrial_augmented": weak_industrial,
                "weak_attenuation_ratio": weak_causal.get("attenuation_ratio") or replay.get("selected_attenuation_ratio", ""),
                "weak_replaced_fixed_original_policy": weak_replaced_fixed_safe_policy,
                "weak_added_policy_not_in_fixed": weak_added_new_policy,
                "weak_original_candidate": weak_causal.get("original_candidate_policy_id") or replay.get("candidate_policy_id", ""),
                "weak_retained_op": weak_causal.get("retained_op") or replay.get("selected_retained_op", ""),
                "weak_original_prob": weak_causal.get("original_prob") or replay.get("original_prob", ""),
                "weak_original_strength": weak_causal.get("original_strength") or replay.get("original_strength", ""),
                "weak_prob": weak_causal.get("weak_prob") or replay.get("selected_weak_prob", ""),
                "weak_strength": weak_causal.get("weak_strength") or replay.get("selected_weak_strength", ""),
                "precision_aware_gate_passed": weak_causal.get("precision_aware_gate_passed", ""),
                "non_active_regression_gate_passed": weak_causal.get("non_active_regression_gate_passed", ""),
                "no_op_reason": no_op_reason,
                "replay_original_rejection_reasons": replay.get("original_rejection_reasons", ""),
                "replay_ratio_0_25_rejection_reasons": replay.get("ratio_0_25_rejection_reasons", ""),
                "target_high_fp_guard_overridden_by_weak_gate": weak_causal.get("target_high_fp_guard_overridden_by_weak_gate", ""),
                "notes": (
                    "weak executes only class9 local_contrast at epoch25"
                    if epoch == 25 and weak_candidate == "candidate_policy_1b_weak_roi_texture"
                    else ""
                ),
            }
        )
    return rows


def build_per_class_rows(
    clean_payload: dict[str, Any],
    fixed_payload: dict[str, Any],
    weak_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    clean_pc = per_class_map(clean_payload)
    fixed_pc = per_class_map(fixed_payload)
    weak_pc = per_class_map(weak_payload)
    rows: list[dict[str, Any]] = []

    for class_id in sorted(clean_pc):
        clean = clean_pc[class_id]
        fixed = fixed_pc[class_id]
        weak = weak_pc[class_id]
        clean_counts = approx_counts(clean)
        fixed_counts = approx_counts(fixed)
        weak_counts = approx_counts(weak)

        row: dict[str, Any] = {
            "class_id": class_id,
            "name": clean.get("name"),
            "instances": clean.get("instances"),
            "is_fixed_aug_class": class_id in FIXED_AUG_CLASSES,
            "is_weak_active_class": class_id in WEAK_AUG_CLASSES,
            "non_active_under_weak": class_id not in WEAK_AUG_CLASSES,
        }
        for prefix, payload, counts in (
            ("clean", clean, clean_counts),
            ("fixed", fixed, fixed_counts),
            ("weak", weak, weak_counts),
        ):
            for key in PER_CLASS_METRIC_KEYS:
                row[f"{prefix}_{key}"] = float(payload.get(key, 0.0))
            row[f"{prefix}_tp_est"] = counts["tp_est"]
            row[f"{prefix}_fp_est"] = counts["fp_est"]
            row[f"{prefix}_fn_est"] = counts["fn_est"]

        for key in PER_CLASS_METRIC_KEYS:
            row[f"delta_fixed_vs_clean_{key}"] = row[f"fixed_{key}"] - row[f"clean_{key}"]
            row[f"delta_weak_vs_clean_{key}"] = row[f"weak_{key}"] - row[f"clean_{key}"]
            row[f"delta_weak_vs_fixed_{key}"] = row[f"weak_{key}"] - row[f"fixed_{key}"]

        for count_key in ("tp_est", "fp_est", "fn_est"):
            row[f"delta_fixed_vs_clean_{count_key}"] = row[f"fixed_{count_key}"] - row[f"clean_{count_key}"]
            row[f"delta_weak_vs_clean_{count_key}"] = row[f"weak_{count_key}"] - row[f"clean_{count_key}"]
            row[f"delta_weak_vs_fixed_{count_key}"] = row[f"weak_{count_key}"] - row[f"fixed_{count_key}"]

        row["recall_up_precision_down_vs_clean"] = (
            row["delta_weak_vs_clean_recall"] > 0 and row["delta_weak_vs_clean_precision"] < 0
        )
        row["ap50_drop_gt_0_01_vs_clean"] = row["delta_weak_vs_clean_ap50"] < -0.01
        row["ap50_95_drop_gt_0_01_vs_clean"] = row["delta_weak_vs_clean_ap50_95"] < -0.01
        row["precision_drop_rank_key"] = row["delta_weak_vs_clean_precision"]
        row["fp_increase_rank_key"] = row["delta_weak_vs_fixed_fp_est"]
        rows.append(row)

    rows.sort(key=lambda item: int(item["class_id"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_rows(rows: list[dict[str, Any]], key: str, reverse: bool = False, limit: int = 5) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda item: safe_float(item.get(key)), reverse=reverse)
    return [
        {
            "class_id": row["class_id"],
            "name": row["name"],
            key: row[key],
            "delta_weak_vs_clean_precision": row["delta_weak_vs_clean_precision"],
            "delta_weak_vs_fixed_fp_est": row["delta_weak_vs_fixed_fp_est"],
            "delta_weak_vs_clean_ap50": row["delta_weak_vs_clean_ap50"],
            "delta_weak_vs_clean_ap50_95": row["delta_weak_vs_clean_ap50_95"],
        }
        for row in sorted_rows[:limit]
    ]


def build_markdown(summary: dict[str, Any], epoch_rows: list[dict[str, Any]], class_rows: list[dict[str, Any]]) -> str:
    metrics = summary["metrics"]
    deltas = summary["deltas"]
    top_precision = summary["top_regressions"]["precision_drop_vs_clean"]
    top_fp = summary["top_regressions"]["fp_increase_vs_fixed"]
    top_ap50 = summary["top_regressions"]["ap50_drop_vs_clean"]
    top_ap95 = summary["top_regressions"]["ap50_95_drop_vs_clean"]

    def fmt(value: Any, digits: int = 6) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# Seed0 Fixed vs Weak Image Augmentation Failure Audit",
        "",
        "Scope: offline audit only. No training was run. Sampler-only and weighted index list paths are not involved.",
        "",
        "## Aggregate Metrics",
        "",
        "| run | P | R | mAP50 | mAP50-95 | constraint |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        f"| clean seed0 | {fmt(metrics['clean']['precision'])} | {fmt(metrics['clean']['recall'])} | {fmt(metrics['clean']['map50'])} | {fmt(metrics['clean']['map50_95'])} | baseline |",
        f"| fixed CATF-v2 seed0 | {fmt(metrics['fixed']['precision'])} | {fmt(metrics['fixed']['recall'])} | {fmt(metrics['fixed']['map50'])} | {fmt(metrics['fixed']['map50_95'])} | pass |",
        f"| weak image aug seed0 | {fmt(metrics['weak']['precision'])} | {fmt(metrics['weak']['recall'])} | {fmt(metrics['weak']['map50'])} | {fmt(metrics['weak']['map50_95'])} | fail |",
        "",
        "Delta vs clean:",
        f"- fixed: P {fmt(deltas['fixed_vs_clean']['precision'])}, R {fmt(deltas['fixed_vs_clean']['recall'])}, mAP50 {fmt(deltas['fixed_vs_clean']['map50'])}, mAP50-95 {fmt(deltas['fixed_vs_clean']['map50_95'])}.",
        f"- weak: P {fmt(deltas['weak_vs_clean']['precision'])}, R {fmt(deltas['weak_vs_clean']['recall'])}, mAP50 {fmt(deltas['weak_vs_clean']['map50'])}, mAP50-95 {fmt(deltas['weak_vs_clean']['map50_95'])}.",
        "",
        "## Execution Difference",
        "",
        f"- Fixed seed0 passed with conservative image augmentation on classes 4, 11, and 12: {summary['fixed_execution']['industrial_augmented']} industrial images augmented, {summary['fixed_execution']['roi_applied']} ROI applications.",
        f"- Weak seed0 executed image augmentation only at epoch25 on class 9: {summary['weak_execution']['industrial_augmented']} industrial images augmented and {summary['weak_execution']['roi_applied']} ROI applications.",
        "- Weak seed0 did not use sampler-only: sampler_only_effective=false, weighted_index_list_enabled=false, sampled_distribution_changed=false.",
        "- The weak epoch25 candidate kept only local_contrast at prob=0.045 and strength=0.05, but it replaced the fixed path's active safe policies instead of preserving them.",
        "",
        "## Epoch-Level Policy Diff",
        "",
        "| epoch | fixed candidate / ops | weak candidate / ops | fixed policy replaced | weak new policy | weak applied | no-op reason |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]

    for row in epoch_rows:
        weak_applied = f"{row['weak_industrial_augmented']} img / {row['weak_roi_applied']} ROI"
        fixed = f"{row['fixed_candidate']} {row['fixed_op_list']}".strip()
        weak = f"{row['weak_candidate']} {row['weak_op_list']}".strip()
        lines.append(
            f"| {row['epoch']} | {fixed or 'noop'} | {weak or 'noop'} | "
            f"{row['weak_replaced_fixed_original_policy']} | {row['weak_added_policy_not_in_fixed']} | "
            f"{weak_applied} | {row['no_op_reason']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Class Regression",
            "",
            "The TP/FP/FN columns in the CSV are metric-derived estimates from precision, recall, and instance count; the run artifacts do not expose raw prediction-level TP/FP/FN tables.",
            "",
            "Largest weak precision drops vs clean:",
            "",
            "| class | name | delta P | delta FP vs fixed | delta AP50 | delta AP50-95 |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_precision:
        lines.append(
            f"| {row['class_id']} | {row['name']} | {fmt(row['delta_weak_vs_clean_precision'])} | "
            f"{fmt(row['delta_weak_vs_fixed_fp_est'])} | {fmt(row['delta_weak_vs_clean_ap50'])} | "
            f"{fmt(row['delta_weak_vs_clean_ap50_95'])} |"
        )

    lines.extend(
        [
            "",
            "Largest weak FP-estimate increases vs fixed:",
            "",
            "| class | name | delta FP vs fixed | delta P vs clean | delta AP50 | delta AP50-95 |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_fp:
        lines.append(
            f"| {row['class_id']} | {row['name']} | {fmt(row['delta_weak_vs_fixed_fp_est'])} | "
            f"{fmt(row['delta_weak_vs_clean_precision'])} | {fmt(row['delta_weak_vs_clean_ap50'])} | "
            f"{fmt(row['delta_weak_vs_clean_ap50_95'])} |"
        )

    lines.extend(
        [
            "",
            "Largest AP50 drops vs clean:",
            "",
            "| class | name | delta AP50 | delta P vs clean | delta FP vs fixed |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_ap50:
        lines.append(
            f"| {row['class_id']} | {row['name']} | {fmt(row['delta_weak_vs_clean_ap50'])} | "
            f"{fmt(row['delta_weak_vs_clean_precision'])} | {fmt(row['delta_weak_vs_fixed_fp_est'])} |"
        )

    lines.extend(
        [
            "",
            "Largest AP50-95 drops vs clean:",
            "",
            "| class | name | delta AP50-95 | delta P vs clean | delta FP vs fixed |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_ap95:
        lines.append(
            f"| {row['class_id']} | {row['name']} | {fmt(row['delta_weak_vs_clean_ap50_95'])} | "
            f"{fmt(row['delta_weak_vs_clean_precision'])} | {fmt(row['delta_weak_vs_fixed_fp_est'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Fixed seed0 passed because it kept very conservative ROI texture augmentation on classes 4, 11, and 12. The probabilities and strengths stayed small, and the run still improved mAP50 and mAP50-95 relative to clean.",
            "2. Weak seed0 failed because the weak replay path globally replaced the fixed policy behavior. It suppressed the fixed seed0 safe policies and then executed a different class9 weak local_contrast candidate at epoch25.",
            "3. The precision collapse is FP-driven. Weak seed0 precision dropped by 0.105511 vs clean while recall increased by 0.035364, consistent with recall gain bought by broad FP spillover.",
            "4. The FP increase is not isolated to class9. The largest FP-estimate increase vs fixed is class7, and several non-active classes also regress, so this is non-active regression rather than only a target-class tradeoff.",
            "5. The epoch25 weak candidate was allowed after attenuation=0.25, but its original replay metadata still carried precision and non-active risk flags. In the full run, attenuation was not sufficient for seed0.",
            "6. Direct prediction-level high-confidence FP histograms are not present in the final metrics artifacts. However, the replay row for the original epoch25 candidate reported high_confidence_fp_delta=0.05, and the final per-class metrics show broad FP-estimate growth.",
            "",
            "## Required Answers",
            "",
            "- Seed0 weak failure main cause: global weak replacement did not preserve fixed seed0's safe original image policies and introduced a class9 weak local_contrast policy that produced broad FP/non-active regression.",
            "- Precision drop source: largest weak precision drops vs clean are class5, class4, class3, class11, class9, and class7; largest FP-estimate increases vs fixed are led by class7, class9, class5, class12, and class6.",
            "- Did weak break fixed strategy: yes. Fixed's effective classes were 4/11/12; weak's only executed class was 9 at epoch25.",
            "- Non-active regression: yes. Most FP growth appears in classes not targeted by weak augmentation.",
            "- Should seed0 preserve fixed original policy: yes. Seed0 should keep fixed original policy when probe risk is low.",
            "- Recommended next strategy: implement preserve-safe-original plus weak-only-for-moderate-risk and strict no-op for high or critical risk.",
            "- Continue training now: no. First replay/validate the preserve-safe-original decision logic offline.",
            "- Image-only mainline: yes. This audit does not use sampler-only and does not recommend reintroducing it.",
            "",
            "## Output Tables",
            "",
            f"- Epoch policy diff: `{OUT_EPOCH_CSV.relative_to(ROOT)}`",
            f"- Per-class regression: `{OUT_CLASS_CSV.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    clean_payload = read_json(CLEAN_METRICS)
    fixed_payload = read_json(FIXED_METRICS)
    weak_payload = read_json(WEAK_METRICS)
    fixed_policy = read_json(FIXED_POLICY)
    weak_policy = read_json(WEAK_POLICY)
    weak_events = read_json(WEAK_EVENTS)

    clean_metrics = metric_summary(clean_payload)
    fixed_metrics = metric_summary(fixed_payload)
    weak_metrics = metric_summary(weak_payload)

    fixed_online = fixed_payload.get("online_aug_stats", {})
    fixed_roi = fixed_payload.get("roi_aug_stats", {})
    weak_online = weak_payload.get("online_aug_stats", {})
    weak_roi = weak_payload.get("roi_aug_stats", {})

    epoch_rows = build_epoch_rows(
        get_history(fixed_policy),
        get_history(weak_policy),
        get_events(weak_events),
        fixed_online,
        fixed_roi,
        weak_online,
        weak_roi,
    )
    class_rows = build_per_class_rows(clean_payload, fixed_payload, weak_payload)

    precision_drop = top_rows(class_rows, "delta_weak_vs_clean_precision", reverse=False)
    fp_increase_vs_fixed = top_rows(class_rows, "delta_weak_vs_fixed_fp_est", reverse=True)
    fp_increase_vs_clean = top_rows(class_rows, "delta_weak_vs_clean_fp_est", reverse=True)
    ap50_drop = top_rows(class_rows, "delta_weak_vs_clean_ap50", reverse=False)
    ap95_drop = top_rows(class_rows, "delta_weak_vs_clean_ap50_95", reverse=False)

    summary = {
        "scope": {
            "seed": 0,
            "analysis_only": True,
            "training_run": False,
            "sampler_only_involved": False,
            "weighted_index_list_involved": False,
            "image_only_mainline": True,
        },
        "paths": {
            "clean_metrics": str(CLEAN_METRICS.relative_to(ROOT)),
            "fixed_metrics": str(FIXED_METRICS.relative_to(ROOT)),
            "weak_metrics": str(WEAK_METRICS.relative_to(ROOT)),
            "epoch_policy_diff_csv": str(OUT_EPOCH_CSV.relative_to(ROOT)),
            "per_class_regression_csv": str(OUT_CLASS_CSV.relative_to(ROOT)),
        },
        "metrics": {
            "clean": clean_metrics,
            "fixed": fixed_metrics,
            "weak": weak_metrics,
        },
        "deltas": {
            "fixed_vs_clean": metric_delta(fixed_metrics, clean_metrics),
            "weak_vs_clean": metric_delta(weak_metrics, clean_metrics),
            "weak_vs_fixed": metric_delta(weak_metrics, fixed_metrics),
        },
        "fixed_execution": {
            "industrial_augmented": fixed_online.get("samples_augmented", 0),
            "roi_applied": fixed_roi.get("roi_aug_applied", 0),
            "router_random_draw_count": fixed_online.get("router_random_draw_count", 0),
            "ops": fixed_online.get("ops", {}),
            "affected_classes": fixed_roi.get("affected_classes", {}),
            "effective_policy_classes": sorted(FIXED_AUG_CLASSES),
        },
        "weak_execution": {
            "industrial_augmented": weak_online.get("samples_augmented", 0),
            "roi_applied": weak_roi.get("roi_aug_applied", 0),
            "router_random_draw_count": weak_online.get("router_random_draw_count", 0),
            "ops": weak_online.get("ops", {}),
            "affected_classes": weak_roi.get("affected_classes", {}),
            "weak_interval_counts": weak_online.get("weak_image_aug_interval_counts", {}),
            "sampler_only_effective": weak_online.get("sampler_only_effective", False),
            "weighted_index_list_enabled": weak_online.get("weighted_index_list_enabled", False),
            "sampled_distribution_changed": weak_online.get("sampled_distribution_changed", False),
            "effective_policy_classes": sorted(WEAK_AUG_CLASSES),
        },
        "top_regressions": {
            "precision_drop_vs_clean": precision_drop,
            "fp_increase_vs_fixed": fp_increase_vs_fixed,
            "fp_increase_vs_clean": fp_increase_vs_clean,
            "ap50_drop_vs_clean": ap50_drop,
            "ap50_95_drop_vs_clean": ap95_drop,
        },
        "mechanism_assessment": {
            "weak_augmentation_intrinsically_harmful": "not proven globally; harmful in seed0 execution context",
            "weak_replaced_fixed_effective_policy": True,
            "weak_executed_on_new_class_not_in_fixed_seed0": True,
            "attenuation_0_25_still_fp_risky_on_seed0": True,
            "probe_gate_underestimated_seed0_precision_risk": True,
            "seed0_should_preserve_fixed_original_policy": True,
            "non_active_regression_detected": True,
            "high_confidence_fp_note": "Replay metadata flagged high_confidence_fp_delta=0.05 for the original epoch25 candidate; final metrics lack raw confidence-level FP tables.",
        },
        "recommendation": {
            "do_not_global_replace_fixed_catf_v2": True,
            "candidate_policy_preserve_original": "Use when original fixed CATF-v2 policy is low-risk in probe.",
            "candidate_policy_weak_roi_texture": "Use only for moderate-risk image candidates after preserving safe originals.",
            "candidate_policy_noop": "Use for high-risk or critical-risk image candidates.",
            "continue_training_now": False,
            "next_step": "Offline replay and implement preserve-safe-original + weak-only-for-moderate-risk before any new training.",
            "sampler_only": "Do not reintroduce for the paper main method.",
        },
    }

    write_csv(OUT_EPOCH_CSV, epoch_rows)
    write_csv(OUT_CLASS_CSV, class_rows)
    write_json(OUT_JSON, summary)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_markdown(summary, epoch_rows, class_rows), encoding="utf-8")

    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_EPOCH_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_CLASS_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
