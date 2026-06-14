from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTISEED_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "catf_v2_image_only_weak_aug_multiseed"
SEED2_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "catf_v2_image_only_weak_aug_seed2_50ep"

METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
RECALL_WARNING_DROP = -0.01


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize image-only weak augmentation multiseed runs.")
    parser.add_argument("--output-root", default=str(MULTISEED_ROOT), help="Multiseed output root.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    seed_reports = [build_seed_report(seed, run_root_for_seed(seed, output_root)) for seed in (0, 1, 2)]
    for report in seed_reports:
        seed = int(report["seed"])
        if seed in (0, 1):
            seed_dir = run_root_for_seed(seed, output_root) / "reports"
            seed_dir.mkdir(parents=True, exist_ok=True)
            write_json(seed_dir / f"seed{seed}_weak_image_aug_report.json", report)
            (seed_dir / f"seed{seed}_weak_image_aug_report.md").write_text(
                render_seed_markdown(report),
                encoding="utf-8",
            )

    summary = build_multiseed_summary(seed_reports)
    write_json(reports_dir / "weak_image_aug_multiseed_summary.json", summary)
    (reports_dir / "weak_image_aug_multiseed_summary.md").write_text(
        render_summary_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))


def run_root_for_seed(seed: int, output_root: Path) -> Path:
    if seed == 2:
        return SEED2_ROOT
    return output_root / f"seed{seed}"


def build_seed_report(seed: int, run_root: Path) -> dict[str, Any]:
    final = read_json(run_root / "reports" / "final_metrics.json")
    online = read_json(run_root / "reports" / "online_aug_stats.json")
    roi = read_json(run_root / "reports" / "roi_aug_stats.json")
    constraint_payload = read_json(run_root / "reports" / "constraint_scoring.json")
    events = read_events(run_root / "reports" / "causal_probe_events.json")
    train_rows = read_results_csv(run_root / "train" / "results.csv")

    clean = metrics_payload(read_json(clean_metrics_path(seed)))
    fixed = fixed_metrics(seed)
    weak = metrics_payload(final)

    delta_clean = metric_delta(weak, clean)
    delta_fixed = metric_delta(weak, fixed)
    constraint_failed, failure_reasons = constraint_status(delta_clean)
    weak_events = [
        event for event in events if event.get("selected_candidate_policy_id") == "candidate_policy_1b_weak_roi_texture"
    ]
    strict_noop_events = [event for event in events if event.get("selected_candidate_action") in {"strict_noop", "strict_no_op"}]
    weak_class_ids = {
        int(event["candidate_class_id"])
        for event in weak_events
        if event.get("candidate_class_id") is not None and int(event.get("candidate_class_id", -1)) >= 0
    }
    class_rows = class_comparison(clean, fixed, weak)
    non_active = non_active_regression_summary(class_rows, weak_class_ids=weak_class_ids)

    final_summary = final.get("summary", {}) if isinstance(final.get("summary", {}), dict) else {}
    sampler_only_enabled = bool(
        final_summary.get("sampler_only_enabled", False) or online.get("sampler_only_enabled", False)
    )
    weighted_index_list_enabled = bool(
        final_summary.get("weighted_index_list_enabled", False) or online.get("weighted_index_list_enabled", False)
    )
    sampled_distribution_changed = bool(
        final_summary.get("sampled_distribution_changed", False) or online.get("sampled_distribution_changed", False)
    )
    final_val_used_for_policy_selection = any(bool(event.get("final_val_used_for_policy_selection", False)) for event in events)
    samples_augmented = int(online.get("samples_augmented", 0) or 0)
    roi_aug_applied = int(roi.get("roi_aug_applied", 0) or 0)
    recall_warning = delta_clean["recall"] < RECALL_WARNING_DROP
    fixed_gain_retained = bool(
        not constraint_failed
        and delta_clean["map50_95"] >= 0.0
        and delta_fixed["map50_95"] >= -0.01
    )

    summary = {
        "seed": seed,
        "completed_50ep": bool(final.get("train", {}).get("success") and final.get("val", {}).get("success") and len(train_rows) == 50),
        "weak_image_augmentation_executed": samples_augmented > 0,
        "industrial_image_augmented": samples_augmented,
        "roi_aug_applied": roi_aug_applied,
        "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
        "sampler_only_enabled": sampler_only_enabled,
        "weighted_index_list_enabled": weighted_index_list_enabled,
        "sampled_distribution_changed": sampled_distribution_changed,
        "final_val_used_for_policy_selection": final_val_used_for_policy_selection,
        "constraint_failed": constraint_failed,
        "failure_reasons": failure_reasons,
        "recall_warning": recall_warning,
        "recall_warning_threshold": RECALL_WARNING_DROP,
        "fixed_catf_v2_benefit_retained": fixed_gain_retained,
        "new_non_active_regression": bool(non_active["mean_ap50_95_delta_weak_vs_fixed"] < -0.01),
    }
    return {
        "seed": seed,
        "run_root": str(run_root),
        "clean": metric_subset(clean),
        "fixed_catf_v2": metric_subset(fixed),
        "weak_image_aug": metric_subset(weak),
        "delta_vs_clean": delta_clean,
        "delta_vs_fixed_catf_v2": delta_fixed,
        "constraint_scoring": {
            "computed_constraint_failed": constraint_failed,
            "computed_failure_reasons": failure_reasons,
            "source_constraint_failed": bool(constraint_payload.get("constraint_failed", constraint_failed)),
            "source_failure_reasons": constraint_payload.get("failure_reasons", []),
            "rules": {
                "precision_drop_gt_0.01": True,
                "map50_drop_gt_0.01": True,
                "map50_95_drop_gt_0.01": True,
                "recall_is_warning_not_hard_constraint": True,
            },
        },
        "weak_candidate_events": weak_events,
        "strict_noop_events": strict_noop_events,
        "weak_candidate_count": len(weak_events),
        "strict_noop_count": len(strict_noop_events),
        "weak_candidate_ops": [weak_event_summary(event) for event in weak_events],
        "online_augmentation": {
            "samples_seen": int(online.get("samples_seen", 0) or 0),
            "samples_augmented": samples_augmented,
            "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
            "weak_image_aug_interval_counts": online.get("weak_image_aug_interval_counts", {}),
            "invalid_bbox_count": int(online.get("invalid_bbox_count", 0) or 0),
            "bbox_oob_count": int(online.get("bbox_oob_count", 0) or 0),
            "class_id_oob_count": int(online.get("class_id_oob_count", 0) or 0),
            "sample_weight_map_generated": bool(online.get("sample_weight_map_generated", False)),
            "weighted_train_core_images_count": int(online.get("weighted_train_core_images_count", 0) or 0),
            "weighted_index_list_enabled": weighted_index_list_enabled,
            "sampled_distribution_changed": sampled_distribution_changed,
            "sampler_only_effective": bool(online.get("sampler_only_effective", False)),
            "sampler_only_status": online.get("sampler_only_status", "not_requested"),
        },
        "roi_augmentation": roi,
        "class_comparison": class_rows,
        "non_active_regression_summary": non_active,
        "summary": summary,
        "paths": {
            "final_metrics": str(run_root / "reports" / "final_metrics.json"),
            "online_aug_stats": str(run_root / "reports" / "online_aug_stats.json"),
            "roi_aug_stats": str(run_root / "reports" / "roi_aug_stats.json"),
            "causal_probe_events": str(run_root / "reports" / "causal_probe_events.json"),
            "results_csv": str(run_root / "train" / "results.csv"),
        },
    }


def weak_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    injected = event.get("candidate_ops_injected") or []
    injected_op = injected[0] if injected else {}
    return {
        "epoch": int(event.get("epoch", -1)),
        "candidate_policy_id": event.get("selected_candidate_policy_id"),
        "original_candidate_policy_id": event.get("original_candidate_policy_id"),
        "candidate_class_id": event.get("candidate_class_id"),
        "retained_op": event.get("retained_op") or injected_op.get("op_name"),
        "op_whitelist": event.get("op_whitelist", []),
        "original_prob": event.get("original_prob"),
        "original_strength": event.get("original_strength"),
        "weak_prob": event.get("weak_prob") or injected_op.get("prob"),
        "weak_strength": event.get("weak_strength") or injected_op.get("strength"),
        "attenuation_ratio": event.get("attenuation_ratio"),
        "max_aug_samples_per_interval": event.get("max_aug_samples_per_interval"),
        "precision_aware_gate_passed": event.get("precision_aware_gate_passed"),
        "non_active_regression_gate_passed": event.get("non_active_regression_gate_passed"),
        "sample_router_allowed": event.get("sample_router_allowed"),
        "image_modification_allowed": event.get("image_modification_allowed"),
        "sample_weighting_allowed": event.get("sample_weighting_allowed"),
    }


def build_multiseed_summary(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for report in seed_reports:
        rows.append(
            {
                "seed": report["seed"],
                "clean": report["clean"],
                "fixed_catf_v2": report["fixed_catf_v2"],
                "weak_image_aug": report["weak_image_aug"],
                "delta_vs_clean": report["delta_vs_clean"],
                "delta_vs_fixed_catf_v2": report["delta_vs_fixed_catf_v2"],
                "constraint_failed": report["summary"]["constraint_failed"],
                "recall_warning": report["summary"]["recall_warning"],
                "weak_image_augmentation_executed": report["summary"]["weak_image_augmentation_executed"],
                "industrial_image_augmented": report["summary"]["industrial_image_augmented"],
                "roi_aug_applied": report["summary"]["roi_aug_applied"],
                "sampler_only_enabled": report["summary"]["sampler_only_enabled"],
                "weighted_index_list_enabled": report["summary"]["weighted_index_list_enabled"],
                "sampled_distribution_changed": report["summary"]["sampled_distribution_changed"],
                "final_val_used_for_policy_selection": report["summary"]["final_val_used_for_policy_selection"],
            }
        )
    weak_metrics = [report["weak_image_aug"] for report in seed_reports]
    clean_metrics = [report["clean"] for report in seed_reports]
    fixed_metrics = [report["fixed_catf_v2"] for report in seed_reports]
    delta_clean = [report["delta_vs_clean"] for report in seed_reports]
    delta_fixed = [report["delta_vs_fixed_catf_v2"] for report in seed_reports]
    pass_count = sum(1 for report in seed_reports if not report["summary"]["constraint_failed"])
    total_industrial = sum(int(report["summary"]["industrial_image_augmented"]) for report in seed_reports)
    total_roi = sum(int(report["summary"]["roi_aug_applied"]) for report in seed_reports)
    sampler_only_any = any(report["summary"]["sampler_only_enabled"] for report in seed_reports)
    weighted_any = any(report["summary"]["weighted_index_list_enabled"] for report in seed_reports)
    sampled_changed_any = any(report["summary"]["sampled_distribution_changed"] for report in seed_reports)
    recall_warning_any = any(report["summary"]["recall_warning"] for report in seed_reports)
    summary = {
        "seed_count": len(seed_reports),
        "constraint_pass_count": pass_count,
        "three_seed_pass": pass_count == len(seed_reports),
        "mean_clean": mean_metrics(clean_metrics),
        "mean_fixed_catf_v2": mean_metrics(fixed_metrics),
        "mean_weak_image_aug": mean_metrics(weak_metrics),
        "mean_delta_vs_clean": mean_metrics(delta_clean),
        "mean_delta_vs_fixed_catf_v2": mean_metrics(delta_fixed),
        "total_industrial_image_augmented": total_industrial,
        "total_roi_aug_applied": total_roi,
        "sampler_only_involved": sampler_only_any,
        "weighted_index_list_involved": weighted_any,
        "sampled_distribution_changed_any": sampled_changed_any,
        "final_val_used_for_policy_selection_any": any(
            report["summary"]["final_val_used_for_policy_selection"] for report in seed_reports
        ),
        "recall_warning_any": recall_warning_any,
        "main_method_candidate": bool(
            pass_count == len(seed_reports)
            and not sampler_only_any
            and not weighted_any
            and not sampled_changed_any
            and mean_metrics(delta_clean)["map50_95"] > 0.0
        ),
        "recommend_recall_aware_constraint": recall_warning_any,
    }
    return {
        "rows": rows,
        "seed_reports": seed_reports,
        "summary": summary,
        "interpretation": {
            "sampler_only": "Sampler-only was demoted and did not participate in these image-only weak augmentation runs.",
            "seed2_recall": "Seed2 passes the hard constraints but recall remains below clean; this is a recall warning, not a hard-constraint failure.",
            "mainline": (
                "Image-only weak CP-CATF is a main-method candidate only if all three seeds pass constraints "
                "and the mean mAP50-95 delta vs clean is positive."
            ),
        },
    }


def clean_metrics_path(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / "multiseed_clean_yolo_default_vs_catf_v2"
        / f"seed_{seed}"
        / "clean_native_yolo_default"
        / "reports"
        / "clean_native_yolo_default_metrics.json"
    )


def fixed_metrics(seed: int) -> dict[str, Any]:
    summary_path = (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / "multiseed_clean_yolo_default_vs_catf_v2_fixed"
        / "reports"
        / "multiseed_catf_v2_fixed_summary.json"
    )
    summary = read_json(summary_path)
    return summary["seeds"][str(seed)]["fixed_catf_v2"]


def metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return (payload.get("val") or {}).get("metrics") or payload.get("metrics") or payload


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0) or 0.0) for key in METRIC_KEYS}


def metric_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(current.get(key, 0.0) or 0.0) - float(reference.get(key, 0.0) or 0.0)
        for key in METRIC_KEYS
    }


def mean_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, float]:
    return {
        key: mean(float(metrics.get(key, 0.0) or 0.0) for metrics in metrics_list)
        for key in METRIC_KEYS
    }


def constraint_status(delta_clean: dict[str, float]) -> tuple[bool, list[str]]:
    reasons = []
    if delta_clean["precision"] < -0.01:
        reasons.append("precision_drop_gt_0.01")
    if delta_clean["map50"] < -0.01:
        reasons.append("map50_drop_gt_0.01")
    if delta_clean["map50_95"] < -0.01:
        reasons.append("map50_95_drop_gt_0.01")
    return bool(reasons), reasons


def per_class_map(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(int(row["class_id"])): row for row in metrics.get("per_class", []) if row.get("class_id") is not None}


def class_comparison(clean: dict[str, Any], fixed: dict[str, Any], weak: dict[str, Any]) -> dict[str, Any]:
    clean_rows = per_class_map(clean)
    fixed_rows = per_class_map(fixed)
    weak_rows = per_class_map(weak)
    out: dict[str, Any] = {}
    for class_id in sorted(set(clean_rows) | set(fixed_rows) | set(weak_rows), key=int):
        c = clean_rows.get(class_id, {})
        f = fixed_rows.get(class_id, {})
        w = weak_rows.get(class_id, {})
        keys = ("precision", "recall", "ap50", "ap50_95")
        out[class_id] = {
            "class_id": int(class_id),
            "name": w.get("name") or c.get("name") or f.get("name"),
            "clean": {key: c.get(key) for key in keys},
            "fixed": {key: f.get(key) for key in keys},
            "weak": {key: w.get(key) for key in keys},
            "delta_weak_vs_clean": {
                key: float(w.get(key, 0.0) or 0.0) - float(c.get(key, 0.0) or 0.0)
                for key in keys
            },
            "delta_weak_vs_fixed": {
                key: float(w.get(key, 0.0) or 0.0) - float(f.get(key, 0.0) or 0.0)
                for key in keys
            },
        }
    return out


def non_active_regression_summary(class_rows: dict[str, Any], *, weak_class_ids: set[int]) -> dict[str, Any]:
    non_active = [row for cid, row in class_rows.items() if int(cid) not in weak_class_ids]
    if not non_active:
        return {
            "weak_target_class_ids": sorted(weak_class_ids),
            "non_active_class_count": 0,
            "mean_ap50_95_delta_weak_vs_fixed": 0.0,
            "mean_ap50_95_delta_weak_vs_clean": 0.0,
            "ap50_95_regressed_vs_fixed_gt_0_01_count": 0,
            "ap50_95_regressed_vs_clean_gt_0_01_count": 0,
            "regressed_vs_fixed_classes": [],
            "regressed_vs_clean_classes": [],
        }
    regressed_fixed = [
        row for row in non_active if row["delta_weak_vs_fixed"].get("ap50_95", 0.0) < -0.01
    ]
    regressed_clean = [
        row for row in non_active if row["delta_weak_vs_clean"].get("ap50_95", 0.0) < -0.01
    ]
    return {
        "weak_target_class_ids": sorted(weak_class_ids),
        "non_active_class_count": len(non_active),
        "mean_ap50_95_delta_weak_vs_fixed": mean(
            row["delta_weak_vs_fixed"].get("ap50_95", 0.0) for row in non_active
        ),
        "mean_ap50_95_delta_weak_vs_clean": mean(
            row["delta_weak_vs_clean"].get("ap50_95", 0.0) for row in non_active
        ),
        "ap50_95_regressed_vs_fixed_gt_0_01_count": len(regressed_fixed),
        "ap50_95_regressed_vs_clean_gt_0_01_count": len(regressed_clean),
        "regressed_vs_fixed_classes": compact_regressed_rows(regressed_fixed, "delta_weak_vs_fixed"),
        "regressed_vs_clean_classes": compact_regressed_rows(regressed_clean, "delta_weak_vs_clean"),
    }


def compact_regressed_rows(rows: list[dict[str, Any]], delta_key: str) -> list[dict[str, Any]]:
    return [
        {
            "class_id": row["class_id"],
            "name": row["name"],
            "delta_ap50_95": row[delta_key].get("ap50_95"),
        }
        for row in rows
    ]


def render_seed_markdown(report: dict[str, Any]) -> str:
    seed = report["seed"]
    summary = report["summary"]
    online = report["online_augmentation"]
    roi = report["roi_augmentation"]
    lines = [
        f"# Seed{seed} Image-Only Weak Augmentation Sanity",
        "",
        "## Scope",
        "",
        f"- Run root: `{report['run_root']}`",
        f"- Seed: `{seed}`",
        "- Method: image-only weak ROI texture attenuation",
        "- Attenuation ratio: `0.25`",
        "- Sampler-only: `disabled`",
        "- Weighted index list: `disabled`",
        "- Sampled distribution changed: `false`",
        "",
        "## Execution",
        "",
        f"- 50ep completed: `{str(summary['completed_50ep']).lower()}`",
        f"- Weak image augmentation executed: `{str(summary['weak_image_augmentation_executed']).lower()}`",
        f"- Industrial images augmented: `{summary['industrial_image_augmented']}`",
        f"- ROI applied: `{summary['roi_aug_applied']}`",
        f"- Router random draw count: `{summary['router_random_draw_count']}`",
        f"- Invalid bbox count: `{online['invalid_bbox_count']}`",
        f"- BBox out-of-bounds count: `{online['bbox_oob_count']}`",
        f"- Class-id out-of-bounds count: `{online['class_id_oob_count']}`",
        f"- Final val used for policy selection: `{str(summary['final_val_used_for_policy_selection']).lower()}`",
        "",
        "## Metrics",
        "",
        "| baseline | P | R | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        metric_row("clean", report["clean"]),
        metric_row("fixed CATF-v2", report["fixed_catf_v2"]),
        metric_row("weak image aug", report["weak_image_aug"]),
        "",
        "## Deltas",
        "",
        "| comparison | dP | dR | d mAP50 | d mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        metric_row("weak - clean", report["delta_vs_clean"]),
        metric_row("weak - fixed", report["delta_vs_fixed_catf_v2"]),
        "",
        "## Constraint",
        "",
        f"- Constraint failed: `{str(summary['constraint_failed']).lower()}`",
        f"- Failure reasons: `{', '.join(summary['failure_reasons']) if summary['failure_reasons'] else 'none'}`",
        f"- Recall warning: `{str(summary['recall_warning']).lower()}`",
        f"- Fixed CATF-v2 benefit retained: `{str(summary['fixed_catf_v2_benefit_retained']).lower()}`",
        f"- New non-active regression vs fixed: `{str(summary['new_non_active_regression']).lower()}`",
        "",
        "## Candidate Decisions",
        "",
        "| epoch | decision | class | retained op | weak prob | weak strength | gates |",
        "| ---: | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for event in report["weak_candidate_ops"]:
        gates = (
            f"precision={str(event.get('precision_aware_gate_passed')).lower()}, "
            f"non_active={str(event.get('non_active_regression_gate_passed')).lower()}"
        )
        lines.append(
            "| {epoch} | weak_roi_texture | {class_id} | {op} | {prob:.6f} | {strength:.6f} | {gates} |".format(
                epoch=event["epoch"],
                class_id=event.get("candidate_class_id"),
                op=event.get("retained_op"),
                prob=float(event.get("weak_prob") or 0.0),
                strength=float(event.get("weak_strength") or 0.0),
                gates=gates,
            )
        )
    if not report["weak_candidate_ops"]:
        lines.append("| - | none | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Augmentation Counts",
            "",
            f"- Weak interval counts: `{json.dumps(online['weak_image_aug_interval_counts'], ensure_ascii=False)}`",
            f"- ROI affected classes: `{json.dumps(roi.get('affected_classes', {}), ensure_ascii=False)}`",
            f"- Strict no-op count: `{report['strict_noop_count']}`",
            f"- Sampler-only effective: `{str(online['sampler_only_effective']).lower()}`",
            f"- Sample weight map generated: `{str(online['sample_weight_map_generated']).lower()}`",
            f"- Weighted train-core images: `{online['weighted_train_core_images_count']}`",
            "",
            "## Non-Active Regression",
            "",
            f"- Mean AP50-95 delta vs fixed: `{report['non_active_regression_summary']['mean_ap50_95_delta_weak_vs_fixed']:.6f}`",
            f"- Mean AP50-95 delta vs clean: `{report['non_active_regression_summary']['mean_ap50_95_delta_weak_vs_clean']:.6f}`",
            f"- Regressed vs fixed count (>0.01): `{report['non_active_regression_summary']['ap50_95_regressed_vs_fixed_gt_0_01_count']}`",
            f"- Regressed vs clean count (>0.01): `{report['non_active_regression_summary']['ap50_95_regressed_vs_clean_gt_0_01_count']}`",
            "",
            "## Conclusion",
            "",
            seed_conclusion(report),
            "",
        ]
    )
    return "\n".join(lines)


def render_summary_markdown(summary: dict[str, Any]) -> str:
    s = summary["summary"]
    lines = [
        "# Image-Only Weak Augmentation Multiseed Summary",
        "",
        "## Scope",
        "",
        "- Method: image-only weak CP-CATF / weak ROI texture attenuation",
        "- Attenuation ratio: `0.25`",
        "- Seeds: `0, 1, 2`",
        "- Seed0/seed1 were run in `catf_v2_image_only_weak_aug_multiseed`.",
        "- Seed2 reuses the completed `catf_v2_image_only_weak_aug_seed2_50ep` result.",
        "- Sampler-only and weighted index list are not part of this mainline.",
        "",
        "## Per-Seed Metrics",
        "",
        "| seed | method | P | R | mAP50 | mAP50-95 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        seed = row["seed"]
        lines.append(metric_row(f"{seed} clean", row["clean"]))
        lines.append(metric_row(f"{seed} fixed CATF-v2", row["fixed_catf_v2"]))
        lines.append(metric_row(f"{seed} weak image aug", row["weak_image_aug"]))
    lines.extend(
        [
            "",
            "## Per-Seed Deltas",
            "",
            "| seed | comparison | dP | dR | d mAP50 | d mAP50-95 | constraint_failed | recall_warning |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary["rows"]:
        seed = row["seed"]
        lines.append(delta_row(seed, "weak - clean", row["delta_vs_clean"], row["constraint_failed"], row["recall_warning"]))
        lines.append(delta_row(seed, "weak - fixed", row["delta_vs_fixed_catf_v2"], row["constraint_failed"], row["recall_warning"]))
    lines.extend(
        [
            "",
            "## Means",
            "",
            "| item | P | R | mAP50 | mAP50-95 |",
            "| --- | ---: | ---: | ---: | ---: |",
            metric_row("mean clean", s["mean_clean"]),
            metric_row("mean fixed CATF-v2", s["mean_fixed_catf_v2"]),
            metric_row("mean weak image aug", s["mean_weak_image_aug"]),
            metric_row("mean weak - clean", s["mean_delta_vs_clean"]),
            metric_row("mean weak - fixed", s["mean_delta_vs_fixed_catf_v2"]),
            "",
            "## Coverage",
            "",
            f"- Constraint pass count: `{s['constraint_pass_count']}/{s['seed_count']}`",
            f"- 3/3 pass: `{str(s['three_seed_pass']).lower()}`",
            f"- Industrial images augmented total: `{s['total_industrial_image_augmented']}`",
            f"- ROI applied total: `{s['total_roi_aug_applied']}`",
            f"- Sampler-only involved: `{str(s['sampler_only_involved']).lower()}`",
            f"- Weighted index list involved: `{str(s['weighted_index_list_involved']).lower()}`",
            f"- Sampled distribution changed any seed: `{str(s['sampled_distribution_changed_any']).lower()}`",
            f"- Final val used for policy selection any seed: `{str(s['final_val_used_for_policy_selection_any']).lower()}`",
            f"- Recall warning any seed: `{str(s['recall_warning_any']).lower()}`",
            f"- Main method candidate: `{str(s['main_method_candidate']).lower()}`",
            f"- Recommend recall-aware constraint: `{str(s['recommend_recall_aware_constraint']).lower()}`",
            "",
            "## Interpretation",
            "",
            "- Seed2 was repaired by image-only weak augmentation without sampler-only or sampling changes.",
            "- Seed1 remains constraint-pass and keeps positive mAP gains vs clean, but does not fully retain fixed CATF-v2 mAP50-95.",
            "- Seed0 fails the current hard constraints because precision and mAP50 drop below clean by more than 0.01.",
            "- Therefore this exact weak augmentation setting is not yet a 3-seed paper main result.",
            "- Seed2 recall remains below clean and should be reported as a recall warning; a recall-aware constraint is the next image-only safety direction.",
            "",
        ]
    )
    return "\n".join(lines)


def seed_conclusion(report: dict[str, Any]) -> str:
    summary = report["summary"]
    if summary["constraint_failed"]:
        return (
            "This seed does not pass the current hard constraints. The weak image augmentation path executed, "
            "but this result should not be treated as retaining the fixed CATF-v2 benefit for this seed."
        )
    if summary["fixed_catf_v2_benefit_retained"]:
        return "This seed passes constraints and retains the fixed CATF-v2 mAP50-95 gain within the configured tolerance."
    return (
        "This seed passes constraints and improves over clean, but it does not fully retain the fixed CATF-v2 "
        "mAP50-95 gain within a 0.01 tolerance."
    )


def metric_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {float(metrics.get('precision', 0.0)):.6f} | "
        f"{float(metrics.get('recall', 0.0)):.6f} | "
        f"{float(metrics.get('map50', 0.0)):.6f} | "
        f"{float(metrics.get('map50_95', 0.0)):.6f} |"
    )


def delta_row(
    seed: int,
    comparison: str,
    metrics: dict[str, Any],
    constraint_failed: bool,
    recall_warning: bool,
) -> str:
    return (
        f"| {seed} | {comparison} | {float(metrics.get('precision', 0.0)):+.6f} | "
        f"{float(metrics.get('recall', 0.0)):+.6f} | "
        f"{float(metrics.get('map50', 0.0)):+.6f} | "
        f"{float(metrics.get('map50_95', 0.0)):+.6f} | "
        f"{str(constraint_failed).lower()} | {str(recall_warning).lower()} |"
    )


def read_events(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    return list(payload.get("events", []))


def read_results_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
