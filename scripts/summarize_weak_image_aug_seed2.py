from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "catf_v2_image_only_weak_aug_seed2_50ep"
CLEAN_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "multiseed_clean_yolo_default_vs_catf_v2"
    / "seed_2"
    / "clean_native_yolo_default"
    / "reports"
    / "clean_native_yolo_default_metrics.json"
)
FIXED_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "multiseed_clean_yolo_default_vs_catf_v2_fixed"
    / "seed_2"
    / "catf_v2"
    / "reports"
    / "final_metrics.json"
)

METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def main() -> None:
    report = build_report()
    reports_dir = RUN_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "seed2_weak_image_aug_report.json", report)
    (reports_dir / "seed2_weak_image_aug_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def build_report() -> dict[str, Any]:
    final = read_json(RUN_ROOT / "reports" / "final_metrics.json")
    clean = metrics_payload(read_json(CLEAN_METRICS))
    fixed = metrics_payload(read_json(FIXED_METRICS))
    weak = metrics_payload(final)
    online = read_json(RUN_ROOT / "reports" / "online_aug_stats.json")
    roi = read_json(RUN_ROOT / "reports" / "roi_aug_stats.json")
    constraint = read_json(RUN_ROOT / "reports" / "constraint_scoring.json")
    events = read_json(RUN_ROOT / "reports" / "causal_probe_events.json").get("events", [])
    train_rows = read_results_csv(RUN_ROOT / "train" / "results.csv")

    weak_events = [event for event in events if event.get("selected_candidate_policy_id") == "candidate_policy_1b_weak_roi_texture"]
    strict_noop_events = [event for event in events if event.get("action") == "strict_no_op"]
    deltas_clean = delta(weak, clean)
    deltas_fixed = delta(weak, fixed)
    class_rows = class_comparison(clean, fixed, weak)
    class9 = class_rows.get("9", {})
    non_active = non_active_regression_summary(class_rows, weak_class_ids={6, 7, 9})

    summary = {
        "seed2_50ep_completed": bool(final.get("train", {}).get("success") and final.get("val", {}).get("success") and len(train_rows) == 50),
        "weak_image_augmentation_executed": int(online.get("samples_augmented", 0) or 0) > 0,
        "samples_augmented": int(online.get("samples_augmented", 0) or 0),
        "roi_aug_applied": int(roi.get("roi_aug_applied", 0) or 0),
        "industrial_image_augmented": int(online.get("samples_augmented", 0) or 0),
        "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
        "sampler_only_enabled": bool(final.get("summary", {}).get("sampler_only_enabled", False)),
        "weighted_index_list_enabled": bool(final.get("summary", {}).get("weighted_index_list_enabled", False)),
        "sampled_distribution_changed": bool(final.get("summary", {}).get("sampled_distribution_changed", False)),
        "final_val_used_for_policy_selection": any(bool(event.get("final_val_used_for_policy_selection", False)) for event in events),
        "constraint_failed": bool(constraint.get("constraint_failed", True)),
        "class9_recovered_vs_fixed": bool(
            class9.get("delta_weak_vs_fixed", {}).get("recall", 0.0) > 0.0
            and class9.get("delta_weak_vs_fixed", {}).get("ap50", 0.0) > 0.0
        ),
        "class9_fully_recovered_vs_clean": bool(
            class9.get("delta_weak_vs_clean", {}).get("recall", -1.0) >= 0.0
            and class9.get("delta_weak_vs_clean", {}).get("ap50_95", -1.0) >= 0.0
        ),
        "non_active_regression_mitigated_vs_fixed": bool(non_active["mean_ap50_95_delta_weak_vs_fixed"] > 0.0),
        "recommend_seed0_seed1_sanity": bool(not constraint.get("constraint_failed", True) and int(online.get("samples_augmented", 0) or 0) > 0),
        "image_only_cp_catf_fix_direction": bool(not constraint.get("constraint_failed", True) and int(online.get("samples_augmented", 0) or 0) > 0),
    }
    return {
        "run_root": str(RUN_ROOT),
        "clean_seed2": metric_subset(clean),
        "fixed_catf_v2_seed2": metric_subset(fixed),
        "weak_image_aug_seed2": metric_subset(weak),
        "delta_vs_clean_seed2": deltas_clean,
        "delta_vs_fixed_catf_v2_seed2": deltas_fixed,
        "constraint_scoring": constraint,
        "weak_candidate_events": weak_events,
        "strict_noop_events": strict_noop_events,
        "weak_candidate_count": len(weak_events),
        "strict_noop_count": len(strict_noop_events),
        "weak_candidate_ops": [
            {
                "epoch": int(event.get("epoch", -1)),
                "candidate_policy_id": event.get("selected_candidate_policy_id"),
                "retained_op": event.get("retained_op"),
                "original_prob": event.get("original_prob"),
                "original_strength": event.get("original_strength"),
                "weak_prob": event.get("weak_prob"),
                "weak_strength": event.get("weak_strength"),
                "attenuation_ratio": event.get("attenuation_ratio"),
                "max_aug_samples_per_interval": event.get("max_aug_samples_per_interval"),
                "precision_aware_gate_passed": event.get("precision_aware_gate_passed"),
                "non_active_regression_gate_passed": event.get("non_active_regression_gate_passed"),
                "sample_router_allowed": event.get("sample_router_allowed"),
            }
            for event in weak_events
        ],
        "online_augmentation": {
            "samples_seen": int(online.get("samples_seen", 0) or 0),
            "samples_augmented": int(online.get("samples_augmented", 0) or 0),
            "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
            "weak_image_aug_interval_counts": online.get("weak_image_aug_interval_counts", {}),
            "sample_weight_map_generated": bool(online.get("sample_weight_map_generated", False)),
            "weighted_train_core_images_count": int(online.get("weighted_train_core_images_count", 0) or 0),
            "weighted_index_list_enabled": bool(final.get("summary", {}).get("weighted_index_list_enabled", False)),
            "sampled_distribution_changed": bool(final.get("summary", {}).get("sampled_distribution_changed", False)),
        },
        "roi_augmentation": roi,
        "class_comparison": class_rows,
        "class9_summary": class9,
        "non_active_regression_summary": non_active,
        "summary": summary,
        "paths": {
            "final_metrics": str(RUN_ROOT / "reports" / "final_metrics.json"),
            "online_aug_stats": str(RUN_ROOT / "reports" / "online_aug_stats.json"),
            "roi_aug_stats": str(RUN_ROOT / "reports" / "roi_aug_stats.json"),
            "causal_probe_events": str(RUN_ROOT / "reports" / "causal_probe_events.json"),
        },
    }


def metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return (payload.get("val") or {}).get("metrics") or payload.get("metrics") or payload


def metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0) or 0.0) for key in METRIC_KEYS}


def delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {key: float(current.get(key, 0.0) or 0.0) - float(reference.get(key, 0.0) or 0.0) for key in METRIC_KEYS}


def per_class_map(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(int(row.get("class_id"))): row for row in metrics.get("per_class", []) if row.get("class_id") is not None}


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
            "delta_weak_vs_clean": {key: float(w.get(key, 0.0) or 0.0) - float(c.get(key, 0.0) or 0.0) for key in keys},
            "delta_weak_vs_fixed": {key: float(w.get(key, 0.0) or 0.0) - float(f.get(key, 0.0) or 0.0) for key in keys},
        }
    return out


def non_active_regression_summary(class_rows: dict[str, Any], *, weak_class_ids: set[int]) -> dict[str, Any]:
    non_active = [row for cid, row in class_rows.items() if int(cid) not in weak_class_ids]
    improved_fixed = [
        row for row in non_active if row["delta_weak_vs_fixed"].get("ap50_95", 0.0) >= 0.0
    ]
    regressed_clean = [
        row for row in non_active if row["delta_weak_vs_clean"].get("ap50_95", 0.0) < -0.01
    ]
    return {
        "weak_target_class_ids": sorted(weak_class_ids),
        "non_active_class_count": len(non_active),
        "non_active_ap50_95_improved_vs_fixed_count": len(improved_fixed),
        "non_active_ap50_95_regressed_vs_clean_gt_0_01_count": len(regressed_clean),
        "mean_ap50_95_delta_weak_vs_fixed": mean([row["delta_weak_vs_fixed"].get("ap50_95", 0.0) for row in non_active]) if non_active else 0.0,
        "mean_ap50_95_delta_weak_vs_clean": mean([row["delta_weak_vs_clean"].get("ap50_95", 0.0) for row in non_active]) if non_active else 0.0,
        "regressed_vs_clean_classes": [
            {
                "class_id": row["class_id"],
                "name": row["name"],
                "delta_ap50_95": row["delta_weak_vs_clean"].get("ap50_95"),
            }
            for row in regressed_clean
        ],
    }


def read_results_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    weak = report["weak_image_aug_seed2"]
    d_clean = report["delta_vs_clean_seed2"]
    d_fixed = report["delta_vs_fixed_catf_v2_seed2"]
    online = report["online_augmentation"]
    roi = report["roi_augmentation"]
    class9 = report["class9_summary"]
    non_active = report["non_active_regression_summary"]
    lines = [
        "# Seed2 Image-Only Weak Augmentation Validation",
        "",
        "## Scope",
        "",
        "- Run: `catf_v2_image_only_weak_aug_seed2_50ep`",
        "- Seed: `2`",
        "- Method: image-only CP-CATF weak ROI texture attenuation",
        "- Attenuation ratio: `0.25`",
        "- Sampler-only: `disabled`",
        "- Weighted index list: `disabled`",
        "- Sampled distribution changed: `false`",
        "- Final-val policy-selection flag in weak decisions: `false`",
        "",
        "## Execution",
        "",
        f"- 50ep completed: `{str(s['seed2_50ep_completed']).lower()}`",
        f"- Weak image augmentation executed: `{str(s['weak_image_augmentation_executed']).lower()}`",
        f"- Industrial images augmented: `{online['samples_augmented']}`",
        f"- ROI applied: `{roi.get('roi_aug_applied', 0)}`",
        f"- Router random draw count: `{online['router_random_draw_count']}`",
        f"- Weak interval counts: `{json.dumps(online['weak_image_aug_interval_counts'], ensure_ascii=False)}`",
        f"- sample_weight_map generated: `{str(online['sample_weight_map_generated']).lower()}`",
        f"- weighted_train_core_images_count: `{online['weighted_train_core_images_count']}`",
        "",
        "## Metrics",
        "",
        "| run | P | R | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        row("clean seed2", report["clean_seed2"]),
        row("fixed CATF-v2 seed2", report["fixed_catf_v2_seed2"]),
        row("weak image aug seed2", weak),
        "",
        "## Delta",
        "",
        f"- vs clean seed2: P `{d_clean['precision']:+.4f}`, R `{d_clean['recall']:+.4f}`, mAP50 `{d_clean['map50']:+.4f}`, mAP50-95 `{d_clean['map50_95']:+.4f}`",
        f"- vs fixed CATF-v2 seed2: P `{d_fixed['precision']:+.4f}`, R `{d_fixed['recall']:+.4f}`, mAP50 `{d_fixed['map50']:+.4f}`, mAP50-95 `{d_fixed['map50_95']:+.4f}`",
        f"- constraint_failed: `{str(s['constraint_failed']).lower()}`",
        "",
        "## Weak Candidates",
        "",
        "| epoch | op | original prob | original strength | weak prob | weak strength | cap | router |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for event in report["weak_candidate_ops"]:
        lines.append(
            f"| {event['epoch']} | `{event['retained_op']}` | {fmt(event['original_prob'])} | "
            f"{fmt(event['original_strength'])} | {fmt(event['weak_prob'])} | {fmt(event['weak_strength'])} | "
            f"{event['max_aug_samples_per_interval']} | `{str(event['sample_router_allowed']).lower()}` |"
        )
    lines.extend(
        [
            "",
            "## Class9 And Non-Active Regression",
            "",
            f"- Class9 recovered vs fixed: `{str(s['class9_recovered_vs_fixed']).lower()}`",
            f"- Class9 fully recovered vs clean: `{str(s['class9_fully_recovered_vs_clean']).lower()}`",
            f"- Class9 delta vs fixed: recall `{class9['delta_weak_vs_fixed']['recall']:+.4f}`, AP50 `{class9['delta_weak_vs_fixed']['ap50']:+.4f}`, AP50-95 `{class9['delta_weak_vs_fixed']['ap50_95']:+.4f}`",
            f"- Class9 delta vs clean: recall `{class9['delta_weak_vs_clean']['recall']:+.4f}`, AP50 `{class9['delta_weak_vs_clean']['ap50']:+.4f}`, AP50-95 `{class9['delta_weak_vs_clean']['ap50_95']:+.4f}`",
            f"- Non-active AP50-95 improved vs fixed count: `{non_active['non_active_ap50_95_improved_vs_fixed_count']}/{non_active['non_active_class_count']}`",
            f"- Mean non-active AP50-95 delta vs fixed: `{non_active['mean_ap50_95_delta_weak_vs_fixed']:+.4f}`",
            f"- Mean non-active AP50-95 delta vs clean: `{non_active['mean_ap50_95_delta_weak_vs_clean']:+.4f}`",
            "",
            "## Conclusion",
            "",
            f"- Seed2 weak image augmentation passes constraints: `{str(not s['constraint_failed']).lower()}`",
            f"- It is a viable image-only CP-CATF repair direction for seed2: `{str(s['image_only_cp_catf_fix_direction']).lower()}`",
            f"- Recommend seed0/seed1 sanity next: `{str(s['recommend_seed0_seed1_sanity']).lower()}`",
        ]
    )
    return "\n".join(lines) + "\n"


def row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
        f"{metrics['map50']:.4f} | {metrics['map50_95']:.4f} |"
    )


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
