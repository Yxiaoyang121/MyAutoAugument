from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity_rerun"
REPORTS = RUN_ROOT / "reports"
OUT_JSON = REPORTS / "seed0_preserve_weak_sanity_rerun_report.json"
OUT_MD = REPORTS / "seed0_preserve_weak_sanity_rerun_report.md"

CLEAN = {"precision": 0.7846, "recall": 0.6765, "map50": 0.7347, "map50_95": 0.4759}
FIXED = {"precision": 0.7785, "recall": 0.6697, "map50": 0.7437, "map50_95": 0.4895}
PRE_FIX = {
    "precision": 0.6665845856514329,
    "recall": 0.7300194435303583,
    "map50": 0.6999344792396978,
    "map50_95": 0.4667435067714014,
}
FIXED_AUG = {"industrial": 41, "roi": 45, "roi_by_class": {"4": 9, "11": 22, "12": 14}}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) - float(baseline[key]) for key in ("precision", "recall", "map50", "map50_95")}


def r4(value: float) -> float:
    return round(float(value), 6)


def main() -> None:
    final_metrics = read_json(REPORTS / "final_metrics.json")
    online = read_json(REPORTS / "online_aug_stats.json")
    roi = read_json(REPORTS / "roi_aug_stats.json")
    events = read_json(REPORTS / "causal_probe_events.json").get("events", [])
    val = final_metrics["val"]["metrics"]
    metrics = {
        "precision": float(val["precision"]),
        "recall": float(val["recall"]),
        "map50": float(val["map50"]),
        "map50_95": float(val["map50_95"]),
    }

    action_counts = Counter(str(event.get("action")) for event in events)
    expected_union = sorted({int(cid) for event in events for cid in (event.get("preserve_expected_active_classes") or [])})
    installed_union = sorted({int(cid) for event in events for cid in (event.get("preserve_installed_active_classes") or [])})
    executable_union = sorted({int(item["class_id"]) for event in events for item in (event.get("preserve_installed_ops") or [])})
    weak_class9_replacement = any(9 in (event.get("preserve_installed_active_classes") or []) for event in events)
    sampler_only_involved = any(bool(event.get("sample_weighting_allowed", False)) for event in events)
    weighted_index_list_enabled = bool(online.get("weighted_index_list_enabled", False))
    sampled_distribution_changed = bool(online.get("sampled_distribution_changed", False))

    deltas = {
        "vs_clean": delta(metrics, CLEAN),
        "vs_fixed": delta(metrics, FIXED),
        "vs_pre_fix_preserve": delta(metrics, PRE_FIX),
    }
    constraint_failed = (
        deltas["vs_clean"]["precision"] < -0.01
        or deltas["vs_clean"]["map50"] < -0.01
        or deltas["vs_clean"]["map50_95"] < -0.01
    )
    failure_reasons = []
    if deltas["vs_clean"]["precision"] < -0.01:
        failure_reasons.append("precision_drop_gt_0.01")
    if deltas["vs_clean"]["map50"] < -0.01:
        failure_reasons.append("map50_drop_gt_0.01")
    if deltas["vs_clean"]["map50_95"] < -0.01:
        failure_reasons.append("map50_95_drop_gt_0.01")

    roi_count = int(roi.get("roi_aug_applied", 0) or 0)
    industrial_count = int(online.get("samples_augmented", 0) or 0)
    preserve_class_union_ok = {4, 11, 12}.issubset(set(executable_union))
    fixed_strategy_retained = preserve_class_union_ok and not weak_class9_replacement
    fixed_application_volume_close = abs(industrial_count - FIXED_AUG["industrial"]) <= 15 and abs(roi_count - FIXED_AUG["roi"]) <= 20
    if constraint_failed and preserve_class_union_ok and not fixed_application_volume_close:
        failure_interpretation = (
            "class_union_execution_parity_fixed_but_fixed_policy_lifetime_or_application_volume_not_reproduced"
        )
    elif constraint_failed:
        failure_interpretation = "method_or_remaining_execution_mismatch_requires_audit"
    else:
        failure_interpretation = "seed0_passed_after_preserve_execution_fix"

    summary = {
        "run_root": str(RUN_ROOT),
        "completed_50ep": bool(final_metrics.get("train", {}).get("success", False) and final_metrics.get("val", {}).get("success", False)),
        "preserve_original_count": int(action_counts.get("preserve_original", 0)),
        "weak_roi_texture_count": int(action_counts.get("accept_offline_probe_candidate", 0)),
        "strict_noop_count": int(action_counts.get("strict_no_op", 0)),
        "event_action_counts": dict(action_counts),
        "expected_class_union": expected_union,
        "runtime_policy_matrix_union": installed_union,
        "sample_router_eligible_union": executable_union,
        "final_executable_union": executable_union,
        "contains_class4_11_12": preserve_class_union_ok,
        "fixed_original_strategy_retained_at_class_op_level": fixed_strategy_retained,
        "fixed_application_volume_close": fixed_application_volume_close,
        "weak_class9_replacement": weak_class9_replacement,
        "image_augmentation_executed": industrial_count > 0 and roi_count > 0,
        "industrial_images_augmented": industrial_count,
        "roi_applied": roi_count,
        "roi_affected_classes": roi.get("affected_classes", {}),
        "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
        "fixed_catf_v2_seed0_aug_reference": FIXED_AUG,
        "sampler_only_enabled": bool(online.get("sampler_only_enabled", False)),
        "sampler_only_effective": bool(online.get("sampler_only_effective", False)),
        "weighted_index_list_enabled": weighted_index_list_enabled,
        "sampled_distribution_changed": sampled_distribution_changed,
        "metrics": {key: r4(value) for key, value in metrics.items()},
        "clean_seed0": CLEAN,
        "fixed_catf_v2_seed0": FIXED,
        "pre_fix_preserve_weak_seed0": {key: r4(value) for key, value in PRE_FIX.items()},
        "delta_vs_clean": {key: r4(value) for key, value in deltas["vs_clean"].items()},
        "delta_vs_fixed": {key: r4(value) for key, value in deltas["vs_fixed"].items()},
        "delta_vs_pre_fix_preserve": {key: r4(value) for key, value in deltas["vs_pre_fix_preserve"].items()},
        "constraint_failed": bool(constraint_failed),
        "failure_reasons": failure_reasons,
        "failure_interpretation": failure_interpretation,
        "recommend_continue_seed2": bool(not constraint_failed),
        "recommendation": (
            "Do not run seed2 yet; audit preserve policy lifetime/application-volume parity against fixed CATF-v2."
            if constraint_failed
            else "Seed0 passed; seed2 can be considered next under the requested protocol."
        ),
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Seed0 Preserve-Weak Sanity Rerun",
        "",
        "## Run Status",
        "",
        f"- Completed 50ep: `{str(summary['completed_50ep']).lower()}`",
        f"- preserve_original / weak / noop: `{summary['preserve_original_count']}/{summary['weak_roi_texture_count']}/{summary['strict_noop_count']}`",
        f"- Expected class union: `{summary['expected_class_union']}`",
        f"- Runtime policy_matrix union: `{summary['runtime_policy_matrix_union']}`",
        f"- Sample router eligible union: `{summary['sample_router_eligible_union']}`",
        f"- Final executable union: `{summary['final_executable_union']}`",
        f"- Contains class4/11/12: `{str(summary['contains_class4_11_12']).lower()}`",
        f"- Weak class9 replacement: `{str(summary['weak_class9_replacement']).lower()}`",
        "",
        "## Augmentation",
        "",
        f"- Image augmentation executed: `{str(summary['image_augmentation_executed']).lower()}`",
        f"- Industrial images augmented: `{summary['industrial_images_augmented']}`",
        f"- ROI applied: `{summary['roi_applied']}`",
        f"- ROI affected classes: `{summary['roi_affected_classes']}`",
        f"- Fixed CATF-v2 seed0 reference: `{summary['fixed_catf_v2_seed0_aug_reference']}`",
        f"- Fixed application volume close: `{str(summary['fixed_application_volume_close']).lower()}`",
        f"- Router random draw count: `{summary['router_random_draw_count']}`",
        "",
        "## Sampling",
        "",
        f"- sampler_only_enabled: `{str(summary['sampler_only_enabled']).lower()}`",
        f"- sampler_only_effective: `{str(summary['sampler_only_effective']).lower()}`",
        f"- weighted_index_list_enabled: `{str(summary['weighted_index_list_enabled']).lower()}`",
        f"- sampled_distribution_changed: `{str(summary['sampled_distribution_changed']).lower()}`",
        "",
        "## Metrics",
        "",
        "| run | P | R | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| clean seed0 | {CLEAN['precision']:.6f} | {CLEAN['recall']:.6f} | {CLEAN['map50']:.6f} | {CLEAN['map50_95']:.6f} |",
        f"| fixed CATF-v2 seed0 | {FIXED['precision']:.6f} | {FIXED['recall']:.6f} | {FIXED['map50']:.6f} | {FIXED['map50_95']:.6f} |",
        f"| pre-fix preserve seed0 | {PRE_FIX['precision']:.6f} | {PRE_FIX['recall']:.6f} | {PRE_FIX['map50']:.6f} | {PRE_FIX['map50_95']:.6f} |",
        f"| rerun preserve seed0 | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['map50']:.6f} | {metrics['map50_95']:.6f} |",
        "",
        "## Delta",
        "",
        "| comparison | dP | dR | dmAP50 | dmAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| vs clean | {deltas['vs_clean']['precision']:+.6f} | {deltas['vs_clean']['recall']:+.6f} | {deltas['vs_clean']['map50']:+.6f} | {deltas['vs_clean']['map50_95']:+.6f} |",
        f"| vs fixed | {deltas['vs_fixed']['precision']:+.6f} | {deltas['vs_fixed']['recall']:+.6f} | {deltas['vs_fixed']['map50']:+.6f} | {deltas['vs_fixed']['map50_95']:+.6f} |",
        f"| vs pre-fix preserve | {deltas['vs_pre_fix_preserve']['precision']:+.6f} | {deltas['vs_pre_fix_preserve']['recall']:+.6f} | {deltas['vs_pre_fix_preserve']['map50']:+.6f} | {deltas['vs_pre_fix_preserve']['map50_95']:+.6f} |",
        "",
        "## Verdict",
        "",
        f"- constraint_failed: `{str(summary['constraint_failed']).lower()}`",
        f"- Failure reasons: `{summary['failure_reasons']}`",
        f"- Interpretation: `{summary['failure_interpretation']}`",
        f"- Recommendation: {summary['recommendation']}",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_MD), "constraint_failed": summary["constraint_failed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
