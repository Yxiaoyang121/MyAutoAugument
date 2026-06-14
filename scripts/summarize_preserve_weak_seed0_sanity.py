from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity"
REPORT_DIR = RUN_ROOT / "reports"
FINAL_METRICS = REPORT_DIR / "final_metrics.json"
OUT_JSON = REPORT_DIR / "seed0_preserve_weak_sanity_report.json"
OUT_MD = REPORT_DIR / "seed0_preserve_weak_sanity_report.md"

CLEAN = {"precision": 0.7846, "recall": 0.6765, "map50": 0.7347, "map50_95": 0.4759}
FIXED = {"precision": 0.7785, "recall": 0.6697, "map50": 0.7437, "map50_95": 0.4895}
WEAK_GLOBAL = {"precision": 0.679043, "recall": 0.711821, "map50": 0.722170, "map50_95": 0.476005}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(payload: dict[str, Any]) -> dict[str, float]:
    src = payload["val"]["metrics"]
    return {key: float(src[key]) for key in ("precision", "recall", "map50", "map50_95")}


def delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: a[key] - b[key] for key in a}


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def active_classes_from_policy_history(payload: dict[str, Any]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    raw_history = payload.get("policy_history") or []
    if isinstance(raw_history, dict):
        history = raw_history.get("history") or []
    else:
        history = raw_history
    if not history:
        policy_path = REPORT_DIR / "policy_history.json"
        if policy_path.exists():
            history = read_json(policy_path).get("history", [])
    for event in history:
        classes = []
        for cid, class_policy in (event.get("accepted_policy", {}).get("classes") or {}).items():
            active = False
            for op in (class_policy.get("ops") or {}).values():
                if float(op.get("prob") or 0.0) > 0.0 or float(op.get("strength") or 0.0) > 0.0:
                    active = True
                    break
            if active:
                classes.append(int(cid))
        out[int(event.get("epoch", 0))] = sorted(classes)
    return out


def build_summary() -> dict[str, Any]:
    payload = read_json(FINAL_METRICS)
    run_metrics = metrics(payload)
    events = payload.get("causal_probe_events", [])
    event_actions = Counter(event.get("final_replay_action") or event.get("selected_candidate_action") for event in events)
    risk_levels = Counter(event.get("risk_level") for event in events)
    online = payload.get("online_aug_stats", {})
    roi = payload.get("roi_aug_stats", {})
    policy_active_by_epoch = active_classes_from_policy_history(payload)
    executable_classes = sorted({cid for ids in policy_active_by_epoch.values() for cid in ids})
    roi_classes = sorted(int(cid) for cid in (roi.get("affected_classes") or {}).keys())
    event_preserved_classes = sorted({cid for event in events for cid in (event.get("preserved_active_classes") or [])})
    replay_requested_text = [str(event.get("preserved_original_fixed_op_list") or "") for event in events]
    replay_requested_classes = sorted(
        {
            cid
            for text in replay_requested_text
            for cid in (4, 11, 12)
            if f"c{cid}:" in text
        }
    )
    delta_clean = delta(run_metrics, CLEAN)
    delta_fixed = delta(run_metrics, FIXED)
    delta_weak = delta(run_metrics, WEAK_GLOBAL)
    constraint_failed = (
        delta_clean["precision"] < -0.01
        or delta_clean["map50"] < -0.01
        or delta_clean["map50_95"] < -0.01
    )
    failure_reasons = []
    if delta_clean["precision"] < -0.01:
        failure_reasons.append("precision_drop_gt_0.01")
    if delta_clean["map50"] < -0.01:
        failure_reasons.append("map50_drop_gt_0.01")
    if delta_clean["map50_95"] < -0.01:
        failure_reasons.append("map50_95_drop_gt_0.01")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(RUN_ROOT.relative_to(ROOT)),
        "completed_50ep": bool(payload.get("summary", {}).get("train_success") and payload.get("summary", {}).get("val_success")),
        "preserve_original_count": int(event_actions.get("preserve_original", 0)),
        "weak_roi_texture_count": int(event_actions.get("weak_roi_texture", 0)),
        "strict_noop_count": int(event_actions.get("strict_noop", 0)),
        "risk_levels": dict(risk_levels),
        "replay_requested_fixed_classes": replay_requested_classes,
        "event_preserved_active_classes": event_preserved_classes,
        "executable_policy_classes": executable_classes,
        "roi_affected_classes": roi_classes,
        "fixed_class_4_11_12_retained": set([4, 11, 12]).issubset(set(executable_classes)) or set([4, 11, 12]).issubset(set(roi_classes)),
        "weak_class9_replacement_avoided": int(event_actions.get("weak_roi_texture", 0)) == 0 and 9 not in roi_classes,
        "image_augmentation_executed": int(online.get("samples_augmented", 0) or 0) > 0,
        "industrial_images_augmented": int(online.get("samples_augmented", 0) or 0),
        "roi_applied": int(roi.get("roi_aug_applied", 0) or 0),
        "roi_affected_class_counts": roi.get("affected_classes", {}),
        "router_random_draw_count": int(online.get("router_random_draw_count", 0) or 0),
        "sampler_only_enabled": bool(payload.get("summary", {}).get("sampler_only_enabled", False)),
        "sampler_only_effective": bool(online.get("sampler_only_effective", False)),
        "weighted_index_list_enabled": bool(online.get("weighted_index_list_enabled", False)),
        "sampled_distribution_changed": bool(online.get("sampled_distribution_changed", False)),
        "metrics": run_metrics,
        "baseline_clean": CLEAN,
        "baseline_fixed_catf_v2": FIXED,
        "baseline_weak_global": WEAK_GLOBAL,
        "delta_vs_clean": delta_clean,
        "delta_vs_fixed_catf_v2": delta_fixed,
        "delta_vs_weak_global": delta_weak,
        "constraint_failed": constraint_failed,
        "failure_reasons": failure_reasons,
        "interpretation": {
            "sanity_passed": not constraint_failed and set([4, 11, 12]).issubset(set(executable_classes)),
            "main_issue": "preserve_original events were selected, but the executable policy did not reproduce fixed seed0 class4/11/12; only class11 received ROI augmentation.",
            "recommend_seed2_next": False,
            "recommend_seed1_final_sanity": False,
            "next_step": "Fix preserve_original execution to install/replay the fixed original class-op policy exactly, then rerun seed0 sanity before seed2.",
        },
    }


def write_markdown(summary: dict[str, Any]) -> None:
    m = summary["metrics"]
    lines = [
        "# Seed0 Preserve-Weak Image CATF Sanity Report",
        "",
        "Scope: seed0-only 50ep sanity. No seed1/seed2, no multiseed, no sampler_only, no weighted index list.",
        "",
        "## Decision Counts",
        "",
        f"- completed_50ep: `{summary['completed_50ep']}`",
        f"- preserve_original: `{summary['preserve_original_count']}`",
        f"- weak_roi_texture: `{summary['weak_roi_texture_count']}`",
        f"- strict_noop: `{summary['strict_noop_count']}`",
        f"- risk_levels: `{summary['risk_levels']}`",
        "",
        "## Execution",
        "",
        f"- replay requested fixed classes: `{summary['replay_requested_fixed_classes']}`",
        f"- executable policy classes observed: `{summary['executable_policy_classes']}`",
        f"- ROI affected classes: `{summary['roi_affected_class_counts']}`",
        f"- fixed class4/11/12 retained: `{summary['fixed_class_4_11_12_retained']}`",
        f"- weak class9 replacement avoided: `{summary['weak_class9_replacement_avoided']}`",
        f"- image augmentation executed: `{summary['image_augmentation_executed']}`",
        f"- industrial images augmented: `{summary['industrial_images_augmented']}`",
        f"- ROI applied: `{summary['roi_applied']}`",
        f"- router random draw count: `{summary['router_random_draw_count']}`",
        f"- sampler_only_enabled: `{summary['sampler_only_enabled']}`",
        f"- sampler_only_effective: `{summary['sampler_only_effective']}`",
        f"- weighted_index_list_enabled: `{summary['weighted_index_list_enabled']}`",
        f"- sampled_distribution_changed: `{summary['sampled_distribution_changed']}`",
        "",
        "## Metrics",
        "",
        "| run | P | R | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| preserve-weak seed0 | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['map50'])} | {fmt(m['map50_95'])} |",
        f"| clean seed0 baseline | {fmt(CLEAN['precision'])} | {fmt(CLEAN['recall'])} | {fmt(CLEAN['map50'])} | {fmt(CLEAN['map50_95'])} |",
        f"| fixed CATF-v2 seed0 | {fmt(FIXED['precision'])} | {fmt(FIXED['recall'])} | {fmt(FIXED['map50'])} | {fmt(FIXED['map50_95'])} |",
        f"| weak global seed0 | {fmt(WEAK_GLOBAL['precision'])} | {fmt(WEAK_GLOBAL['recall'])} | {fmt(WEAK_GLOBAL['map50'])} | {fmt(WEAK_GLOBAL['map50_95'])} |",
        "",
        "Deltas:",
        f"- vs clean: `{summary['delta_vs_clean']}`",
        f"- vs fixed CATF-v2: `{summary['delta_vs_fixed_catf_v2']}`",
        f"- vs weak global: `{summary['delta_vs_weak_global']}`",
        "",
        f"constraint_failed: `{summary['constraint_failed']}`; reasons: `{summary['failure_reasons']}`",
        "",
        "## Interpretation",
        "",
        "- The offline decision selection worked at the event level: all 9 feedback epochs selected `preserve_original`; weak and strict no-op were both 0.",
        "- The run avoided the weak class9 replacement: class9 had no weak execution and no ROI applications.",
        "- The run did not preserve the fixed seed0 class4/11/12 executable policy. The actual executable policy and ROI stats show only class11 augmentation.",
        "- Therefore this sanity run failed. It does not validate the three-stage strategy yet.",
        "- Do not proceed to seed2 until preserve_original execution installs or replays the fixed original class-op policy exactly and seed0 passes.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary)
    print(json.dumps({
        "completed_50ep": summary["completed_50ep"],
        "counts": {
            "preserve": summary["preserve_original_count"],
            "weak": summary["weak_roi_texture_count"],
            "noop": summary["strict_noop_count"],
        },
        "metrics": summary["metrics"],
        "constraint_failed": summary["constraint_failed"],
        "fixed_class_4_11_12_retained": summary["fixed_class_4_11_12_retained"],
        "weak_class9_replacement_avoided": summary["weak_class9_replacement_avoided"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
