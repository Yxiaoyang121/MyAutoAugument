from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MULTISEED_ROOT = ROOT / "outputs" / "experiments" / "multiseed_cp_catf_paper_mode_sampler_only"
SEED_RUNS = {
    0: ROOT / "outputs" / "experiments" / "cp_catf_paper_mode_sampler_only_seed0",
    1: MULTISEED_ROOT / "cp_catf_seed_1",
    2: MULTISEED_ROOT / "cp_catf_seed_2",
}
CLEAN_BASELINES = {
    0: {"precision": 0.7513, "recall": 0.6763, "map50": 0.7566, "map50_95": 0.5114},
    1: {"precision": 0.7220, "recall": 0.7582, "map50": 0.7777, "map50_95": 0.5251},
    2: {"precision": 0.6290, "recall": 0.6385, "map50": 0.6590, "map50_95": 0.4381},
}
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
CONSTRAINT_KEYS = ("precision", "map50", "map50_95")
CONSTRAINT_DROP_LIMIT = -0.01


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def signed(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def latest_epoch_artifact(reports_dir: Path, stem: str) -> tuple[int | None, Path | None]:
    candidates: list[tuple[int, Path]] = []
    for path in reports_dir.glob(f"{stem}_epoch_*.json"):
        try:
            epoch = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((epoch, path))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def copy_latest_sampler_artifacts(reports_dir: Path) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for stem, output_name in (
        ("sample_weight_map", "sample_weight_map.json"),
        ("weighted_train_indices", "weighted_train_indices.json"),
        ("sampled_distribution_before_after", "sampled_distribution_before_after.json"),
    ):
        epoch, source = latest_epoch_artifact(reports_dir, stem)
        if source is None:
            copied[output_name] = {"copied": False, "reason": f"missing {stem}_epoch_*.json"}
            continue
        target = reports_dir / output_name
        shutil.copyfile(source, target)
        copied[output_name] = {
            "copied": True,
            "source": str(source),
            "target": str(target),
            "epoch": epoch,
        }
    return copied


def event_list(reports_dir: Path) -> list[dict[str, Any]]:
    payload = read_json(reports_dir / "causal_probe_events.json")
    if isinstance(payload, dict):
        events = payload.get("events", [])
    else:
        events = payload
    return [event for event in events if isinstance(event, dict)]


def final_val_leakage_false(audit: dict[str, Any]) -> bool:
    return (
        bool(audit.get("paper_probe_mode"))
        and not bool(audit.get("final_val_used_for_policy_selection"))
        and int(audit.get("train_core_probe_overlap_count", 0) or 0) == 0
        and int(audit.get("train_core_final_val_overlap_count", 0) or 0) == 0
        and int(audit.get("probe_final_val_overlap_count", 0) or 0) == 0
    )


def protected_weighting_summary(sample_weight_map: dict[str, Any]) -> dict[str, Any]:
    weighted_images = sample_weight_map.get("weighted_images", [])
    bad_rows = [
        row
        for row in weighted_images
        if row.get("high_fp_protected") or row.get("ok_protected") or row.get("no_aug_protected")
    ]
    return {
        "weighted_images_count": len(weighted_images),
        "protected_weighted_count": len(bad_rows),
        "high_fp_or_ok_or_no_aug_wrongly_weighted": len(bad_rows) > 0,
        "examples": bad_rows[:5],
    }


def target_class_summary(sample_weight_map: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_classes = sample_weight_map.get("target_classes", {})
    for raw_key in sorted(target_classes, key=lambda key: int(key) if str(key).isdigit() else str(key)):
        item = target_classes[raw_key]
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "class_id": item.get("class_id"),
                "class_name": item.get("class_name"),
                "dominant_issue": item.get("dominant_issue"),
                "weight": item.get("weight"),
                "reasons": item.get("reasons", []),
            }
        )
    return rows


def feedback_decisions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for event in events:
        decisions.append(
            {
                "epoch": event.get("epoch"),
                "candidate_policy_id": event.get("selected_candidate_policy_id"),
                "candidate_action": event.get("selected_candidate_action"),
                "candidate_class_id": event.get("candidate_class_id"),
                "causal_score": event.get("selected_causal_score"),
                "image_candidate_rejected_by_precision_gate": bool(event.get("probe_reject_image_aug")),
                "image_modification_allowed": bool(event.get("image_modification_allowed")),
                "sampler_only_selected": event.get("selected_candidate_action") == "sampler_only",
                "sampler_only_effective": bool(event.get("sampler_only_effective")),
                "weighted_train_core_images_count": event.get("weighted_train_core_images_count"),
                "weighted_index_list_enabled": bool(event.get("weighted_index_list_enabled")),
                "sampled_distribution_changed": bool(event.get("sampled_distribution_changed")),
                "sample_weighting_status": event.get("sample_weighting_status"),
                "sampler_only_blocker": event.get("sampler_only_blocker"),
            }
        )
    return decisions


def build_seed_report(seed: int, run_dir: Path) -> dict[str, Any]:
    reports_dir = run_dir / "reports"
    copied_artifacts = copy_latest_sampler_artifacts(reports_dir)
    metrics_payload = read_json(reports_dir / "final_metrics.json")
    metrics = metrics_payload.get("val", {}).get("metrics", metrics_payload)
    metrics = {key: float(metrics[key]) for key in METRIC_KEYS}
    baseline = CLEAN_BASELINES[seed]
    deltas = {key: metrics[key] - baseline[key] for key in METRIC_KEYS}
    constraint_reasons = [
        f"{key}_drop_gt_0.01" for key in CONSTRAINT_KEYS if deltas[key] < CONSTRAINT_DROP_LIMIT
    ]
    online_stats = read_json(reports_dir / "online_aug_stats.json")
    roi_stats = read_json(reports_dir / "roi_aug_stats.json")
    leakage_audit = read_json(reports_dir / "paper_probe_leakage_audit.json")
    events = event_list(reports_dir)
    sample_weight_map = read_json(reports_dir / "sample_weight_map.json")
    weighted_indices = read_json(reports_dir / "weighted_train_indices.json")
    distribution = read_json(reports_dir / "sampled_distribution_before_after.json")
    protected_summary = protected_weighting_summary(sample_weight_map)
    target_summary = target_class_summary(sample_weight_map)
    decisions = feedback_decisions(events)

    report = {
        "seed": seed,
        "run_dir": str(run_dir),
        "clean_paper_baseline": baseline,
        "cp_catf_paper_mode_sampler_only": metrics,
        "delta_vs_clean": deltas,
        "constraint_failed": bool(constraint_reasons),
        "constraint_reasons": constraint_reasons,
        "constraint_drop_limit": 0.01,
        "paper_mode": bool(online_stats.get("paper_probe_mode")),
        "causal_probe_mode": bool(online_stats.get("catf_causal_probe_mode")),
        "precision_aware_accept_gate": True,
        "sampler_only_enabled": True,
        "final_val_leakage_false": final_val_leakage_false(leakage_audit),
        "final_val_used_for_policy_selection": bool(leakage_audit.get("final_val_used_for_policy_selection")),
        "sampler_only": {
            "effective": bool(online_stats.get("sampler_only_effective")),
            "status": online_stats.get("sampler_only_status"),
            "effective_feedback_epoch_count": sum(1 for event in events if event.get("sampler_only_effective")),
            "effective_feedback_epochs": [event.get("epoch") for event in events if event.get("sampler_only_effective")],
            "weighted_train_core_images_count": online_stats.get("weighted_train_core_images_count"),
            "weighted_sampler_enabled": bool(online_stats.get("weighted_sampler_enabled")),
            "weighted_index_list_enabled": bool(online_stats.get("weighted_index_list_enabled")),
            "weighted_train_indices_count": weighted_indices.get("weighted_train_indices_count"),
            "sampled_distribution_changed": bool(online_stats.get("sampled_distribution_changed")),
            "sample_weight_map_generated": bool(online_stats.get("sample_weight_map_generated")),
        },
        "augmentation_counts": {
            "image_augmented": int(online_stats.get("samples_augmented") or 0),
            "roi_applied": int(roi_stats.get("roi_aug_applied") or 0),
            "industrial_image_augmented": int(online_stats.get("industrial_image_augmented") or 0),
            "router_random_draw_count": int(online_stats.get("router_random_draw_count") or 0),
        },
        "weighted_classes": target_summary,
        "protected_weighting": protected_summary,
        "sampled_distribution": {
            "generated": bool(distribution.get("generated")),
            "changed": bool(distribution.get("sampled_distribution_changed")),
            "before": distribution.get("before", {}),
            "after": distribution.get("after", {}),
            "delta_class_fraction": distribution.get("delta_class_fraction", {}),
        },
        "feedback_decisions": decisions,
        "copied_latest_artifacts": copied_artifacts,
        "required_artifacts": {
            "final_metrics": str(reports_dir / "final_metrics.json"),
            "sampler_only_report_md": str(reports_dir / "sampler_only_report.md"),
            "sampler_only_report_json": str(reports_dir / "sampler_only_report.json"),
            "sample_weight_map": str(reports_dir / "sample_weight_map.json"),
            "weighted_train_indices": str(reports_dir / "weighted_train_indices.json"),
            "sampled_distribution_before_after": str(reports_dir / "sampled_distribution_before_after.json"),
            "policy_history": str(reports_dir / "policy_history.json"),
            "class_policy_history": str(reports_dir / "class_policy_history.json"),
            "causal_probe_decisions_used": str(reports_dir / "causal_probe_decisions_used.json"),
            "online_aug_stats": str(reports_dir / "online_aug_stats.json"),
            "roi_aug_stats": str(reports_dir / "roi_aug_stats.json"),
        },
    }
    write_json(reports_dir / "sampler_only_report.json", report)
    (reports_dir / "sampler_only_report.md").write_text(render_seed_markdown(report), encoding="utf-8")
    return report


def render_seed_markdown(report: dict[str, Any]) -> str:
    seed = report["seed"]
    metrics = report["cp_catf_paper_mode_sampler_only"]
    baseline = report["clean_paper_baseline"]
    deltas = report["delta_vs_clean"]
    sampler = report["sampler_only"]
    aug = report["augmentation_counts"]
    lines = [
        f"# CP-CATF Paper-Mode Sampler-Only Seed {seed}",
        "",
        "## Result",
        "",
        f"- run_dir: `{report['run_dir']}`",
        f"- paper_mode: `{str(report['paper_mode']).lower()}`",
        f"- final_val_leakage_false: `{str(report['final_val_leakage_false']).lower()}`",
        f"- final_val_used_for_policy_selection: `{str(report['final_val_used_for_policy_selection']).lower()}`",
        f"- constraint_failed: `{str(report['constraint_failed']).lower()}`",
        f"- constraint_reasons: `{', '.join(report['constraint_reasons']) if report['constraint_reasons'] else 'none'}`",
        "",
        "| run | P | R | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        f"| clean paper seed{seed} | {fmt(baseline['precision'])} | {fmt(baseline['recall'])} | {fmt(baseline['map50'])} | {fmt(baseline['map50_95'])} |",
        f"| CP-CATF sampler-only seed{seed} | {fmt(metrics['precision'])} | {fmt(metrics['recall'])} | {fmt(metrics['map50'])} | {fmt(metrics['map50_95'])} |",
        f"| delta | {signed(deltas['precision'])} | {signed(deltas['recall'])} | {signed(deltas['map50'])} | {signed(deltas['map50_95'])} |",
        "",
        "## Sampler-Only Status",
        "",
        f"- sampler_only_effective: `{str(sampler['effective']).lower()}`",
        f"- effective feedback epochs: `{sampler['effective_feedback_epochs']}`",
        f"- weighted_train_core_images_count: `{sampler['weighted_train_core_images_count']}`",
        f"- weighted_sampler_enabled: `{str(sampler['weighted_sampler_enabled']).lower()}`",
        f"- weighted_index_list_enabled: `{str(sampler['weighted_index_list_enabled']).lower()}`",
        f"- sampled_distribution_changed: `{str(sampler['sampled_distribution_changed']).lower()}`",
        "",
        "## Augmentation Counts",
        "",
        f"- image_augmented: `{aug['image_augmented']}`",
        f"- ROI applied: `{aug['roi_applied']}`",
        f"- industrial image augmented: `{aug['industrial_image_augmented']}`",
        f"- router random draw count: `{aug['router_random_draw_count']}`",
        "",
        "## Weighted Classes",
        "",
    ]
    if report["weighted_classes"]:
        lines.extend(["| class_id | class_name | dominant_issue | weight | reasons |", "|---:|---|---|---:|---|"])
        for row in report["weighted_classes"]:
            lines.append(
                f"| {row.get('class_id')} | {row.get('class_name')} | {row.get('dominant_issue')} | "
                f"{fmt(row.get('weight'))} | {', '.join(row.get('reasons', []))} |"
            )
    else:
        lines.append("- none")
    protected = report["protected_weighting"]
    lines.extend(
        [
            "",
            "## Protected-Class Check",
            "",
            f"- high-FP / OK / no_aug wrongly weighted: `{str(protected['high_fp_or_ok_or_no_aug_wrongly_weighted']).lower()}`",
            f"- protected weighted count: `{protected['protected_weighted_count']}`",
            "",
            "## Feedback Decisions",
            "",
            "| epoch | candidate_policy_id | action | class | score | image rejected | sampler effective | weighted images | distribution changed |",
            "|---:|---|---|---:|---:|---|---|---:|---|",
        ]
    )
    for row in report["feedback_decisions"]:
        lines.append(
            f"| {row.get('epoch')} | {row.get('candidate_policy_id')} | {row.get('candidate_action')} | "
            f"{row.get('candidate_class_id')} | {fmt(row.get('causal_score'))} | "
            f"{str(row.get('image_candidate_rejected_by_precision_gate')).lower()} | "
            f"{str(row.get('sampler_only_effective')).lower()} | "
            f"{row.get('weighted_train_core_images_count')} | "
            f"{str(row.get('sampled_distribution_changed')).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Required Artifacts",
            "",
        ]
    )
    for key, path in report["required_artifacts"].items():
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def build_summary(seed_reports: dict[int, dict[str, Any]]) -> dict[str, Any]:
    mean_clean = {key: mean(report["clean_paper_baseline"][key] for report in seed_reports.values()) for key in METRIC_KEYS}
    mean_cp = {
        key: mean(report["cp_catf_paper_mode_sampler_only"][key] for report in seed_reports.values())
        for key in METRIC_KEYS
    }
    mean_delta = {key: mean(report["delta_vs_clean"][key] for report in seed_reports.values()) for key in METRIC_KEYS}
    pass_count = sum(1 for report in seed_reports.values() if not report["constraint_failed"])
    effective_event_count = sum(
        report["sampler_only"]["effective_feedback_epoch_count"] for report in seed_reports.values()
    )
    weighted_by_seed = {
        str(seed): report["sampler_only"]["weighted_train_core_images_count"]
        for seed, report in seed_reports.items()
    }
    all_image_aug_zero = all(
        report["augmentation_counts"]["image_augmented"] == 0
        and report["augmentation_counts"]["roi_applied"] == 0
        and report["augmentation_counts"]["industrial_image_augmented"] == 0
        and report["augmentation_counts"]["router_random_draw_count"] == 0
        for report in seed_reports.values()
    )
    all_leakage_false = all(report["final_val_leakage_false"] for report in seed_reports.values())
    all_distribution_changed = all(
        report["sampler_only"]["sampled_distribution_changed"] for report in seed_reports.values()
    )
    main_candidate = pass_count == 3 and mean_delta["map50_95"] > 0.0
    summary = {
        "clean_paper_baseline": {str(seed): report["clean_paper_baseline"] for seed, report in seed_reports.items()},
        "cp_catf_paper_mode_sampler_only": {
            str(seed): report["cp_catf_paper_mode_sampler_only"] for seed, report in seed_reports.items()
        },
        "delta_vs_clean": {str(seed): report["delta_vs_clean"] for seed, report in seed_reports.items()},
        "constraint_failed": {str(seed): report["constraint_failed"] for seed, report in seed_reports.items()},
        "pass_count": pass_count,
        "three_seed_pass": pass_count == 3,
        "mean_clean": mean_clean,
        "mean_cp_catf_sampler_only": mean_cp,
        "mean_delta": mean_delta,
        "sampler_only_effective_seed_count": sum(
            1 for report in seed_reports.values() if report["sampler_only"]["effective"]
        ),
        "sampler_only_effective_event_count": effective_event_count,
        "weighted_train_core_images_total": sum(value or 0 for value in weighted_by_seed.values()),
        "weighted_train_core_images_by_seed": weighted_by_seed,
        "sampled_distribution_changed_all_seeds": all_distribution_changed,
        "image_augmentation_zero_all_seeds": all_image_aug_zero,
        "final_val_leakage_false_all_seeds": all_leakage_false,
        "paper_main_method_candidate": main_candidate,
        "development_mode_difference": (
            "Development-mode CP-CATF used existing validation diagnostics and included executable image/ROI "
            "augmentation evidence, so it is not the leakage-free paper-mode result."
        ),
        "noop_paper_mode_difference": (
            "The no-op paper-mode run had sampler-only pending or ineffective and did not change train sampling; "
            "this sampler-only run changes the dataloader via weighted index lists."
        ),
        "weak_image_aug_or_attenuation_recommendation": (
            "Do not promote weak image augmentation as the current main result. With seed1 failing constraints, "
            "first analyze sampler-only failure modes; weak_image_aug or attenuation remains an extension direction."
        ),
        "conclusion": (
            "Not a 3/3 paper-mode main-method result."
            if not main_candidate
            else "3/3 pass with positive mean mAP50-95 delta; viable paper-mode main-method candidate."
        ),
    }
    return summary


def render_summary_markdown(summary: dict[str, Any], seed_reports: dict[int, dict[str, Any]]) -> str:
    lines = [
        "# CP-CATF Paper-Mode Sampler-Only Multiseed Summary",
        "",
        "## Conclusion",
        "",
        f"- 3/3 pass: `{str(summary['three_seed_pass']).lower()}`",
        f"- pass_count: `{summary['pass_count']}/3`",
        f"- paper main-method candidate: `{str(summary['paper_main_method_candidate']).lower()}`",
        f"- image augmentation zero all seeds: `{str(summary['image_augmentation_zero_all_seeds']).lower()}`",
        f"- final val leakage false all seeds: `{str(summary['final_val_leakage_false_all_seeds']).lower()}`",
        f"- sampled distribution changed all seeds: `{str(summary['sampled_distribution_changed_all_seeds']).lower()}`",
        "",
        "Seed1 fails the requested constraints, so this multiseed run should not be reported as the paper main method.",
        "",
        "## Metrics",
        "",
        "| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | CP P | CP R | CP mAP50 | CP mAP50-95 | dP | dR | dM50 | dM95 | constraint_failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for seed, report in seed_reports.items():
        clean = report["clean_paper_baseline"]
        cp = report["cp_catf_paper_mode_sampler_only"]
        delta = report["delta_vs_clean"]
        lines.append(
            f"| {seed} | {fmt(clean['precision'])} | {fmt(clean['recall'])} | {fmt(clean['map50'])} | {fmt(clean['map50_95'])} | "
            f"{fmt(cp['precision'])} | {fmt(cp['recall'])} | {fmt(cp['map50'])} | {fmt(cp['map50_95'])} | "
            f"{signed(delta['precision'])} | {signed(delta['recall'])} | {signed(delta['map50'])} | {signed(delta['map50_95'])} | "
            f"{str(report['constraint_failed']).lower()} |"
        )
    mean_clean = summary["mean_clean"]
    mean_cp = summary["mean_cp_catf_sampler_only"]
    mean_delta = summary["mean_delta"]
    lines.append(
        f"| mean | {fmt(mean_clean['precision'])} | {fmt(mean_clean['recall'])} | {fmt(mean_clean['map50'])} | {fmt(mean_clean['map50_95'])} | "
        f"{fmt(mean_cp['precision'])} | {fmt(mean_cp['recall'])} | {fmt(mean_cp['map50'])} | {fmt(mean_cp['map50_95'])} | "
        f"{signed(mean_delta['precision'])} | {signed(mean_delta['recall'])} | {signed(mean_delta['map50'])} | {signed(mean_delta['map50_95'])} | n/a |"
    )
    lines.extend(
        [
            "",
            "## Sampler-Only Coverage",
            "",
            f"- sampler_only effective seed count: `{summary['sampler_only_effective_seed_count']}/3`",
            f"- sampler_only effective feedback event count: `{summary['sampler_only_effective_event_count']}`",
            f"- weighted train_core images total: `{summary['weighted_train_core_images_total']}`",
            f"- weighted train_core images by seed: `{summary['weighted_train_core_images_by_seed']}`",
            "",
            "| seed | effective epochs | weighted images | weighted_index_list_enabled | sampled distribution changed | image augmented | ROI applied | router draws | final val leakage false |",
            "|---:|---|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for seed, report in seed_reports.items():
        sampler = report["sampler_only"]
        aug = report["augmentation_counts"]
        lines.append(
            f"| {seed} | {sampler['effective_feedback_epochs']} | {sampler['weighted_train_core_images_count']} | "
            f"{str(sampler['weighted_index_list_enabled']).lower()} | {str(sampler['sampled_distribution_changed']).lower()} | "
            f"{aug['image_augmented']} | {aug['roi_applied']} | {aug['router_random_draw_count']} | "
            f"{str(report['final_val_leakage_false']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Development-mode difference: {summary['development_mode_difference']}",
            f"- No-op paper-mode difference: {summary['noop_paper_mode_difference']}",
            f"- Weak image augmentation / attenuation: {summary['weak_image_aug_or_attenuation_recommendation']}",
            "",
            "## Report Paths",
            "",
        ]
    )
    for seed, report in seed_reports.items():
        paths = report["required_artifacts"]
        lines.append(f"- seed{seed}: `{paths['sampler_only_report_md']}`")
    lines.extend(
        [
            f"- multiseed JSON: `{MULTISEED_ROOT / 'reports' / 'multiseed_sampler_only_summary.json'}`",
            f"- multiseed Markdown: `{MULTISEED_ROOT / 'reports' / 'multiseed_sampler_only_summary.md'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    seed_reports = {seed: build_seed_report(seed, run_dir) for seed, run_dir in SEED_RUNS.items()}
    summary = build_summary(seed_reports)
    reports_dir = MULTISEED_ROOT / "reports"
    write_json(reports_dir / "multiseed_sampler_only_summary.json", summary)
    (reports_dir / "multiseed_sampler_only_summary.md").write_text(
        render_summary_markdown(summary, seed_reports), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
