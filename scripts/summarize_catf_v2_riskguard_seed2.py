from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RISK_DIR = ROOT / "outputs/experiments/catf_v2_riskguard_seed2_50ep"
CLEAN_METRICS = (
    ROOT
    / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2/seed_2/clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json"
)
FIXED_METRICS = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_2/catf_v2/reports/final_metrics.json"
FIXED_ONLINE = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_2/catf_v2/reports/online_aug_stats.json"
FIXED_ROI = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_2/catf_v2/reports/roi_aug_stats.json"


def main() -> None:
    reports = RISK_DIR / "reports"
    risk_payload = read_json(reports / "final_metrics.json")
    clean_payload = read_json(CLEAN_METRICS)
    fixed_payload = read_json(FIXED_METRICS)
    fixed_online = read_json(FIXED_ONLINE)
    fixed_roi = read_json(FIXED_ROI)
    risk_events = read_json(reports / "riskguard_events.json").get("events", [])
    policy_history = read_json(reports / "policy_history.json").get("history", [])
    issue_epoch5 = read_json(reports / "issue_attribution_epoch_5.json")

    clean_metrics = metric_payload(clean_payload)
    fixed_metrics = metric_payload(fixed_payload)
    risk_metrics = metric_payload(risk_payload)
    risk_online = risk_payload.get("online_aug_stats", {})
    risk_roi = risk_payload.get("roi_aug_stats", {})
    constraint = risk_payload.get("constraint_scoring", {})

    class_rows = per_class_rows(clean_metrics, fixed_metrics, risk_metrics)
    class9 = next(row for row in class_rows if row["class_id"] == 9)
    non_active_regressions = [
        row
        for row in class_rows
        if row["class_id"] not in set(active_or_blocked_classes(policy_history))
        and (row["risk_delta_map50_95_vs_clean"] < -0.01 or row["risk_delta_ap50_vs_clean"] < -0.01)
    ]

    blocked_ops = [event.get("canonical_op_name") for event in risk_events if int(event.get("class_id", -1)) == 9]
    summary = {
        "seed": 2,
        "completed_50ep": bool(risk_payload.get("train", {}).get("success") and len(read_epoch_rows(RISK_DIR / "train/results.csv")) == 50),
        "riskguard_metrics": pick_metrics(risk_metrics),
        "clean_metrics": pick_metrics(clean_metrics),
        "fixed_catf_v2_metrics": pick_metrics(fixed_metrics),
        "delta_vs_clean": metric_delta(pick_metrics(risk_metrics), pick_metrics(clean_metrics)),
        "delta_vs_fixed": metric_delta(pick_metrics(risk_metrics), pick_metrics(fixed_metrics)),
        "constraint_failed": bool(constraint.get("constraint_failed")),
        "failure_reasons": list(constraint.get("failure_reasons") or []),
        "epoch5_class9_issue": (issue_epoch5.get("classes") or {}).get("9", {}),
        "class9_texture_ops_blocked": sorted(blocked_ops),
        "riskguard_blocked_op_count": len(risk_events),
        "riskguard_events_path": str((reports / "riskguard_events.json").resolve()),
        "sampler_only_fallback": any(event.get("sampler_only_fallback") for event in risk_events),
        "sample_weighting_effective": any(event.get("sample_weighting_effective") for event in risk_events),
        "industrial_samples_augmented": int(risk_online.get("samples_augmented", 0) or 0),
        "fixed_industrial_samples_augmented": int(fixed_online.get("samples_augmented", 0) or 0),
        "router_random_draw_count": int(risk_online.get("router_random_draw_count", 0) or 0),
        "roi_applied": int(risk_roi.get("roi_aug_applied", 0) or 0),
        "fixed_roi_applied": int(fixed_roi.get("roi_aug_applied", 0) or 0),
        "roi_affected_classes": risk_roi.get("affected_classes", {}),
        "ok3_active": any(1 in (record.get("active_classes") or []) for record in policy_history),
        "ok3_roi_applied": int((risk_roi.get("affected_classes") or {}).get("1", 0) or 0),
        "class9": class9,
        "non_active_regressions": non_active_regressions,
        "seed0_seed1_sanity_check_run": False,
        "seed0_seed1_sanity_check_reason": "not_run_because_seed2_constraint_failed",
        "recommendation": "RiskGuard should be kept as a targeted safety guard, but this seed2 run shows it is necessary and not sufficient as a final CATF-v2 fix.",
    }
    write_json(reports / "riskguard_seed2_summary.json", summary)
    report = build_report(summary, class_rows)
    write_text(reports / "compare_with_clean_and_fixed_catf_v2.md", report)
    append_final_report(reports / "final_report.md", report)


def metric_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return (payload.get("val") or {}).get("metrics") or payload.get("metrics") or {}


def pick_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0) or 0.0) for key in ("precision", "recall", "map50", "map50_95")}


def metric_delta(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in ("precision", "recall", "map50", "map50_95")}


def per_class_rows(clean_metrics: dict[str, Any], fixed_metrics: dict[str, Any], risk_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    clean = {int(row["class_id"]): row for row in clean_metrics.get("per_class", [])}
    fixed = {int(row["class_id"]): row for row in fixed_metrics.get("per_class", [])}
    risk = {int(row["class_id"]): row for row in risk_metrics.get("per_class", [])}
    rows: list[dict[str, Any]] = []
    for class_id in sorted(set(clean) | set(fixed) | set(risk)):
        c = clean.get(class_id, {})
        f = fixed.get(class_id, {})
        r = risk.get(class_id, {})
        rows.append(
            {
                "class_id": class_id,
                "name": r.get("name") or c.get("name") or f.get("name") or str(class_id),
                "clean_recall": c.get("recall"),
                "fixed_recall": f.get("recall"),
                "risk_recall": r.get("recall"),
                "risk_delta_recall_vs_clean": value(r, "recall") - value(c, "recall"),
                "risk_delta_recall_vs_fixed": value(r, "recall") - value(f, "recall"),
                "clean_ap50": c.get("ap50"),
                "fixed_ap50": f.get("ap50"),
                "risk_ap50": r.get("ap50"),
                "risk_delta_ap50_vs_clean": value(r, "ap50") - value(c, "ap50"),
                "risk_delta_ap50_vs_fixed": value(r, "ap50") - value(f, "ap50"),
                "clean_ap50_95": c.get("ap50_95"),
                "fixed_ap50_95": f.get("ap50_95"),
                "risk_ap50_95": r.get("ap50_95"),
                "risk_delta_map50_95_vs_clean": value(r, "ap50_95") - value(c, "ap50_95"),
                "risk_delta_map50_95_vs_fixed": value(r, "ap50_95") - value(f, "ap50_95"),
            }
        )
    return rows


def active_or_blocked_classes(history: list[dict[str, Any]]) -> list[int]:
    classes: set[int] = set()
    for record in history:
        classes.update(int(item) for item in (record.get("active_classes") or []))
        classes.update(int(item) for item in (record.get("riskguard_blocked_classes") or []))
    return sorted(classes)


def value(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def build_report(summary: dict[str, Any], class_rows: list[dict[str, Any]]) -> str:
    delta_clean = summary["delta_vs_clean"]
    delta_fixed = summary["delta_vs_fixed"]
    class9 = summary["class9"]
    lines = [
        "# CATF-v2 RiskGuard Seed2 Validation",
        "",
        "## Verdict",
        "",
        f"- Completed 50ep: `{str(summary['completed_50ep']).lower()}`",
        f"- constraint_failed: `{str(summary['constraint_failed']).lower()}`",
        f"- Failure reasons: `{summary['failure_reasons']}`",
        "- Seed0/seed1 sanity check: `not run` because seed2 did not pass the constraint gate.",
        "- Interpretation: RiskGuard successfully blocks the audited class 9 texture intervention, but it is not sufficient to recover seed2 overall.",
        "",
        "## Metrics",
        "",
        "| group | P | R | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        metrics_row("clean seed2", summary["clean_metrics"]),
        metrics_row("fixed CATF-v2 seed2", summary["fixed_catf_v2_metrics"]),
        metrics_row("RiskGuard seed2", summary["riskguard_metrics"]),
        "",
        "## Deltas",
        "",
        f"- RiskGuard vs clean: P `{fmt(delta_clean['precision'])}`, R `{fmt(delta_clean['recall'])}`, "
        f"mAP50 `{fmt(delta_clean['map50'])}`, mAP50-95 `{fmt(delta_clean['map50_95'])}`.",
        f"- RiskGuard vs fixed CATF-v2: P `{fmt(delta_fixed['precision'])}`, R `{fmt(delta_fixed['recall'])}`, "
        f"mAP50 `{fmt(delta_fixed['map50'])}`, mAP50-95 `{fmt(delta_fixed['map50_95'])}`.",
        "",
        "## RiskGuard Events",
        "",
        f"- Epoch5 class 9 dominant issue: `{summary['epoch5_class9_issue'].get('dominant_issue')}`",
        f"- Class 9 texture ops blocked: `{summary['class9_texture_ops_blocked']}`",
        f"- Blocked op count: `{summary['riskguard_blocked_op_count']}`",
        f"- Sampler-only fallback: `{str(summary['sampler_only_fallback']).lower()}`",
        f"- Sample weighting effective: `{str(summary['sample_weighting_effective']).lower()}`; dataloader integration remains pending.",
        f"- Event JSON: `{summary['riskguard_events_path']}`",
        "",
        "## Augmentation Stats",
        "",
        f"- Industrial samples augmented: `{summary['industrial_samples_augmented']}` vs fixed `{summary['fixed_industrial_samples_augmented']}`.",
        f"- ROI applied: `{summary['roi_applied']}` vs fixed `{summary['fixed_roi_applied']}`.",
        f"- ROI affected classes: `{summary['roi_affected_classes']}`.",
        f"- Router random draw count: `{summary['router_random_draw_count']}`.",
        f"- OK3 active: `{str(summary['ok3_active']).lower()}`; OK3 ROI applied: `{summary['ok3_roi_applied']}`.",
        "",
        "## Class 9",
        "",
        f"- Recall clean/fixed/RiskGuard: `{fmt(class9['clean_recall'])}` / `{fmt(class9['fixed_recall'])}` / `{fmt(class9['risk_recall'])}`.",
        f"- AP50 clean/fixed/RiskGuard: `{fmt(class9['clean_ap50'])}` / `{fmt(class9['fixed_ap50'])}` / `{fmt(class9['risk_ap50'])}`.",
        f"- AP50-95 clean/fixed/RiskGuard: `{fmt(class9['clean_ap50_95'])}` / `{fmt(class9['fixed_ap50_95'])}` / `{fmt(class9['risk_ap50_95'])}`.",
        "- Class 9 recovered versus fixed CATF-v2 after blocking the texture ROI combination, but global constraints still failed.",
        "",
        "## Per-Class Residual Risk",
        "",
        "| class | name | dRecall vs clean | dAP50 vs clean | dAP50-95 vs clean |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in sorted(class_rows, key=lambda item: item["risk_delta_map50_95_vs_clean"]):
        if row["risk_delta_map50_95_vs_clean"] < -0.01 or row["risk_delta_ap50_vs_clean"] < -0.01:
            lines.append(
                f"| {row['class_id']} | {row['name']} | {fmt(row['risk_delta_recall_vs_clean'])} | "
                f"{fmt(row['risk_delta_ap50_vs_clean'])} | {fmt(row['risk_delta_map50_95_vs_clean'])} |"
            )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "- RiskGuard should be added as a targeted CATF-v2 safety mechanism because it prevents the known class 9 + ROI texture failure path without seed-specific logic.",
            "- This seed2 validation does not justify running seed0/seed1 sanity yet: seed2 still fails due to residual global Precision/mAP degradation and class12-only ROI activity.",
            "- The next minimal fix should combine RiskGuard with a gate/causal probe that rejects later class-op interventions when global Precision/mAP constraints deteriorate.",
        ]
    )
    return "\n".join(lines) + "\n"


def metrics_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {fmt(metrics['precision'])} | {fmt(metrics['recall'])} | "
        f"{fmt(metrics['map50'])} | {fmt(metrics['map50_95'])} |"
    )


def fmt(value: Any) -> str:
    return f"{float(value):.4f}"


def read_epoch_rows(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:] if line.strip()]


def append_final_report(path: Path, addendum: str) -> None:
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "## RiskGuard Seed2 Validation"
    addendum_text = addendum.replace("# CATF-v2 RiskGuard Seed2 Validation", marker, 1)
    if marker in original:
        original = original.split(marker, 1)[0].rstrip() + "\n"
    write_text(path, original.rstrip() + "\n\n" + addendum_text)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
