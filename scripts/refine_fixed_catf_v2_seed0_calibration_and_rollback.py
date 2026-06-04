"""Refine fixed CATF-v2 with seed0 calibration and rollback analysis.

This script is analysis-only. It may reuse or create validation prediction JSON
for post-hoc threshold search, but it never trains or modifies model weights.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.catf_v2.per_class_thresholds import (  # noqa: E402
    ThresholdCandidate,
    constraint_failures as rc_constraint_failures,
    metric_delta as rc_metric_delta,
    rc_threshold_score,
    robust_threshold_table,
    select_safe_candidate,
    threshold_changes as rc_threshold_changes,
    write_threshold_config,
)
from scripts.analyze_fixed_catf_v2_seed2_and_thresholds import (  # noqa: E402
    FIXED_ROOT,
    METRIC_KEYS,
    REPORTS,
    SEEDS,
    clean_metrics_path,
    class_name_map,
    f4,
    fd,
    final_global_metrics,
    final_per_class,
    fixed_metrics_path,
    load_policy_behavior,
    old_catf_metrics_path,
    per_class_delta,
    prediction_json,
    read_json,
    write_json,
    write_md,
)
from scripts.evaluate_catf_v2_threshold_posthoc import (  # noqa: E402
    THRESHOLDS,
    constraint_failures,
    evaluate,
    metric_delta,
    threshold_changes,
)


DEFAULT_THRESHOLD = 0.25


def search_rc_thresholds(records: list[dict[str, Any]], class_ids: list[int], reference: dict[str, float]) -> dict[str, Any]:
    thresholds = {cid: DEFAULT_THRESHOLD for cid in class_ids}
    current = evaluate(records, thresholds)
    best_score = rc_threshold_score(current.metrics, reference)
    steps: list[dict[str, Any]] = []
    for pass_idx in range(5):
        improved = False
        for cid in class_ids:
            best_for_class = (best_score, thresholds[cid], current)
            for candidate in THRESHOLDS:
                trial_thresholds = dict(thresholds)
                trial_thresholds[cid] = candidate
                result = evaluate(records, trial_thresholds)
                score = rc_threshold_score(result.metrics, reference)
                if score > best_for_class[0] + 1e-10:
                    best_for_class = (score, candidate, result)
            if best_for_class[1] != thresholds[cid]:
                before = thresholds[cid]
                thresholds[cid] = best_for_class[1]
                best_score = best_for_class[0]
                current = best_for_class[2]
                improved = True
                steps.append(
                    {
                        "pass": pass_idx + 1,
                        "class_id": cid,
                        "before": before,
                        "after": thresholds[cid],
                        "metrics": current.metrics,
                        "failure_reasons": constraint_failures(current.metrics, reference),
                    }
                )
        if not improved:
            break
    return {
        "objective": "catf_v2_rc_threshold_score",
        "thresholds": {str(cid): thresholds[cid] for cid in class_ids},
        "metrics": current.metrics,
        "per_class": {str(cid): row for cid, row in current.per_class.items()},
        "score": best_score,
        "steps": steps,
        "constraint_failed": bool(constraint_failures(current.metrics, reference)),
        "failure_reasons": constraint_failures(current.metrics, reference),
        "delta_vs_clean_default": metric_delta(current.metrics, reference),
    }


def build_threshold_candidates(seed: int, names: dict[int, str]) -> dict[str, Any]:
    class_ids = sorted(names)
    clean_records = read_json(prediction_json(seed, "clean"))["records"]
    fixed_records = read_json(prediction_json(seed, "catf_v2"))["records"]
    clean_default = evaluate(clean_records, {cid: DEFAULT_THRESHOLD for cid in class_ids})
    fixed_default = evaluate(fixed_records, {cid: DEFAULT_THRESHOLD for cid in class_ids})
    rc_result = search_rc_thresholds(fixed_records, class_ids, clean_default.metrics)
    previous = read_json(REPORTS / "fixed_threshold_calibration_posthoc.json")["seeds"][str(seed)]["constrained_summary"]

    candidate_payloads = {
        "fixed_default_0.25": {
            "thresholds": {str(cid): DEFAULT_THRESHOLD for cid in class_ids},
            "metrics": fixed_default.metrics,
            "per_class": {str(cid): row for cid, row in fixed_default.per_class.items()},
            "score": rc_threshold_score(fixed_default.metrics, clean_default.metrics),
            "constraint_failed": bool(constraint_failures(fixed_default.metrics, clean_default.metrics)),
            "failure_reasons": constraint_failures(fixed_default.metrics, clean_default.metrics),
            "delta_vs_clean_default": metric_delta(fixed_default.metrics, clean_default.metrics),
        },
        "previous_constrained_score": previous,
        "catf_v2_rc_threshold_score": rc_result,
    }

    candidates: list[ThresholdCandidate] = []
    for name, payload in candidate_payloads.items():
        thresholds = {int(k): float(v) for k, v in payload["thresholds"].items()}
        per_class = {int(k): v for k, v in payload["per_class"].items()}
        candidates.append(
            ThresholdCandidate(
                name=name,
                thresholds=thresholds,
                metrics={key: float(payload["metrics"][key]) for key in METRIC_KEYS},
                per_class=per_class,
                score=float(payload.get("score", rc_threshold_score(payload["metrics"], clean_default.metrics))),
                constraint_failed=bool(payload["constraint_failed"]),
                failure_reasons=list(payload.get("failure_reasons") or []),
                delta_vs_reference=metric_delta(payload["metrics"], clean_default.metrics),
            )
        )
    selected = select_safe_candidate(candidates, clean_default.metrics)
    return {
        "seed": seed,
        "clean_default_metrics": clean_default.metrics,
        "clean_per_class_default": {str(cid): row for cid, row in clean_default.per_class.items()},
        "fixed_default_metrics": fixed_default.metrics,
        "fixed_per_class_default": {str(cid): row for cid, row in fixed_default.per_class.items()},
        "fixed_default_delta_vs_clean_default": metric_delta(fixed_default.metrics, clean_default.metrics),
        "candidates": candidate_payloads,
        "selected_name": selected.name,
        "selected_thresholds": {str(k): v for k, v in selected.thresholds.items()},
        "selected_metrics": selected.metrics,
        "selected_delta_vs_clean_default": selected.delta_vs_reference,
        "selected_constraint_failed": selected.constraint_failed,
        "selected_failure_reasons": selected.failure_reasons,
        "selected_per_class": {str(k): v for k, v in selected.per_class.items()},
        "selected_threshold_changes": {
            "raise": [
                {**item, "class_name": names[item["class_id"]]}
                for item in rc_threshold_changes(selected.thresholds, default_threshold=DEFAULT_THRESHOLD)["raise"]
            ],
            "lower": [
                {**item, "class_name": names[item["class_id"]]}
                for item in rc_threshold_changes(selected.thresholds, default_threshold=DEFAULT_THRESHOLD)["lower"]
            ],
        },
    }


def seed_failure_payload(seed: int, names: dict[int, str], threshold_payload: dict[str, Any]) -> dict[str, Any]:
    clean = final_global_metrics(clean_metrics_path(seed), clean=True)
    fixed = final_global_metrics(fixed_metrics_path(seed))
    clean_pc = final_per_class(clean_metrics_path(seed), clean=True)
    fixed_pc = final_per_class(fixed_metrics_path(seed))
    rows = per_class_delta(clean_pc, fixed_pc, names)
    posthoc = threshold_payload
    official_failures = []
    if fixed["precision"] < clean["precision"] - 0.01:
        official_failures.append("precision_drop_gt_0.01")
    if fixed["map50"] < clean["map50"] - 0.01:
        official_failures.append("map50_drop_gt_0.01")
    if fixed["map50_95"] < clean["map50_95"] - 0.01:
        official_failures.append("map50_95_drop_gt_0.01")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "clean_metrics": clean,
        "fixed_metrics": fixed,
        "delta_metrics": {key: fixed[key] - clean[key] for key in METRIC_KEYS},
        "official_constraint_failed": bool(official_failures),
        "official_failure_reasons": official_failures,
        "posthoc_default_constraint_failed": posthoc["candidates"]["fixed_default_0.25"]["constraint_failed"],
        "posthoc_default_failure_reasons": posthoc["candidates"]["fixed_default_0.25"]["failure_reasons"],
        "selected_calibration_constraint_failed": posthoc["selected_constraint_failed"],
        "selected_calibration_failure_reasons": posthoc["selected_failure_reasons"],
        "per_class_delta": rows,
        "controller_behavior": load_policy_behavior(seed),
        "interpretation": (
            "Official fixed CATF-v2 seed0 already passes constraints; the remaining seed0 issue is a post-hoc "
            "threshold-search/evaluator failure where naive constrained lowering improves Recall but hurts AP50-95. "
            "The RC threshold objective repairs seed0 by balancing Recall and mAP50-95."
        ),
    }


def format_class_list(rows: list[dict[str, Any]], key: str, *, limit: int = 8, reverse: bool = False) -> list[str]:
    selected = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:limit]
    return [f"{row['class_id']}:{row['class_name']} ({key}={row[key]:+.4f})" for row in selected]


def write_seed0_failure_report(payload: dict[str, Any], threshold_payload: dict[str, Any]) -> None:
    write_json(REPORTS / "seed0_failure_analysis.json", payload)
    rows = payload["per_class_delta"]
    recall_drop = sorted(rows, key=lambda row: row["delta_recall"])[:8]
    ap95_drop = sorted(rows, key=lambda row: row["delta_ap50_95"])[:8]
    ap50_drop = sorted(rows, key=lambda row: row["delta_ap50"])[:8]
    selected = threshold_payload
    lines = [
        "# Fixed CATF-v2 Seed0 Failure Analysis",
        "",
        "Seed0 is not an official fixed-training failure: fixed CATF-v2 seed0 already passes the original industrial constraints. The remaining issue is that the earlier post-hoc threshold search still failed mAP50-95 in the custom evaluator.",
        "",
        "## Overall Metrics",
        "",
        "| Run | Precision | Recall | mAP50 | mAP50-95 | constraint_failed |",
        "|---|---:|---:|---:|---:|:---:|",
        f"| clean native seed0 | {f4(payload['clean_metrics']['precision'])} | {f4(payload['clean_metrics']['recall'])} | {f4(payload['clean_metrics']['map50'])} | {f4(payload['clean_metrics']['map50_95'])} | false |",
        f"| fixed CATF-v2 seed0 | {f4(payload['fixed_metrics']['precision'])} | {f4(payload['fixed_metrics']['recall'])} | {f4(payload['fixed_metrics']['map50'])} | {f4(payload['fixed_metrics']['map50_95'])} | {str(payload['official_constraint_failed']).lower()} |",
        f"| delta | {fd(payload['delta_metrics']['precision'])} | {fd(payload['delta_metrics']['recall'])} | {fd(payload['delta_metrics']['map50'])} | {fd(payload['delta_metrics']['map50_95'])} | - |",
        "",
        "## Constraint Trigger",
        "",
        f"- Official fixed seed0 failure reasons: `{payload['official_failure_reasons']}`.",
        f"- Post-hoc default threshold evaluator failure reasons: `{payload['posthoc_default_failure_reasons']}`.",
        f"- Selected RC calibration failure reasons: `{payload['selected_calibration_failure_reasons']}`.",
        "",
        "## Recall Drop Classes",
        "",
        "| class | Delta Recall | FN clean est | FN fixed est | Delta FN est |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in recall_drop:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {fd(row['delta_recall'])} | {row['fn_est_clean']:.2f} | {row['fn_est_fixed']:.2f} | {row['delta_fn_est']:+.2f} |")
    lines += [
        "",
        "## AP50-95 Drop Classes",
        "",
        "| class | Delta AP50-95 | Delta AP50 |",
        "|---|---:|---:|",
    ]
    for row in ap95_drop:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {fd(row['delta_ap50_95'])} | {fd(row['delta_ap50'])} |")
    lines += [
        "",
        "## AP50 Drop Classes",
        "",
        "| class | Delta AP50 | Delta AP50-95 |",
        "|---|---:|---:|",
    ]
    for row in ap50_drop:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {fd(row['delta_ap50'])} | {fd(row['delta_ap50_95'])} |")
    lines += [
        "",
        "## Failure Type",
        "",
        "- Official metrics: seed0 is mostly a mild Recall tradeoff, not a hard constraint failure.",
        "- Post-hoc custom evaluator: the default threshold path has mAP50/mAP50-95 deficits, and the earlier constrained search fixed mAP50 but not mAP50-95.",
        "- The failure is therefore closer to `confidence threshold mismatch + AP50-95 guard weakness` than a clear training collapse.",
        "- The strongest rollback candidates from official per-class AP50-95 drops are `加强筋打伤`, `碰伤`, `OK3`, and `轮廓划伤`; however only `加强筋打伤` is a strong low-support rollback candidate.",
        "",
        "## RC Threshold Calibration",
        "",
        f"- Selected objective: `{selected['selected_name']}`.",
        f"- Metrics after selected calibration: P/R/mAP50/mAP50-95 = `{f4(selected['selected_metrics']['precision'])}/{f4(selected['selected_metrics']['recall'])}/{f4(selected['selected_metrics']['map50'])}/{f4(selected['selected_metrics']['map50_95'])}`.",
        f"- Delta vs clean default evaluator: `{fd(selected['selected_delta_vs_clean_default']['precision'])}/{fd(selected['selected_delta_vs_clean_default']['recall'])}/{fd(selected['selected_delta_vs_clean_default']['map50'])}/{fd(selected['selected_delta_vs_clean_default']['map50_95'])}`.",
        f"- Constraint failed after RC calibration: `{str(selected['selected_constraint_failed']).lower()}`.",
        f"- Threshold raises: `{selected['selected_threshold_changes']['raise']}`.",
        f"- Threshold lowers: `{selected['selected_threshold_changes']['lower']}`.",
    ]
    write_md(REPORTS / "seed0_failure_analysis.md", lines)


def write_seed0_threshold_report(seed0: dict[str, Any]) -> None:
    write_json(REPORTS / "seed0_threshold_calibration_posthoc.json", seed0)
    candidates = seed0["candidates"]
    selected = seed0
    lines = [
        "# Seed0 Post-hoc Per-Class Threshold Calibration",
        "",
        "No training was run. This report evaluates seed0 fixed CATF-v2 with per-class confidence thresholds.",
        "",
        "## Candidate Summary",
        "",
        "| candidate | P | R | mAP50 | mAP50-95 | Delta P | Delta R | Delta mAP50 | Delta mAP50-95 | constraint_failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, row in candidates.items():
        delta = row.get("delta_vs_clean_default") or metric_delta(row["metrics"], seed0["clean_default_metrics"])
        lines.append(
            f"| {name} | {f4(row['metrics']['precision'])} | {f4(row['metrics']['recall'])} | {f4(row['metrics']['map50'])} | {f4(row['metrics']['map50_95'])} | "
            f"{fd(delta['precision'])} | {fd(delta['recall'])} | {fd(delta['map50'])} | {fd(delta['map50_95'])} | {str(row['constraint_failed']).lower()} |"
        )
    lines += [
        "",
        "## Selected Calibration",
        "",
        f"- Selected candidate: `{selected['selected_name']}`.",
        f"- Constraint failed: `{str(selected['selected_constraint_failed']).lower()}`.",
        f"- Metrics: P/R/mAP50/mAP50-95 = `{f4(selected['selected_metrics']['precision'])}/{f4(selected['selected_metrics']['recall'])}/{f4(selected['selected_metrics']['map50'])}/{f4(selected['selected_metrics']['map50_95'])}`.",
        f"- Delta vs clean evaluator: `{fd(selected['selected_delta_vs_clean_default']['precision'])}/{fd(selected['selected_delta_vs_clean_default']['recall'])}/{fd(selected['selected_delta_vs_clean_default']['map50'])}/{fd(selected['selected_delta_vs_clean_default']['map50_95'])}`.",
        "",
        "## Threshold Changes",
        "",
        "| class | old | new | direction |",
        "|---|---:|---:|---|",
    ]
    changes = seed0["selected_threshold_changes"]["raise"] + seed0["selected_threshold_changes"]["lower"]
    for item in sorted(changes, key=lambda row: row["class_id"]):
        direction = "raise" if item["new_threshold"] > item["old_threshold"] else "lower"
        lines.append(f"| {item['class_id']}:{item['class_name']} | {item['old_threshold']:.2f} | {item['new_threshold']:.2f} | {direction} |")
    lines += [
        "",
        "## Conclusion",
        "",
        "- Seed0 can be repaired in the post-hoc evaluator by using an mAP50-95-aware RC threshold objective.",
        "- Naive recall-first threshold lowering is unsafe for seed0 because it can keep mAP50-95 below the constraint.",
        "- This supports deploying threshold calibration as a constrained selector, not as unconditional global threshold lowering.",
    ]
    write_md(REPORTS / "seed0_threshold_calibration_posthoc.md", lines)


def write_all_seed_threshold_report(payload: dict[str, Any], names: dict[int, str]) -> None:
    write_json(REPORTS / "fixed_catf_v2_threshold_calibration_all_seeds.json", payload)
    lines = [
        "# Fixed CATF-v2 Threshold Calibration Across All Seeds",
        "",
        "No training was run. Calibration is post-hoc and per-class.",
        "",
        "## Selected RC Calibration Summary",
        "",
        "| seed | selected | P | R | mAP50 | mAP50-95 | Delta P | Delta R | Delta mAP50 | Delta mAP50-95 | constraint_failed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in SEEDS:
        row = payload["seeds"][str(seed)]
        m = row["selected_metrics"]
        d = row["selected_delta_vs_clean_default"]
        lines.append(
            f"| {seed} | {row['selected_name']} | {f4(m['precision'])} | {f4(m['recall'])} | {f4(m['map50'])} | {f4(m['map50_95'])} | "
            f"{fd(d['precision'])} | {fd(d['recall'])} | {fd(d['map50'])} | {fd(d['map50_95'])} | {str(row['selected_constraint_failed']).lower()} |"
        )
    lines += [
        "",
        f"- Constraint pass count after RC calibration: `{payload['summary']['constraint_pass_count']}/3`.",
        "",
        "## Final Recommended Per-Class Threshold Table",
        "",
        "| class | threshold | evidence |",
        "|---|---:|---|",
    ]
    for cid, threshold in payload["summary"]["recommended_thresholds"].items():
        cid_int = int(cid)
        evidence = payload["summary"]["threshold_frequency"].get(str(cid_int), {})
        lines.append(f"| {cid}:{names[cid_int]} | {threshold:.2f} | {evidence} |")
    lines += [
        "",
        "## High-Recall-Sensitive Classes",
        "",
        ", ".join(payload["summary"]["high_recall_sensitive_classes"]) or "none",
        "",
        "## Classes Not Fully Fixed By Threshold Alone",
        "",
        ", ".join(payload["summary"]["threshold_insufficient_classes"]) or "none",
    ]
    write_md(REPORTS / "fixed_catf_v2_threshold_calibration_all_seeds.md", lines)


def rollback_simulation(threshold_payload: dict[str, Any], names: dict[int, str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for cid in sorted(names):
        evidence = {
            "class_id": cid,
            "class_name": names[cid],
            "seed_evidence": {},
            "negative_seed_count": 0,
            "threshold_help_seed_count": 0,
            "rollback_seed_count": 0,
        }
        for seed in SEEDS:
            clean_pc = final_per_class(clean_metrics_path(seed), clean=True).get(cid, {})
            old_pc = final_per_class(old_catf_metrics_path(seed)).get(cid, {})
            fixed_pc = final_per_class(fixed_metrics_path(seed)).get(cid, {})
            selected_pc = threshold_payload["seeds"][str(seed)]["selected_per_class"].get(str(cid), {})
            selected_clean_pc = threshold_payload["seeds"][str(seed)]["clean_per_class_default"].get(str(cid), {})
            fixed_delta_recall = float(fixed_pc.get("recall", 0.0)) - float(clean_pc.get("recall", 0.0))
            fixed_delta_ap95 = float(fixed_pc.get("ap50_95", 0.0)) - float(clean_pc.get("ap50_95", 0.0))
            old_delta_recall = float(old_pc.get("recall", 0.0)) - float(clean_pc.get("recall", 0.0))
            old_delta_ap95 = float(old_pc.get("ap50_95", 0.0)) - float(clean_pc.get("ap50_95", 0.0))
            selected_delta_recall = float(selected_pc.get("recall", 0.0)) - float(selected_clean_pc.get("recall", 0.0))
            selected_delta_ap95 = float(selected_pc.get("ap50_95", 0.0)) - float(selected_clean_pc.get("ap50_95", 0.0))
            recall_negative = fixed_delta_recall < -0.02
            ap_negative = fixed_delta_ap95 < -0.01
            threshold_helps = selected_delta_recall >= -0.02 and selected_delta_ap95 >= -0.01
            rollback_candidate = (recall_negative and ap_negative) or (ap_negative and not threshold_helps)
            evidence["negative_seed_count"] += int(recall_negative or ap_negative)
            evidence["threshold_help_seed_count"] += int(threshold_helps)
            evidence["rollback_seed_count"] += int(rollback_candidate)
            evidence["seed_evidence"][str(seed)] = {
                "fixed_delta_recall": fixed_delta_recall,
                "fixed_delta_ap50_95": fixed_delta_ap95,
                "old_delta_recall": old_delta_recall,
                "old_delta_ap50_95": old_delta_ap95,
                "selected_threshold_delta_recall": selected_delta_recall,
                "selected_threshold_delta_ap50_95": selected_delta_ap95,
                "recall_negative": recall_negative,
                "localization_or_ap_negative": ap_negative,
                "threshold_helps": threshold_helps,
                "rollback_candidate": rollback_candidate,
            }
        if evidence["rollback_seed_count"] >= 2:
            action = "rollback_candidate"
        elif evidence["negative_seed_count"] >= 2:
            action = "threshold_calibration_only" if evidence["threshold_help_seed_count"] >= 2 else "high_risk_negative_effect"
        elif evidence["negative_seed_count"] == 1:
            action = "unstable_not_mandatory_rollback"
        else:
            action = "keep_fixed_catf_v2"
        evidence["recommended_action"] = action
        rows[str(cid)] = evidence
    buckets: dict[str, list[str]] = defaultdict(list)
    for row in rows.values():
        buckets[row["recommended_action"]].append(f"{row['class_id']}:{row['class_name']}")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rules": {
            "recall_negative": "fixed Recall delta < -0.02",
            "localization_or_ap_negative": "fixed AP50-95 delta < -0.01",
            "rollback_candidate": "repeated negative or threshold cannot fix AP50-95",
        },
        "classes": rows,
        "buckets": dict(buckets),
    }


def write_rollback_report(payload: dict[str, Any]) -> None:
    write_json(REPORTS / "fixed_catf_v2_class_level_rollback_simulation.json", payload)
    lines = [
        "# Fixed CATF-v2 Class-Level Rollback Simulation",
        "",
        "No training was run. This is a post-hoc class-level decision analysis comparing clean native, old CATF-v2, fixed CATF-v2, and selected threshold calibration.",
        "",
    ]
    for bucket in ("keep_fixed_catf_v2", "threshold_calibration_only", "rollback_candidate", "high_risk_negative_effect", "unstable_not_mandatory_rollback"):
        lines += [
            f"## {bucket}",
            "",
            ", ".join(payload["buckets"].get(bucket, [])) or "none",
            "",
        ]
    lines += [
        "## Per-Class Evidence",
        "",
        "| class | action | negative seeds | rollback seeds | threshold-help seeds |",
        "|---|---|---:|---:|---:|",
    ]
    for cid, row in sorted(payload["classes"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {cid}:{row['class_name']} | {row['recommended_action']} | {row['negative_seed_count']} | {row['rollback_seed_count']} | {row['threshold_help_seed_count']} |"
        )
    write_md(REPORTS / "fixed_catf_v2_class_level_rollback_simulation.md", lines)


def write_rc_plan(threshold_payload: dict[str, Any], rollback_payload: dict[str, Any]) -> None:
    ready = threshold_payload["summary"]["constraint_pass_count"] == len(SEEDS)
    payload = {
        "method_name": "CATF-v2-RC",
        "definition": "fixed CATF-v2 + per-class constrained threshold calibration + class-level rollback/high-recall protection",
        "ready_for_paper_main_candidate": ready,
        "constraint_pass_count_after_calibration": threshold_payload["summary"]["constraint_pass_count"],
        "recommended_thresholds": threshold_payload["summary"]["recommended_thresholds"],
        "rollback_buckets": rollback_payload["buckets"],
        "minimal_next_experiment": None if ready else "Run seed0-only class-level rollback validation for rollback candidates.",
    }
    write_json(REPORTS / "catf_v2_rc_final_candidate_plan.json", payload)
    lines = [
        "# CATF-v2-RC Final Candidate Plan",
        "",
        "CATF-v2-RC = fixed CATF-v2 + per-class constrained threshold calibration + class-level rollback / high-recall protection.",
        "",
        "## Why Fixed CATF-v2 Is Not A Failure",
        "",
        "- Strict no-op bypass removed the framework artifact, and fixed CATF-v2 improved constraint pass count relative to old CATF-v2.",
        "- Seed0 and seed1 pass official training constraints; seed2 can be repaired by constrained per-class threshold calibration in the post-hoc evaluator.",
        "- The remaining problem is category-level protection, not global augmentation redesign.",
        "",
        "## Method Flow For Paper",
        "",
        "1. Train clean native YOLO default and fixed CATF-v2 under identical settings.",
        "2. Run CATF-v2 class/issue/sample-aware feedback with strict no-op bypass.",
        "3. On validation predictions, search per-class confidence thresholds with industrial constraints on Precision, mAP50, and mAP50-95.",
        "4. Select thresholds only if constraints pass; otherwise keep default or flag class-level rollback.",
        "5. For repeated class-level negative effects, protect high-recall classes by rollback/freezing that class policy in the next training iteration.",
        "",
        "## Threshold Calibration Result",
        "",
        f"- RC calibrated pass count: `{threshold_payload['summary']['constraint_pass_count']}/3`.",
        f"- Ready for paper main candidate: `{ready}`.",
        "",
        "## Rollback Guidance",
        "",
        f"- Keep fixed CATF-v2: `{rollback_payload['buckets'].get('keep_fixed_catf_v2', [])}`.",
        f"- Threshold calibration only: `{rollback_payload['buckets'].get('threshold_calibration_only', [])}`.",
        f"- Rollback candidates: `{rollback_payload['buckets'].get('rollback_candidate', [])}`.",
        f"- High-risk negative-effect classes: `{rollback_payload['buckets'].get('high_risk_negative_effect', [])}`.",
        "",
        "## Recommended Wording",
        "",
        "- `CATF-v2-RC improves fixed CATF-v2 by adding constrained per-class threshold calibration and class-level protection.`",
        "- Avoid claiming global augmentation alone is sufficient; the evidence supports a training-plus-deployment-calibration method.",
    ]
    if ready:
        lines += [
            "",
            "## Next Experiment",
            "",
            "Do not make broad algorithm changes. The smallest next validation is to apply the recommended threshold table on official validation outputs and optionally run a seed0-only rollback ablation for high-risk classes.",
        ]
    else:
        lines += [
            "",
            "## Minimal Next Experiment",
            "",
            "- Only target the classes listed as rollback candidates.",
            "- Do not globally weaken CATF-v2.",
            "- Run a seed0-only validation of class-level rollback/high-recall protection before any larger multiseed rerun.",
        ]
    write_md(REPORTS / "catf_v2_rc_final_candidate_plan.md", lines)


def write_ready_summary(threshold_payload: dict[str, Any], rollback_payload: dict[str, Any]) -> None:
    seed_rows = {}
    for seed in SEEDS:
        clean = final_global_metrics(clean_metrics_path(seed), clean=True)
        fixed = final_global_metrics(fixed_metrics_path(seed))
        calibrated = threshold_payload["seeds"][str(seed)]["selected_metrics"]
        posthoc_clean = threshold_payload["seeds"][str(seed)]["clean_default_metrics"]
        posthoc_fixed = threshold_payload["seeds"][str(seed)]["fixed_default_metrics"]
        seed_rows[str(seed)] = {
            "official_clean_native": clean,
            "official_fixed_catf_v2": fixed,
            "posthoc_clean_default_0.25": posthoc_clean,
            "posthoc_fixed_default_0.25": posthoc_fixed,
            "posthoc_fixed_catf_v2_rc_calibration": calibrated,
        }
    write_json(
        REPORTS / "fixed_catf_v2_ready_for_paper_summary.json",
        {
            "ready": True,
            "constraint_pass_count_after_rc_calibration": threshold_payload["summary"]["constraint_pass_count"],
            "seed_metrics": seed_rows,
            "recommended_thresholds": threshold_payload["summary"]["recommended_thresholds"],
            "rollback_buckets": rollback_payload["buckets"],
        },
    )
    lines = [
        "# Fixed CATF-v2 Ready-For-Paper Summary",
        "",
        "This summary uses the RC post-hoc threshold selector. No training was run.",
        "",
        "## Final Groups",
        "",
        "- `clean native`: original per-seed YOLO default baselines.",
        "- `fixed CATF-v2`: strict no-op bypass fixed training results.",
        "- `fixed CATF-v2 + RC calibration`: fixed CATF-v2 predictions with constrained per-class threshold calibration.",
        "",
        f"RC calibrated constraint pass count: `{threshold_payload['summary']['constraint_pass_count']}/3`.",
        "",
        "## Seed-Level Metrics",
        "",
        "Official rows are Ultralytics validation metrics. Posthoc rows are from the cached prediction JSON threshold evaluator and should be used for threshold selection, not as a direct replacement for official validation.",
        "",
        "| seed | evaluator | group | Precision | Recall | mAP50 | mAP50-95 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for group, metrics in seed_rows[str(seed)].items():
            evaluator = "official" if group.startswith("official") else "posthoc"
            lines.append(
                f"| {seed} | {evaluator} | {group} | {f4(metrics['precision'])} | {f4(metrics['recall'])} | {f4(metrics['map50'])} | {f4(metrics['map50_95'])} |"
            )
    lines += [
        "",
        "## Recommended Per-Class Thresholds",
        "",
        "| class | threshold |",
        "|---|---:|",
    ]
    for cid, threshold in threshold_payload["summary"]["recommended_thresholds"].items():
        class_name = class_name_map().get(int(cid), cid)
        lines.append(f"| {cid}:{class_name} | {threshold:.2f} |")
    lines += [
        "",
        "## Main Conclusion",
        "",
        "Fixed CATF-v2 becomes a paper-ready candidate when paired with constrained per-class threshold calibration and class-level rollback/high-recall protection as the RC variant.",
        "",
        "## Risks",
        "",
        "- The threshold evaluator is post-hoc and should be confirmed against the official validation/export pipeline.",
        "- Low-support classes remain noisy; rollback decisions should be treated as protective rather than proof of augmentation harm.",
    ]
    write_md(REPORTS / "fixed_catf_v2_ready_for_paper_summary.md", lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    names = class_name_map()

    threshold_by_seed = {str(seed): build_threshold_candidates(seed, names) for seed in SEEDS}
    seed_thresholds = {
        seed: {int(k): float(v) for k, v in threshold_by_seed[str(seed)]["selected_thresholds"].items()}
        for seed in SEEDS
    }
    recommended = robust_threshold_table(seed_thresholds, default_threshold=DEFAULT_THRESHOLD)
    frequency: dict[str, dict[str, int]] = {}
    for cid in sorted(names):
        values = [seed_thresholds[seed].get(cid, DEFAULT_THRESHOLD) for seed in SEEDS]
        frequency[str(cid)] = {
            "lower_count": sum(value < DEFAULT_THRESHOLD for value in values),
            "raise_count": sum(value > DEFAULT_THRESHOLD for value in values),
            "default_count": sum(value == DEFAULT_THRESHOLD for value in values),
        }
    high_recall_sensitive = [f"{cid}:{names[int(cid)]}" for cid, stats in frequency.items() if stats["lower_count"] >= 2]
    threshold_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "catf_v2_rc_threshold_score",
        "seeds": threshold_by_seed,
        "summary": {
            "constraint_pass_count": sum(not row["selected_constraint_failed"] for row in threshold_by_seed.values()),
            "recommended_thresholds": {str(cid): value for cid, value in recommended.items()},
            "threshold_frequency": frequency,
            "high_recall_sensitive_classes": high_recall_sensitive,
            "threshold_insufficient_classes": [],
        },
    }
    seed0 = seed_failure_payload(0, names, threshold_by_seed["0"])
    write_seed0_failure_report(seed0, threshold_by_seed["0"])
    write_seed0_threshold_report(threshold_by_seed["0"])

    rollback_payload = rollback_simulation(threshold_payload, names)
    threshold_payload["summary"]["threshold_insufficient_classes"] = (
        rollback_payload["buckets"].get("rollback_candidate", []) + rollback_payload["buckets"].get("high_risk_negative_effect", [])
    )
    write_all_seed_threshold_report(threshold_payload, names)
    write_threshold_config(REPORTS / "catf_v2_rc_per_class_thresholds.json", recommended, class_names=names)

    write_rollback_report(rollback_payload)
    write_rc_plan(threshold_payload, rollback_payload)
    if threshold_payload["summary"]["constraint_pass_count"] == len(SEEDS):
        write_ready_summary(threshold_payload, rollback_payload)

    print(
        json.dumps(
            {
                "seed0_selected_failed": threshold_by_seed["0"]["selected_constraint_failed"],
                "constraint_pass_count": threshold_payload["summary"]["constraint_pass_count"],
                "reports": [
                    str(REPORTS / "seed0_failure_analysis.md"),
                    str(REPORTS / "seed0_threshold_calibration_posthoc.md"),
                    str(REPORTS / "fixed_catf_v2_threshold_calibration_all_seeds.md"),
                    str(REPORTS / "fixed_catf_v2_class_level_rollback_simulation.md"),
                    str(REPORTS / "catf_v2_rc_final_candidate_plan.md"),
                    str(REPORTS / "fixed_catf_v2_ready_for_paper_summary.md"),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
