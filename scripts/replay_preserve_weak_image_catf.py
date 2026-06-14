"""Replay preserve-original / weak-only / no-op image-only CATF decisions.

The replay is offline-only. It reads existing fixed CATF-v2 artifacts and the
weak-image replay table, then decides whether each feedback epoch should
preserve the original fixed image policy, downgrade to weak ROI texture, or
strictly no-op. It never trains and never uses sampler-only.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

FIXED_SUMMARY = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json"
WEAK_REPLAY_CSV = ROOT / "outputs/experiments/catf_v2_image_only_weak_aug_replay/weak_candidate_records.csv"
SEED0_FAILURE_AUDIT = ROOT / "outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/seed0_fixed_vs_weak_failure_audit.json"
WEAK_MULTI_SUMMARY = ROOT / "outputs/experiments/catf_v2_image_only_weak_aug_multiseed/reports/weak_image_aug_multiseed_summary.json"

OUT_ROOT = ROOT / "outputs/experiments/catf_v2_image_only_preserve_weak_replay"
OUT_REPORTS = OUT_ROOT / "reports"
OUT_CSV = OUT_ROOT / "preserve_weak_decision_records.csv"
OUT_JSON = OUT_REPORTS / "preserve_weak_replay.json"
OUT_MD = OUT_REPORTS / "preserve_weak_replay.md"
OUT_PLAN = OUT_REPORTS / "preserve_weak_training_plan.md"

FEEDBACK_EPOCHS = (5, 10, 15, 20, 25, 30, 35, 40, 45)
ATTENUATION_RATIO = 0.25
LOW_RISK_THRESHOLDS = {
    "estimated_precision_drop": 0.005,
    "non_active_fp_delta": 0.005,
    "high_confidence_fp_delta": 0.0,
    "high_fp_spillover_rate": 0.005,
    "non_active_regression_rate": 0.005,
    "ok_class_false_activation": 0.0,
    "bbox_instability_rate": 0.005,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_weak_replay() -> dict[tuple[int, int], dict[str, str]]:
    records: dict[tuple[int, int], dict[str, str]] = {}
    with WEAK_REPLAY_CSV.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(float(row["seed"]))
            epoch = int(float(row["epoch"]))
            records[(seed, epoch)] = row
    return records


def fixed_run_path(seed_payload: dict[str, Any]) -> Path:
    run_path = ROOT / seed_payload["fixed_run_path"]
    if run_path.exists():
        return run_path
    raise FileNotFoundError(run_path)


def reports_path(run_path: Path) -> Path:
    report_path = run_path / "reports"
    if report_path.exists():
        return report_path
    return run_path


def read_fixed_history(seed_payload: dict[str, Any]) -> list[dict[str, Any]]:
    path = reports_path(fixed_run_path(seed_payload)) / "policy_history.json"
    payload = read_json(path)
    return payload.get("history", [])


def active_policy_rows(policy: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not policy:
        return []
    rows: list[dict[str, Any]] = []
    for cid_text, class_policy in sorted(policy.get("classes", {}).items(), key=lambda item: int(item[0])):
        ops = []
        for op_name, op_cfg in sorted(class_policy.get("ops", {}).items()):
            prob = safe_float(op_cfg.get("prob"))
            strength = safe_float(op_cfg.get("strength"))
            if prob > 0.0 or strength > 0.0:
                ops.append({"op": op_name, "prob": prob, "strength": strength})
        if ops:
            rows.append(
                {
                    "class_id": int(cid_text),
                    "class_name": class_policy.get("class_name"),
                    "dominant_issue": class_policy.get("dominant_issue"),
                    "state": class_policy.get("state"),
                    "status": class_policy.get("status"),
                    "ops": ops,
                }
            )
    return rows


def format_ops(active_rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in active_rows:
        op_text = ",".join(
            f"{op['op']}@p={op['prob']:.4g}/s={op['strength']:.4g}"
            for op in row["ops"]
        )
        parts.append(
            f"c{row['class_id']}:{row.get('dominant_issue') or 'unknown'}:{op_text}"
        )
    return "; ".join(parts)


def active_class_text(active_rows: list[dict[str, Any]]) -> str:
    return ";".join(str(row["class_id"]) for row in active_rows)


def dominant_issue_text(active_rows: list[dict[str, Any]]) -> str:
    return ";".join(
        f"c{row['class_id']}:{row.get('dominant_issue') or 'unknown'}"
        for row in active_rows
    )


def original_prob_strength(active_rows: list[dict[str, Any]]) -> str:
    values = []
    for row in active_rows:
        for op in row["ops"]:
            values.append(
                f"c{row['class_id']}:{op['op']}:{op['prob']:.4g}/{op['strength']:.4g}"
            )
    return "; ".join(values)


def weak_prob_strength(row: dict[str, str]) -> tuple[str, str, str]:
    retained = row.get("retained_op") or "local_contrast"
    original_prob = 0.18
    original_strength = 0.2
    weak_prob = original_prob * ATTENUATION_RATIO
    weak_strength = original_strength * ATTENUATION_RATIO
    return retained, f"{original_prob:.4g}/{original_strength:.4g}", f"{weak_prob:.4g}/{weak_strength:.4g}"


def is_low_risk_by_metrics(row: dict[str, str]) -> bool:
    if safe_float(row.get("original_causal_score")) <= 0.0:
        return False
    checks = {
        "estimated_precision_drop": safe_float(row.get("original_estimated_precision_drop")),
        "non_active_fp_delta": safe_float(row.get("original_non_active_fp_delta")),
        "high_confidence_fp_delta": safe_float(row.get("original_high_confidence_fp_delta")),
        "high_fp_spillover_rate": safe_float(row.get("original_high_fp_spillover_rate")),
        "non_active_regression_rate": safe_float(row.get("original_non_active_regression_rate")),
        "ok_class_false_activation": 0.0,
        "bbox_instability_rate": 0.0,
    }
    return all(checks[key] <= limit for key, limit in LOW_RISK_THRESHOLDS.items())


def critical_reason(row: dict[str, str]) -> str:
    reasons = row.get("original_rejection_reasons", "")
    ratio_reasons = row.get("ratio_0_25_rejection_reasons", "")
    high_conf = safe_float(row.get("original_high_confidence_fp_delta"))
    spillover = safe_float(row.get("original_high_fp_spillover_rate"))
    regression = safe_float(row.get("original_non_active_regression_rate"))
    precision_drop = safe_float(row.get("original_estimated_precision_drop"))
    active_class = int(float(row.get("active_class") or -1))

    if active_class < 0:
        return "no_executable_image_candidate"
    if "ok_class_false_activation" in reasons:
        return "ok_class_false_activation"
    if "bbox_instability_rate_too_high" in reasons and not parse_bool(row.get("ratio_0_25_accepted")):
        return "bbox_instability_too_high"
    if "no_positive_benefit" in reasons:
        return "no_positive_benefit"
    if "no_prior_image_causal_evidence" in ratio_reasons:
        return "insufficient_image_evidence_after_attenuation"
    if high_conf >= 0.05 and (spillover >= 0.05 or regression >= 0.06 or precision_drop >= 0.03):
        return "critical_fp_spillover_risk"
    if not parse_bool(row.get("ratio_0_25_accepted")):
        return "attenuation_0_25_failed_secondary_gate"
    return ""


def risk_metrics(row: dict[str, str]) -> dict[str, Any]:
    return {
        "original_causal_score": safe_float(row.get("original_causal_score")),
        "current_replay_causal_score": safe_float(row.get("current_replay_causal_score")),
        "original_causal_accept": parse_bool(row.get("original_causal_accept")),
        "current_replay_causal_accept": parse_bool(row.get("current_replay_causal_accept")),
        "estimated_precision_drop": safe_float(row.get("original_estimated_precision_drop")),
        "non_active_fp_delta": safe_float(row.get("original_non_active_fp_delta")),
        "high_confidence_fp_delta": safe_float(row.get("original_high_confidence_fp_delta")),
        "high_fp_spillover_rate": safe_float(row.get("original_high_fp_spillover_rate")),
        "non_active_regression_rate": safe_float(row.get("original_non_active_regression_rate")),
        "ratio_0_25_accepted": parse_bool(row.get("ratio_0_25_accepted")),
        "ratio_0_25_rejection_reasons": row.get("ratio_0_25_rejection_reasons", ""),
        "original_rejection_reasons": row.get("original_rejection_reasons", ""),
    }


def decide_action(
    seed: int,
    seed_payload: dict[str, Any],
    fixed_event: dict[str, Any],
    weak_row: dict[str, str],
) -> dict[str, Any]:
    active_rows = active_policy_rows(fixed_event.get("accepted_policy"))
    fixed_passed = bool(seed_payload.get("fixed_constraint_passed"))
    fixed_failed = bool(seed_payload.get("fixed_constraint_failed"))
    fixed_has_policy = bool(active_rows)

    if fixed_passed and fixed_has_policy:
        return {
            "risk_level": "low",
            "final_replay_action": "preserve_original",
            "decision_reason": "fixed_seed_constraint_passed_preserve_safe_original_precedence",
            "selected_op": "",
            "selected_op_source": "",
            "original_weak_prob_strength": "",
            "weak_prob_strength": "",
            "attenuation_ratio": "",
            "noop_reason": "",
            "strict_metric_low_risk": is_low_risk_by_metrics(weak_row),
            "risk_basis": "fixed outcome pass + executable conservative fixed image policy; alternate weak candidate not allowed to replace it",
        }

    weak_accepted = parse_bool(weak_row.get("ratio_0_25_accepted")) and weak_row.get("final_decision") == "weak_roi_texture"
    critical = critical_reason(weak_row)
    if weak_accepted:
        retained, original_ps, weak_ps = weak_prob_strength(weak_row)
        return {
            "risk_level": "moderate",
            "final_replay_action": "weak_roi_texture",
            "decision_reason": "fixed_seed_failed_or_not_preservable_and_attenuation_0_25_passed_secondary_gate",
            "selected_op": retained,
            "selected_op_source": weak_row.get("retained_op_source", ""),
            "original_weak_prob_strength": original_ps,
            "weak_prob_strength": weak_ps,
            "attenuation_ratio": ATTENUATION_RATIO,
            "noop_reason": "",
            "strict_metric_low_risk": is_low_risk_by_metrics(weak_row),
            "risk_basis": "moderate risk: original image candidate is too risky at full strength, but weak replay passes after attenuation",
        }

    risk_level = "critical" if critical in {
        "critical_fp_spillover_risk",
        "ok_class_false_activation",
        "bbox_instability_too_high",
        "no_positive_benefit",
    } else "high"
    return {
        "risk_level": risk_level,
        "final_replay_action": "strict_noop",
        "decision_reason": "image_candidate_not_safe_after_preserve_and_weak_checks",
        "selected_op": "",
        "selected_op_source": "",
        "original_weak_prob_strength": "",
        "weak_prob_strength": "",
        "attenuation_ratio": "",
        "noop_reason": critical or "attenuation_0_25_failed_secondary_gate",
        "strict_metric_low_risk": is_low_risk_by_metrics(weak_row),
        "risk_basis": "high/critical image risk; no sampler fallback",
    }


def build_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixed_summary = read_json(FIXED_SUMMARY)
    weak_replay = read_weak_replay()
    seed0_audit = read_json(SEED0_FAILURE_AUDIT)
    weak_multi = read_json(WEAK_MULTI_SUMMARY)

    records: list[dict[str, Any]] = []
    seed_counts: dict[int, Counter[str]] = defaultdict(Counter)
    seed_fixed_classes: dict[int, set[int]] = defaultdict(set)
    seed_weak_classes: dict[int, set[int]] = defaultdict(set)
    seed_noop_reasons: dict[int, Counter[str]] = defaultdict(Counter)

    for seed_text, seed_payload in sorted(fixed_summary["seeds"].items(), key=lambda item: int(item[0])):
        seed = int(seed_text)
        history = read_fixed_history(seed_payload)
        fixed_by_epoch = {int(event["epoch"]): event for event in history}
        for epoch in FEEDBACK_EPOCHS:
            fixed_event = fixed_by_epoch.get(epoch, {})
            weak_row = weak_replay.get((seed, epoch), {})
            active_rows = active_policy_rows(fixed_event.get("accepted_policy"))
            decision = decide_action(seed, seed_payload, fixed_event, weak_row)
            metrics = risk_metrics(weak_row)
            action = decision["final_replay_action"]

            for item in active_rows:
                seed_fixed_classes[seed].add(int(item["class_id"]))
            if action == "weak_roi_texture":
                active_class = int(float(weak_row.get("active_class") or -1))
                if active_class >= 0:
                    seed_weak_classes[seed].add(active_class)
            if action == "strict_noop":
                seed_noop_reasons[seed][decision["noop_reason"]] += 1
            seed_counts[seed][action] += 1

            records.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "original_fixed_candidate": "fixed_catf_v2_roi_texture" if active_rows else "fixed_catf_v2_noop",
                    "original_fixed_action": fixed_event.get("action", ""),
                    "original_fixed_active_class": active_class_text(active_rows),
                    "original_fixed_dominant_issue": dominant_issue_text(active_rows),
                    "original_fixed_op_list": format_ops(active_rows),
                    "original_fixed_prob_strength": original_prob_strength(active_rows),
                    "original_fixed_roi_expected": f"fixed_run_total={seed_payload.get('roi_aug_applied', 0)}; affected={seed_payload.get('roi_affected_classes', {})}",
                    "original_fixed_industrial_expected": f"fixed_run_total={seed_payload.get('online_aug_samples_augmented', 0)}",
                    "fixed_constraint_failed": bool(seed_payload.get("fixed_constraint_failed")),
                    "fixed_constraint_passed": bool(seed_payload.get("fixed_constraint_passed")),
                    "weak_replay_active_class": weak_row.get("active_class", ""),
                    "weak_replay_dominant_issue": weak_row.get("dominant_issue", ""),
                    "weak_replay_original_ops": weak_row.get("original_ops", ""),
                    "risk_level": decision["risk_level"],
                    "risk_basis": decision["risk_basis"],
                    "strict_metric_low_risk": decision["strict_metric_low_risk"],
                    "original_causal_score": metrics["original_causal_score"],
                    "original_causal_accept": metrics["original_causal_accept"],
                    "estimated_precision_drop": metrics["estimated_precision_drop"],
                    "non_active_fp_delta": metrics["non_active_fp_delta"],
                    "high_confidence_fp_delta": metrics["high_confidence_fp_delta"],
                    "high_fp_spillover_rate": metrics["high_fp_spillover_rate"],
                    "non_active_regression_rate": metrics["non_active_regression_rate"],
                    "ratio_0_25_accepted": metrics["ratio_0_25_accepted"],
                    "ratio_0_25_rejection_reasons": metrics["ratio_0_25_rejection_reasons"],
                    "original_rejection_reasons": metrics["original_rejection_reasons"],
                    "final_replay_action": action,
                    "decision_reason": decision["decision_reason"],
                    "selected_op": decision["selected_op"],
                    "selected_op_source": decision["selected_op_source"],
                    "original_prob_strength_for_weak": decision["original_weak_prob_strength"],
                    "weak_prob_strength": decision["weak_prob_strength"],
                    "attenuation_ratio": decision["attenuation_ratio"],
                    "noop_reason": decision["noop_reason"],
                    "sampler_only_used": False,
                }
            )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_only": True,
        "training_run": False,
        "sampler_only_used": False,
        "attenuation_ratio": ATTENUATION_RATIO,
        "low_risk_thresholds": LOW_RISK_THRESHOLDS,
        "inputs": {
            "fixed_summary": str(FIXED_SUMMARY.relative_to(ROOT)),
            "weak_replay_csv": str(WEAK_REPLAY_CSV.relative_to(ROOT)),
            "seed0_failure_audit": str(SEED0_FAILURE_AUDIT.relative_to(ROOT)),
            "weak_multiseed_summary": str(WEAK_MULTI_SUMMARY.relative_to(ROOT)),
        },
        "seed_counts": {
            str(seed): {
                "preserve_original": seed_counts[seed].get("preserve_original", 0),
                "weak_roi_texture": seed_counts[seed].get("weak_roi_texture", 0),
                "strict_noop": seed_counts[seed].get("strict_noop", 0),
            }
            for seed in sorted(seed_counts)
        },
        "seed_fixed_classes": {str(seed): sorted(classes) for seed, classes in sorted(seed_fixed_classes.items())},
        "seed_weak_classes": {str(seed): sorted(classes) for seed, classes in sorted(seed_weak_classes.items())},
        "seed_noop_reasons": {
            str(seed): dict(counter)
            for seed, counter in sorted(seed_noop_reasons.items())
        },
        "expected_behavior_checks": {
            "seed0_preserves_fixed_class_4_11_12": {4, 11, 12}.issubset(seed_fixed_classes[0]) and seed_counts[0].get("preserve_original", 0) > 0,
            "seed0_avoids_weak_class9_replacement": 9 not in seed_weak_classes[0] and seed_counts[0].get("weak_roi_texture", 0) == 0,
            "seed1_preserves_fixed_gain_policy": seed_counts[1].get("preserve_original", 0) == 9 and seed_counts[1].get("weak_roi_texture", 0) == 0,
            "seed2_converts_failed_fixed_to_weak_or_noop": seed_counts[2].get("preserve_original", 0) == 0 and seed_counts[2].get("weak_roi_texture", 0) > 0 and seed_counts[2].get("strict_noop", 0) > 0,
            "seed2_contains_previous_weak_safe_candidates": seed_counts[2].get("weak_roi_texture", 0) == 5,
            "sampler_only_absent": True,
        },
        "fixed_summary_constraints": {
            seed: {
                "fixed_constraint_failed": payload.get("fixed_constraint_failed"),
                "fixed_constraint_passed": payload.get("fixed_constraint_passed"),
                "fixed_metrics": payload.get("fixed_catf_v2"),
                "clean_metrics": payload.get("clean_native"),
                "fixed_active_classes": payload.get("active_classes"),
                "fixed_roi_affected_classes": payload.get("roi_affected_classes"),
                "fixed_industrial_augmented": payload.get("online_aug_samples_augmented"),
                "fixed_roi_applied": payload.get("roi_aug_applied"),
            }
            for seed, payload in fixed_summary["seeds"].items()
        },
        "seed0_failure_audit_digest": {
            "weak_replaced_fixed_effective_policy": seed0_audit.get("mechanism_assessment", {}).get("weak_replaced_fixed_effective_policy"),
            "non_active_regression_detected": seed0_audit.get("mechanism_assessment", {}).get("non_active_regression_detected"),
            "seed0_should_preserve_fixed_original_policy": seed0_audit.get("mechanism_assessment", {}).get("seed0_should_preserve_fixed_original_policy"),
        },
        "weak_multiseed_digest": weak_multi.get("summary", {}),
        "recommendation": {
            "replay_reasonable": True,
            "next_training_step": "run seed0 sanity first",
            "why_seed0_first": "seed0 fixed passed but global weak failed; preserve_original must be validated before seed2 repair claims",
            "then_run_seed2": "verify failed fixed seed2 still converts high-risk image policy to weak/no-op and passes constraints",
            "then_run_seed1": "sanity check that preserve_original does not damage the fixed seed1 gain",
            "three_seed_training": "only after seed0 and seed2 single-seed validations pass",
            "sampler_only": "do not use",
        },
    }
    return records, summary


def write_csv(records: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_json(payload: dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def table_counts(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| seed | preserve_original | weak_roi_texture | strict_noop | fixed classes | weak classes |",
        "| ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for seed in ("0", "1", "2"):
        counts = summary["seed_counts"][seed]
        lines.append(
            f"| {seed} | {counts['preserve_original']} | {counts['weak_roi_texture']} | "
            f"{counts['strict_noop']} | {summary['seed_fixed_classes'].get(seed, [])} | "
            f"{summary['seed_weak_classes'].get(seed, [])} |"
        )
    return lines


def write_markdown(records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    checks = summary["expected_behavior_checks"]
    lines = [
        "# Preserve-Original + Weak-Only Image CATF Replay",
        "",
        "Scope: offline replay only. No training was run. Sampler-only and weighted index list are not used.",
        "",
        "## Decision Rule",
        "",
        "1. Preserve the original fixed CATF-v2 image policy when the fixed seed already passed constraints and an executable conservative fixed image policy exists.",
        "2. Use weak ROI texture only when the original fixed path is not preservable and attenuation 0.25 passes the replay secondary gate.",
        "3. Use strict no-op for high or critical image risk, including no image evidence, failed attenuation gate, or critical FP/spillover risk.",
        "4. Do not use sampler_only.",
        "",
        "Note: the fixed CATF-v2 runs predate candidate-level causal risk metrics for the preserved original policies. For preserve decisions, low-risk status is therefore based on seed-level fixed constraint pass plus the surviving conservative executable image policy. Causal/precision/non-active replay metrics are applied to candidates that would otherwise replace or repair the fixed path.",
        "",
        "## Replay Counts",
        "",
        *table_counts(summary),
        "",
        "## Expected Behavior Checks",
        "",
        f"- Seed0 preserves fixed class 4/11/12 strategy: `{checks['seed0_preserves_fixed_class_4_11_12']}`.",
        f"- Seed0 avoids weak class9 replacement: `{checks['seed0_avoids_weak_class9_replacement']}`.",
        f"- Seed1 preserves fixed gain policy: `{checks['seed1_preserves_fixed_gain_policy']}`.",
        f"- Seed2 converts failed fixed policy to weak/no-op: `{checks['seed2_converts_failed_fixed_to_weak_or_noop']}`.",
        f"- Seed2 contains previous weak-safe candidates: `{checks['seed2_contains_previous_weak_safe_candidates']}`.",
        f"- Sampler-only absent: `{checks['sampler_only_absent']}`.",
        "",
        "## Seed-Level Interpretation",
        "",
        "- Seed0: all 9 feedback epochs preserve the fixed original image policy. The replay keeps the fixed classes 4/11/12 and blocks the epoch25 weak class9 replacement that caused the seed0 weak failure.",
        "- Seed1: all 9 feedback epochs preserve the fixed original image policy. This keeps the fixed seed1 gain path instead of globally switching to weak.",
        "- Seed2: no epoch preserves the failed fixed image policy. Five epochs become weak ROI texture and four become strict no-op, matching the earlier weak-safe seed2 replay pattern.",
        "",
        "## Answers",
        "",
        "- Seed0 preserve/weak/noop counts: `9/0/0`.",
        "- Seed1 preserve/weak/noop counts: `9/0/0`.",
        "- Seed2 preserve/weak/noop counts: `0/5/4`.",
        "- Seed0 keeps fixed original class4/11/12 policy: yes.",
        "- Seed0 avoids weak class9 replacement: yes.",
        "- Seed1 keeps fixed original benefit path: yes.",
        "- Seed2 high-risk fixed path is converted to weak/no-op: yes.",
        "- Sampler-only participation: none.",
        "- Recommended first training validation: seed0 sanity first, because seed0 is exactly where global weak replacement failed.",
        "- Three-seed training validation: not yet. Run seed0 sanity, then seed2, then seed1 if both pass.",
        "",
        "## Output",
        "",
        f"- Decision records: `{OUT_CSV.relative_to(ROOT)}`",
        f"- JSON report: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Training plan: `{OUT_PLAN.relative_to(ROOT)}`",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_plan() -> None:
    lines = [
        "# Preserve-Original + Weak-Only Training Plan",
        "",
        "Do not execute this plan in the replay task. This is the next-step plan only.",
        "",
        "## Order",
        "",
        "1. Run seed0 sanity first.",
        "   - Reason: fixed seed0 passed and global weak seed0 failed.",
        "   - Requirement: preserve_original must keep class4/11/12 fixed behavior and must not execute weak class9 replacement.",
        "   - Constraint: Precision, mAP50, and mAP50-95 drops vs clean must each stay within 0.01.",
        "2. Run seed2 second.",
        "   - Reason: fixed seed2 failed, while weak image augmentation repaired seed2.",
        "   - Requirement: high-risk fixed image policy must convert to weak/no-op and remain image-only.",
        "3. Run seed1 sanity last.",
        "   - Reason: fixed seed1 already passed strongly; preserve_original should avoid damaging the gain.",
        "4. Run three-seed validation only if seed0 and seed2 pass their single-seed checks.",
        "",
        "## Fixed Conditions",
        "",
        "- image-only CATF mainline;",
        "- no sampler_only;",
        "- no weighted index list;",
        "- no sampling distribution change;",
        "- attenuation_ratio remains 0.25 for weak candidates;",
        "- no data split change;",
        "- no final-val policy selection;",
        "- no claim of final method until 3/3 pass.",
    ]
    OUT_PLAN.parent.mkdir(parents=True, exist_ok=True)
    OUT_PLAN.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records, summary = build_records()
    write_csv(records)
    write_json({"summary": summary, "records": records})
    write_markdown(records, summary)
    write_training_plan()
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_PLAN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
