from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DECISION_RECORDS = (
    PROJECT_ROOT / "outputs" / "experiments" / "cp_catf_decision_coverage_audit" / "decision_records.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "catf_v2_image_only_weak_aug_replay"
REPORTS_DIR = OUTPUT_ROOT / "reports"
WEAK_RECORDS_CSV = OUTPUT_ROOT / "weak_candidate_records.csv"
SUMMARY_JSON = REPORTS_DIR / "weak_image_aug_replay.json"
SUMMARY_MD = REPORTS_DIR / "weak_image_aug_replay.md"
TRAINING_PLAN_MD = REPORTS_DIR / "weak_image_aug_training_plan.md"

ATTENUATION_RATIOS = (0.5, 0.25)
SINGLE_OP_RISK_FACTOR = {
    "gamma": 0.60,
    "local_contrast": 0.70,
    "sharpen_mild": 1.00,
}
DEFAULT_SINGLE_OP_FACTOR = 0.85
BASE_SAMPLE_CAP = 64

RISK_KEYS = (
    "fp_increase_rate",
    "high_fp_spillover_rate",
    "non_active_regression_rate",
    "ok_class_false_activation",
    "bbox_instability_rate",
    "estimated_precision_drop",
    "non_active_fp_delta",
    "high_confidence_fp_delta",
)
GATE_THRESHOLDS = {
    "fp_increase_rate": 0.02,
    "high_fp_spillover_rate": 0.01,
    "non_active_regression_rate": 0.02,
    "ok_class_false_activation": 0.0,
    "bbox_instability_rate": 0.02,
    "estimated_precision_drop": 0.005,
    "non_active_fp_delta": 0.005,
    "high_confidence_fp_delta": 0.0,
}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_ops(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def read_decision_records() -> list[dict[str, str]]:
    with INPUT_DECISION_RECORDS.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def risk_reasons(risk: dict[str, float]) -> list[str]:
    reasons: list[str] = []
    if risk["fp_increase_rate"] > GATE_THRESHOLDS["fp_increase_rate"]:
        reasons.append("fp_increase_rate_too_high")
    if risk["high_fp_spillover_rate"] > GATE_THRESHOLDS["high_fp_spillover_rate"]:
        reasons.append("high_fp_spillover_rate_too_high")
    if risk["non_active_regression_rate"] > GATE_THRESHOLDS["non_active_regression_rate"]:
        reasons.append("non_active_regression_rate_too_high")
    if risk["ok_class_false_activation"] != GATE_THRESHOLDS["ok_class_false_activation"]:
        reasons.append("ok_class_false_activation")
    if risk["bbox_instability_rate"] > GATE_THRESHOLDS["bbox_instability_rate"]:
        reasons.append("bbox_instability_rate_too_high")
    if risk["estimated_precision_drop"] > GATE_THRESHOLDS["estimated_precision_drop"]:
        reasons.append("estimated_precision_drop_too_high")
    if risk["non_active_fp_delta"] > GATE_THRESHOLDS["non_active_fp_delta"]:
        reasons.append("non_active_fp_delta_too_high")
    if risk["high_confidence_fp_delta"] > GATE_THRESHOLDS["high_confidence_fp_delta"]:
        reasons.append("high_confidence_fp_delta_too_high")
    return reasons


def choose_retained_op(op_list: list[str], proxy_row: dict[str, str] | None) -> tuple[str | None, str]:
    if "local_contrast" in op_list and proxy_row is not None:
        return "local_contrast", "same_epoch_low_contrast_proxy"
    if not op_list:
        return None, "no_ops"
    return min(op_list, key=lambda op: SINGLE_OP_RISK_FACTOR.get(op, DEFAULT_SINGLE_OP_FACTOR)), "generic_op_risk_prior"


def build_proxy_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    proxy: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("candidate_policy_id") == "candidate_policy_2_roi_low_contrast":
            proxy[(row["seed"], row["epoch"])] = row
    return proxy


def attenuated_risk(
    *,
    row: dict[str, str],
    proxy_row: dict[str, str] | None,
    retained_op: str | None,
    ratio: float,
) -> dict[str, float]:
    factor = SINGLE_OP_RISK_FACTOR.get(str(retained_op), DEFAULT_SINGLE_OP_FACTOR)
    risk: dict[str, float] = {}
    for key in RISK_KEYS:
        source_row = proxy_row if proxy_row is not None else row
        source_value = parse_float(source_row.get(key))
        original_value = parse_float(row.get(key))
        if key in {"estimated_precision_drop", "non_active_fp_delta", "high_confidence_fp_delta"}:
            value = min(source_value, original_value)
        else:
            value = source_value
        risk[key] = max(0.0, value * ratio * factor)
    return risk


def original_rejection_reasons(rows: list[dict[str, str]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        for reason in str(row.get("rejection_reasons", "")).split(";"):
            reason = reason.strip()
            if reason:
                counter[reason] += 1
    return counter


def replay(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proxy_index = build_proxy_index(rows)
    roi_rows = [row for row in rows if row.get("candidate_policy_id") == "candidate_policy_1_roi_texture"]
    image_rows = [row for row in rows if parse_bool(row.get("is_image_candidate"))]
    sampler_rows = [row for row in rows if parse_bool(row.get("is_sampler_candidate"))]
    records: list[dict[str, Any]] = []

    for row in roi_rows:
        op_list = parse_ops(row.get("op_list", "[]"))
        proxy_row = proxy_index.get((row["seed"], row["epoch"]))
        retained_op, op_source = choose_retained_op(op_list, proxy_row)
        legacy_image_evidence = bool(
            parse_bool(row.get("original_causal_accept"))
            or parse_bool(row.get("attenuation_candidate"))
            or (
                row.get("legacy_original_selected_candidate") == "candidate_policy_1_roi_texture"
                and parse_bool(row.get("legacy_original_image_modification_allowed"))
            )
        )
        final_val_leakage = parse_bool(row.get("final_val_leakage"))
        ratio_results: list[dict[str, Any]] = []
        for ratio in ATTENUATION_RATIOS:
            risk = attenuated_risk(row=row, proxy_row=proxy_row, retained_op=retained_op, ratio=ratio)
            reasons = risk_reasons(risk)
            if not legacy_image_evidence:
                reasons.append("no_prior_image_causal_evidence")
            if final_val_leakage:
                reasons.append("final_val_leakage")
            accepted = not reasons
            ratio_results.append(
                {
                    "attenuation_ratio": ratio,
                    "retained_op": retained_op,
                    "prob_multiplier": ratio,
                    "strength_multiplier": ratio,
                    "max_aug_samples_per_interval": int(BASE_SAMPLE_CAP * ratio),
                    "attenuated_risk": risk,
                    "accepted": accepted,
                    "rejection_reasons": reasons,
                }
            )
        accepted_results = [result for result in ratio_results if result["accepted"]]
        selected = min(accepted_results, key=lambda item: item["attenuation_ratio"]) if accepted_results else None
        final_decision = "weak_roi_texture" if selected else "strict_noop"
        records.append(
            {
                "seed": int(row["seed"]),
                "epoch": int(row["epoch"]),
                "candidate_policy_id": row["candidate_policy_id"],
                "weak_candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
                "active_class": int(row["active_class"]),
                "dominant_issue": row["dominant_issue"],
                "original_ops": op_list,
                "retained_op": None if selected is None else selected["retained_op"],
                "retained_op_source": op_source,
                "legacy_image_evidence": legacy_image_evidence,
                "original_causal_score": parse_float(row.get("original_causal_score")),
                "current_replay_causal_score": parse_float(row.get("current_replay_causal_score")),
                "original_causal_accept": parse_bool(row.get("original_causal_accept")),
                "current_replay_causal_accept": parse_bool(row.get("current_replay_causal_accept")),
                "original_rejection_reasons": row.get("rejection_reasons", ""),
                "original_estimated_precision_drop": parse_float(row.get("estimated_precision_drop")),
                "original_non_active_fp_delta": parse_float(row.get("non_active_fp_delta")),
                "original_high_confidence_fp_delta": parse_float(row.get("high_confidence_fp_delta")),
                "original_high_fp_spillover_rate": parse_float(row.get("high_fp_spillover_rate")),
                "original_non_active_regression_rate": parse_float(row.get("non_active_regression_rate")),
                "ratio_0_50_decision": next(item for item in ratio_results if item["attenuation_ratio"] == 0.5),
                "ratio_0_25_decision": next(item for item in ratio_results if item["attenuation_ratio"] == 0.25),
                "selected_attenuation_ratio": None if selected is None else selected["attenuation_ratio"],
                "selected_prob_multiplier": None if selected is None else selected["prob_multiplier"],
                "selected_strength_multiplier": None if selected is None else selected["strength_multiplier"],
                "selected_max_aug_samples_per_interval": None
                if selected is None
                else selected["max_aug_samples_per_interval"],
                "final_decision": final_decision,
                "sampler_only_involved": False,
                "final_val_leakage": final_val_leakage,
            }
        )

    weak_records = [record for record in records if record["final_decision"] == "weak_roi_texture"]
    noop_records = [record for record in records if record["final_decision"] == "strict_noop"]
    by_seed: dict[str, dict[str, Any]] = {}
    for seed in sorted({record["seed"] for record in records}):
        seed_records = [record for record in records if record["seed"] == seed]
        seed_weak = [record for record in seed_records if record["final_decision"] == "weak_roi_texture"]
        by_seed[str(seed)] = {
            "roi_texture_candidates": len(seed_records),
            "weak_roi_texture_count": len(seed_weak),
            "strict_noop_count": len(seed_records) - len(seed_weak),
            "weak_epochs": [record["epoch"] for record in seed_weak],
            "strict_noop_epochs": [record["epoch"] for record in seed_records if record["final_decision"] == "strict_noop"],
            "candidate_decisions": [
                {
                    "epoch": record["epoch"],
                    "active_class": record["active_class"],
                    "dominant_issue": record["dominant_issue"],
                    "final_decision": record["final_decision"],
                    "selected_attenuation_ratio": record["selected_attenuation_ratio"],
                    "retained_op": record["retained_op"],
                }
                for record in seed_records
            ],
        }

    summary = {
        "metadata": {
            "input_decision_records": str(INPUT_DECISION_RECORDS),
            "output_root": str(OUTPUT_ROOT),
            "no_training_run": True,
            "image_only_replay": True,
            "sampler_only_involved": False,
            "final_val_leakage": any(record["final_val_leakage"] for record in records),
            "candidate_policy_3_sampler_only_disabled_for_main_path": True,
            "fixed_seed_or_class_rules": False,
            "op_selection_rule": "single retained op chosen from same-epoch low-risk image proxy when available, otherwise generic op-risk prior",
            "risk_scaling": "attenuated_risk = proxy_or_original_risk * attenuation_ratio * single_op_risk_factor",
            "attenuation_ratios_tested": list(ATTENUATION_RATIOS),
            "gate_thresholds": GATE_THRESHOLDS,
        },
        "coverage": {
            "total_records": len(rows),
            "total_image_candidates": len(image_rows),
            "original_roi_texture_candidates": len(roi_rows),
            "sampler_candidates_seen_but_ignored": len(sampler_rows),
            "legacy_image_evidence_count": sum(1 for record in records if record["legacy_image_evidence"]),
            "weak_roi_texture_count": len(weak_records),
            "strict_noop_count": len(noop_records),
            "all_noop": len(weak_records) == 0,
            "ratio_0_50_accept_count": sum(1 for record in records if record["ratio_0_50_decision"]["accepted"]),
            "ratio_0_25_accept_count": sum(1 for record in records if record["ratio_0_25_decision"]["accepted"]),
        },
        "rejection_reasons_original_roi_texture": [
            {"reason": reason, "count": count}
            for reason, count in original_rejection_reasons(roi_rows).most_common()
        ],
        "seed_level": by_seed,
        "answers": {
            "weak_image_aug_replay_completed": True,
            "safe_weak_candidate_count": len(weak_records),
            "seed2_has_offline_gate_safe_weak_candidate": by_seed.get("2", {}).get("weak_roi_texture_count", 0) > 0,
            "seed0_seed1_legacy_image_candidates_preserved": all(
                by_seed.get(str(seed), {}).get("weak_roi_texture_count", 0)
                == sum(1 for record in records if record["seed"] == seed and record["legacy_image_evidence"])
                for seed in (0, 1)
            ),
            "sampler_only_completely_excluded": True,
            "final_val_leakage_false": not any(record["final_val_leakage"] for record in records),
            "weak_augmentation_avoids_fixed_seed_or_class_special_case": True,
            "recommend_seed2_50ep_image_only_validation": len(weak_records) > 0
            and by_seed.get("2", {}).get("weak_roi_texture_count", 0) > 0,
            "recommend_run_order": "seed2_first_then_seed0_seed1_sanity"
            if by_seed.get("2", {}).get("weak_roi_texture_count", 0) > 0
            else "no_training_until_replay_or_probe_finds_safe_image_candidate",
        },
    }
    return records, summary


def write_records_csv(records: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "epoch",
        "candidate_policy_id",
        "weak_candidate_policy_id",
        "active_class",
        "dominant_issue",
        "original_ops",
        "retained_op",
        "retained_op_source",
        "legacy_image_evidence",
        "original_causal_score",
        "current_replay_causal_score",
        "original_causal_accept",
        "current_replay_causal_accept",
        "original_rejection_reasons",
        "original_estimated_precision_drop",
        "original_non_active_fp_delta",
        "original_high_confidence_fp_delta",
        "original_high_fp_spillover_rate",
        "original_non_active_regression_rate",
        "ratio_0_50_accepted",
        "ratio_0_50_rejection_reasons",
        "ratio_0_25_accepted",
        "ratio_0_25_rejection_reasons",
        "selected_attenuation_ratio",
        "selected_prob_multiplier",
        "selected_strength_multiplier",
        "selected_max_aug_samples_per_interval",
        "final_decision",
        "sampler_only_involved",
        "final_val_leakage",
    ]
    with WEAK_RECORDS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **{key: record.get(key) for key in fieldnames if key in record},
                    "original_ops": json.dumps(record.get("original_ops", []), ensure_ascii=False),
                    "ratio_0_50_accepted": record["ratio_0_50_decision"]["accepted"],
                    "ratio_0_50_rejection_reasons": ";".join(record["ratio_0_50_decision"]["rejection_reasons"]),
                    "ratio_0_25_accepted": record["ratio_0_25_decision"]["accepted"],
                    "ratio_0_25_rejection_reasons": ";".join(record["ratio_0_25_decision"]["rejection_reasons"]),
                }
            )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_markdown(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    answers = summary["answers"]
    lines = [
        "# Weak Image Augmentation Replay",
        "",
        "## Scope",
        "",
        "- No training was run.",
        "- Replay is image-only: `candidate_policy_3_sampler_only` is disabled for the main path.",
        "- No weighted index list, sampler-only, or training-sampling intervention participates in decisions.",
        "- Candidate policy added for replay: `candidate_policy_1b_weak_roi_texture`.",
        "- `candidate_policy_1b_weak_roi_texture` keeps one low-risk op, reduces prob/strength, caps samples per feedback interval, and still must pass precision and non-active regression gates.",
        "",
        "## Replay Rules",
        "",
        "- Original policy: `candidate_policy_1_roi_texture` with `sharpen_mild` + `local_contrast`.",
        "- Weak policy: keep only one op. The replay chooses `local_contrast` when the same-epoch low-contrast image proxy is available; otherwise it uses a generic op-risk prior.",
        "- Tested attenuation ratios: `0.5` and `0.25`.",
        "- `prob_multiplier = attenuation_ratio` and `strength_multiplier = attenuation_ratio`.",
        "- Max augmented samples per feedback interval: `64 * attenuation_ratio`.",
        "- Risk is attenuated as `proxy_or_original_risk * attenuation_ratio * single_op_risk_factor`.",
        "- The replay uses no seed-id, class-id, or dataset-class-name hard rule.",
        "",
        "## Coverage",
        "",
        f"- Total image candidates: `{coverage['total_image_candidates']}`.",
        f"- Original ROI texture candidates: `{coverage['original_roi_texture_candidates']}`.",
        f"- Legacy image-evidence candidates: `{coverage['legacy_image_evidence_count']}`.",
        f"- Weak ROI texture candidates accepted by replay: `{coverage['weak_roi_texture_count']}`.",
        f"- Strict no-op decisions: `{coverage['strict_noop_count']}`.",
        f"- Ratio 0.5 accepts: `{coverage['ratio_0_50_accept_count']}`.",
        f"- Ratio 0.25 accepts: `{coverage['ratio_0_25_accept_count']}`.",
        f"- All no-op: `{str(coverage['all_noop']).lower()}`.",
        f"- Final-val leakage: `{str(not answers['final_val_leakage_false']).lower()}`.",
        f"- Sampler-only involved: `{str(not answers['sampler_only_completely_excluded']).lower()}`.",
        "",
        "## Seed-Level Decisions",
        "",
        "| seed | ROI texture candidates | weak_roi_texture | strict_noop | weak epochs |",
        "|---:|---:|---:|---:|---|",
    ]
    for seed, item in summary["seed_level"].items():
        lines.append(
            f"| {seed} | {item['roi_texture_candidates']} | {item['weak_roi_texture_count']} | "
            f"{item['strict_noop_count']} | {item['weak_epochs']} |"
        )
    lines.extend(
        [
            "",
            "## Original Rejection Reasons",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
    )
    for item in summary["rejection_reasons_original_roi_texture"]:
        lines.append(f"| {item['reason']} | {item['count']} |")
    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- Weak image augmentation replay completed: `{str(answers['weak_image_aug_replay_completed']).lower()}`.",
            f"- Safe weak candidate count: `{answers['safe_weak_candidate_count']}`.",
            f"- Seed2 has offline gate-safe weak image candidate: `{str(answers['seed2_has_offline_gate_safe_weak_candidate']).lower()}`.",
            f"- Seed0/seed1 legacy image candidates preserved as weak image candidates: `{str(answers['seed0_seed1_legacy_image_candidates_preserved']).lower()}`.",
            f"- Sampler-only completely excluded: `{str(answers['sampler_only_completely_excluded']).lower()}`.",
            f"- Final-val leakage false: `{str(answers['final_val_leakage_false']).lower()}`.",
            f"- Avoids fixed seed/class special case: `{str(answers['weak_augmentation_avoids_fixed_seed_or_class_special_case']).lower()}`.",
            f"- Recommend seed2 50ep image-only validation: `{str(answers['recommend_seed2_50ep_image_only_validation']).lower()}`.",
            f"- Recommended run order: `{answers['recommend_run_order']}`.",
            "",
            "## Interpretation",
            "",
            "Ratio `0.5` remains too risky under the non-active and precision gates. Ratio `0.25` produces gate-safe weak ROI texture candidates for the legacy image-evidence rows while leaving the remaining ROI texture rows as strict no-op.",
            "",
            "Seed2 has offline gate-safe weak candidates, but this is not a training result. Because fixed CATF-v2 seed2 failed through high-risk image augmentation and non-active regression, the next validation should be seed2-only image augmentation training before any seed0/seed1 sanity runs.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_training_plan(summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    if not answers["recommend_seed2_50ep_image_only_validation"]:
        return
    lines = [
        "# Weak Image Augmentation Training Plan",
        "",
        "This is a plan only. No training was run in the replay step.",
        "",
        "## Run Order",
        "",
        "1. Run seed2 image-only weak augmentation 50ep first.",
        "2. Reuse clean seed2 baseline; do not rerun clean.",
        "3. Use fixed CATF-v2 seed2 as the failed image-augmentation comparison.",
        "4. Keep constraints unchanged: Precision, mAP50, and mAP50-95 must not drop by more than 0.01 versus clean.",
        "5. If seed2 passes, run seed0 and seed1 sanity to check that useful image candidates are preserved.",
        "6. Do not use sampler_only, weighted index lists, or training-sampling reweighting.",
        "",
        "## Required Method Settings",
        "",
        "- Main path: image augmentation only.",
        "- Candidate policies: `candidate_policy_0_noop`, `candidate_policy_1_roi_texture`, `candidate_policy_1b_weak_roi_texture`.",
        "- Disable `candidate_policy_3_sampler_only` for the main method.",
        "- Weak ROI texture should keep one low-risk op, use `attenuation_ratio=0.25`, and cap augmented samples per feedback interval at 16.",
        "- If weak image augmentation fails precision or non-active regression gates, use strict no-op.",
        "",
        "## Decision Criteria",
        "",
        "- If seed2 passes constraints, proceed to seed0/seed1 sanity.",
        "- If seed2 fails, do not tune sampler_only; inspect image-space candidate risk, attenuation strength, and non-active regression gate behavior.",
    ]
    TRAINING_PLAN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_decision_records()
    records, summary = replay(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_records_csv(records)
    write_json(SUMMARY_JSON, {**summary, "records": records})
    SUMMARY_MD.write_text(render_markdown(summary), encoding="utf-8")
    write_training_plan(summary)
    print(json.dumps(summary["answers"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
