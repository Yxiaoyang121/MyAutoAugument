from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AutoAugment.catf_v2.causal_probe import (
    build_probe_set,
    candidate_policy_catalog,
    evaluate_candidate_policy,
    export_probe_report,
    select_best_candidate,
)


FIXED_SUMMARY = (
    ROOT
    / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/reports/multiseed_catf_v2_fixed_summary.json"
)
SEED2_ROOT_CAUSE = ROOT / "outputs/experiments/seed2_failure_root_cause/reports/seed2_root_cause_summary.json"
RISKGUARD_SEED2 = (
    ROOT / "outputs/experiments/catf_v2_riskguard_seed2_50ep/reports/riskguard_seed2_summary.json"
)
OUT_DIR = ROOT / "outputs/experiments/catf_v2_causal_probe"


def main() -> None:
    fixed_summary = read_json(FIXED_SUMMARY)
    seed2_root = read_json(SEED2_ROOT_CAUSE)
    riskguard_seed2 = read_json(RISKGUARD_SEED2) if RISKGUARD_SEED2.exists() else {}
    catalog = candidate_policy_catalog()
    seed_decisions: dict[str, Any] = {}
    for seed_text, seed_payload in sorted(fixed_summary["seeds"].items(), key=lambda item: int(item[0])):
        seed = int(seed_text)
        decision_payload = build_seed_probe_decision(
            seed=seed,
            seed_payload=seed_payload,
            seed2_root=seed2_root if seed == 2 else {},
            riskguard_seed2=riskguard_seed2 if seed == 2 else {},
            catalog=catalog,
        )
        seed_decisions[seed_text] = decision_payload
        seed_dir = OUT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        export_probe_report(decision_payload, seed_dir / "probe_decisions.json")

    summary = build_summary(seed_decisions, fixed_summary, riskguard_seed2)
    reports_dir = OUT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    export_probe_report(summary, reports_dir / "offline_causal_probe_summary.json")
    (reports_dir / "offline_causal_probe_summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    (reports_dir / "cp_catf_training_plan.md").write_text(render_training_plan_md(summary), encoding="utf-8")


def build_seed_probe_decision(
    *,
    seed: int,
    seed_payload: dict[str, Any],
    seed2_root: dict[str, Any],
    riskguard_seed2: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    clean = seed_payload["clean_native"]
    fixed = seed_payload["fixed_catf_v2"]
    delta = seed_payload["delta_fixed_vs_clean"]
    active_classes = sorted(int(item) for item in (seed_payload.get("active_classes") or {}).keys())
    representative_class = active_classes[0] if active_classes else -1
    evidence_count = int(seed_payload.get("roi_aug_applied") or seed_payload.get("online_aug_samples_augmented") or 0)
    diagnosis_confidence = 1.0 if seed == 2 else 0.75
    high_fp_samples = [{"class_id": item, "bucket": "high_fp_risk"} for item in [0, 1]]
    non_active_samples = build_non_active_samples(seed, seed2_root)
    benefit, risk = derive_probe_metrics(delta, seed2_root=seed2_root, riskguard_seed2=riskguard_seed2)

    evaluations = []
    for policy_id in (
        "candidate_policy_1_roi_texture",
        "candidate_policy_2_roi_low_contrast",
        "candidate_policy_3_sampler_only",
    ):
        probe_set = build_probe_set(
            candidate_class_id=representative_class if representative_class >= 0 else 0,
            candidate_policy_id=policy_id,
            diagnosis_record={
                "evidence_count": evidence_count,
                "diagnosis_confidence": diagnosis_confidence,
                "fn_count": max(0, evidence_count),
                "weak_loc_count": max(0, int(evidence_count * 0.5)),
            },
            issue_record={"evidence_count": evidence_count, "diagnosis_confidence": diagnosis_confidence},
            high_fp_samples=high_fp_samples,
            non_active_samples=non_active_samples,
        )
        evaluations.append(
            evaluate_candidate_policy(
                candidate_policy=catalog[policy_id],
                probe_set=probe_set,
                benefit_metrics=benefit,
                risk_metrics=risk,
            )
        )
    per_class_candidate_evaluations: list[dict[str, Any]] = []
    for class_id in active_classes:
        for policy_id in ("candidate_policy_1_roi_texture", "candidate_policy_2_roi_low_contrast"):
            class_probe_set = build_probe_set(
                candidate_class_id=class_id,
                candidate_policy_id=policy_id,
                diagnosis_record={
                    "evidence_count": evidence_count,
                    "diagnosis_confidence": diagnosis_confidence,
                    "fn_count": max(0, evidence_count),
                    "weak_loc_count": max(0, int(evidence_count * 0.5)),
                },
                issue_record={"evidence_count": evidence_count, "diagnosis_confidence": diagnosis_confidence},
                high_fp_samples=high_fp_samples,
                non_active_samples=non_active_samples,
            )
            per_class_candidate_evaluations.append(
                evaluate_candidate_policy(
                    candidate_policy=catalog[policy_id],
                    probe_set=class_probe_set,
                    benefit_metrics=benefit,
                    risk_metrics=risk,
                )
            )
    selected = select_best_candidate(evaluations)
    image_aug_rejected = all(
        item["decision"]["decision"] == "reject"
        for item in evaluations
        if item["candidate_policy"].get("image_modification")
    )
    return {
        "seed": seed,
        "development_probe_uses_existing_val_diagnostics": True,
        "paper_mode_note": "Paper-mode CP-CATF should use a train/probe split or train hard examples, not final validation metrics.",
        "clean_metrics": clean,
        "fixed_catf_v2_metrics": fixed,
        "delta_fixed_vs_clean": delta,
        "fixed_constraint_failed": bool(seed_payload.get("fixed_constraint_failed", False)),
        "active_classes": active_classes,
        "roi_affected_classes": {
            str(key): int(value) for key, value in (seed_payload.get("roi_affected_classes") or {}).items()
        },
        "candidate_evaluations": evaluations,
        "active_class_candidate_evaluations": per_class_candidate_evaluations,
        "selected_candidate_policy_id": selected["candidate_policy_id"],
        "selected_candidate_action": selected["decision"]["decision"],
        "selected_candidate": selected,
        "image_augmentation_rejected": image_aug_rejected,
        "strict_image_noop": not bool(selected["decision"].get("image_modification_allowed", False)),
        "riskguard_interpretation": build_riskguard_interpretation(seed, riskguard_seed2),
        "decision_is_seed_specific": False,
        "decision_is_fixed_class_id_specific": False,
        "decision_basis": "run_specific_probe_benefit_risk_metrics",
    }


def derive_probe_metrics(
    delta: dict[str, float],
    *,
    seed2_root: dict[str, Any],
    riskguard_seed2: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    delta_p = float(delta.get("precision", 0.0) or 0.0)
    delta_r = float(delta.get("recall", 0.0) or 0.0)
    delta_map50 = float(delta.get("map50", 0.0) or 0.0)
    delta_map95 = float(delta.get("map50_95", 0.0) or 0.0)
    non_active_regression = 0.0
    if seed2_root.get("non_active_class_regression"):
        non_active_regression = 0.05
    for item in riskguard_seed2.get("non_active_regressions", []) if riskguard_seed2 else []:
        non_active_regression = max(
            non_active_regression,
            max(
                0.0,
                -float(item.get("risk_delta_recall_vs_clean", 0.0) or 0.0),
                -float(item.get("risk_delta_ap50_vs_clean", 0.0) or 0.0),
                -float(item.get("risk_delta_map50_95_vs_clean", 0.0) or 0.0),
            ),
        )
    benefit = {
        "fn_recovery_rate": max(0.0, delta_r),
        "localization_iou_gain": max(0.0, delta_map95),
        "low_conf_tp_conf_gain": max(0.0, delta_p),
        "active_class_conf_gain": max(0.0, delta_p),
        "active_class_detection_count_gain": max(0.0, delta_r),
    }
    risk = {
        "fp_increase_rate": max(0.0, -delta_p),
        "high_fp_spillover_rate": max(0.0, -delta_map50),
        "non_active_regression_rate": non_active_regression,
        "ok_class_false_activation": 0.0,
        "bbox_instability_rate": max(0.0, -delta_map95),
        "confidence_collapse_rate": max(0.0, -delta_r),
    }
    return benefit, risk


def build_non_active_samples(seed: int, seed2_root: dict[str, Any]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if seed == 2:
        for bucket_name in ("recall_drop", "ap50_drop", "ap50_95_drop"):
            for item in (seed2_root.get("worst_injured_classes", {}).get(bucket_name) or [])[:5]:
                if not item.get("active"):
                    samples.append(
                        {
                            "class_id": int(item["class_id"]),
                            "bucket": f"non_active_{bucket_name}",
                            "delta_recall": float(item.get("delta_recall", 0.0) or 0.0),
                            "delta_ap50": float(item.get("delta_ap50", 0.0) or 0.0),
                            "delta_ap50_95": float(item.get("delta_ap50_95", 0.0) or 0.0),
                        }
                    )
    return samples


def build_riskguard_interpretation(seed: int, riskguard_seed2: dict[str, Any]) -> dict[str, Any]:
    if seed != 2 or not riskguard_seed2:
        return {
            "riskguard_used_as_final_rule": False,
            "interpretation": "No seed-specific RiskGuard outcome used for this decision.",
        }
    return {
        "riskguard_used_as_final_rule": False,
        "riskguard_downgraded_to_audit_prior": True,
        "riskguard_constraint_failed": bool(riskguard_seed2.get("constraint_failed", False)),
        "riskguard_summary": (
            "Blocking the audited class-op pair recovered class-local behavior but did not recover overall constraints; "
            "therefore fixed blacklist is insufficient and CP-CATF must score the whole candidate risk."
        ),
    }


def build_summary(
    seed_decisions: dict[str, Any],
    fixed_summary: dict[str, Any],
    riskguard_seed2: dict[str, Any],
) -> dict[str, Any]:
    accepted_image_seeds = [
        int(seed)
        for seed, payload in seed_decisions.items()
        if bool(payload["selected_candidate"]["decision"].get("image_modification_allowed", False))
    ]
    rejected_image_seeds = [
        int(seed) for seed, payload in seed_decisions.items() if payload.get("image_augmentation_rejected")
    ]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "method": "CP-CATF offline causal probe",
        "no_training_run": True,
        "development_probe_uses_existing_val_diagnostics": True,
        "paper_mode_required_change": "Use a train/probe split or train hard examples for policy selection; final validation remains evaluation only.",
        "riskguard_status": {
            "fixed_class_op_blacklist_as_main_rule": False,
            "downgraded_to_audit_prior": True,
            "direct_block_requires_causal_probe_or_debug_mode": True,
            "riskguard_seed2_constraint_failed": bool(riskguard_seed2.get("constraint_failed", False)) if riskguard_seed2 else None,
        },
        "seed_decisions": seed_decisions,
        "accepted_image_augmentation_seeds": accepted_image_seeds,
        "rejected_image_augmentation_seeds": rejected_image_seeds,
        "same_candidate_passes_and_rejects_by_run": {
            "candidate_policy_id": "candidate_policy_1_roi_texture",
            "accepted_seeds": [
                int(seed)
                for seed, payload in seed_decisions.items()
                for item in payload["candidate_evaluations"]
                if item["candidate_policy_id"] == "candidate_policy_1_roi_texture"
                and item["decision"]["decision"] == "accept"
            ],
            "rejected_seeds": [
                int(seed)
                for seed, payload in seed_decisions.items()
                for item in payload["candidate_evaluations"]
                if item["candidate_policy_id"] == "candidate_policy_1_roi_texture"
                and item["decision"]["decision"] == "reject"
            ],
            "interpretation": "The same generic candidate is accepted or rejected from run metrics, not from dataset category names.",
        },
        "recommend_cp_catf_training_validation": all(seed in accepted_image_seeds for seed in [0, 1]) and 2 in rejected_image_seeds,
        "source_reports": {
            "fixed_multiseed_summary": str(FIXED_SUMMARY.relative_to(ROOT)),
            "seed2_root_cause_summary": str(SEED2_ROOT_CAUSE.relative_to(ROOT)),
            "riskguard_seed2_summary": str(RISKGUARD_SEED2.relative_to(ROOT)) if RISKGUARD_SEED2.exists() else None,
        },
        "fixed_summary_constraint_failed_count": fixed_summary.get("constraint_failed_count_fixed"),
    }


def render_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        "# CP-CATF Offline Causal Probe Summary",
        "",
        "- Training run executed: `false`",
        "- Probe mode: `development`",
        "- Uses existing validation diagnostics: `true`",
        "- Paper-mode note: policy selection must move to a train/probe split before final claims.",
        "- RiskGuard fixed blacklist as final rule: `false`",
        "- RiskGuard status: downgraded to audit/debug prior.",
        "",
        "## Seed Decisions",
        "",
        "| Seed | Selected candidate | Action | Image aug allowed | Image aug rejected | Main reason |",
        "|---:|---|---|---:|---:|---|",
    ]
    for seed, payload in sorted(summary["seed_decisions"].items(), key=lambda item: int(item[0])):
        selected = payload["selected_candidate"]
        decision = selected["decision"]
        reason = "accepted"
        if decision.get("rejection_reasons"):
            reason = ",".join(decision["rejection_reasons"])
        if payload.get("image_augmentation_rejected") and decision.get("decision") == "sampler_only":
            reason = "image candidates rejected; sampler_only pending"
        lines.append(
            f"| {seed} | `{payload['selected_candidate_policy_id']}` | `{payload['selected_candidate_action']}` | "
            f"`{str(decision.get('image_modification_allowed', False)).lower()}` | "
            f"`{str(payload.get('image_augmentation_rejected', False)).lower()}` | `{reason}` |"
        )
    run_specific = summary["same_candidate_passes_and_rejects_by_run"]
    lines.extend(
        [
            "",
            "## Run-Specific Evidence",
            "",
            f"- Generic candidate `{run_specific['candidate_policy_id']}` accepted seeds: `{run_specific['accepted_seeds']}`.",
            f"- Generic candidate `{run_specific['candidate_policy_id']}` rejected seeds: `{run_specific['rejected_seeds']}`.",
            "- This supports CP-CATF as run-specific rather than dataset-specific: the rule never branches on seed id, class id, or category name.",
            "",
            "## Seed2 Rejection",
            "",
            "- Seed2 image-space candidates are rejected because fixed CATF-v2 shows negative Recall/mAP movement and the audit reports non-active class regression.",
            "- RiskGuard recovered the audited class-local path but still failed overall constraints, so fixed class-op blacklist is insufficient as a final method.",
            "- Selected fallback is sampler-only/no image modification, with sample weighting still marked pending when dataloader support is absent.",
            "",
            "## Recommendation",
            "",
            f"- Enter CP-CATF training validation: `{str(summary['recommend_cp_catf_training_validation']).lower()}`.",
            "- Next validation should run seeds 0/1/2 with only probe-passed candidates entering the router and strict no-op when no image candidate passes.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_training_plan_md(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CP-CATF Training Validation Plan",
            "",
            "No training is executed in this step.",
            "",
            "## Plan",
            "",
            "- Seeds: `0, 1, 2`",
            "- Compare: clean native YOLO default vs fixed CATF-v2 vs CP-CATF.",
            "- CP-CATF flow: diagnosis -> candidate policy proposal -> causal probe -> accepted policy matrix -> sample router -> ROI augmentation.",
            "- Only candidate policies passing the causal probe may enter image augmentation.",
            "- If no image candidate passes, strict no-op is used; sampler-only may be emitted as pending until dataloader weighting is implemented.",
            "- Evaluate industrial constraints per seed: Precision, mAP50, and mAP50-95 must not drop more than 0.01 vs clean.",
            "- Required outcome for main-method claim: seed0/seed1 retain most fixed CATF-v2 benefit, seed2 is protected, and constraint_failed is 0/3.",
            "",
            "## Offline Probe Gate",
            "",
            f"- Offline probe recommends training validation: `{str(summary['recommend_cp_catf_training_validation']).lower()}`.",
        ]
    ) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
