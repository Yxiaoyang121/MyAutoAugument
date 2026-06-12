from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.catf_v2.causal_probe import (  # noqa: E402
    DEFAULT_PRECISION_GATE_THRESHOLDS,
    decide_candidate_acceptance,
)
from AutoAugment.catf_v2.policy_matrix import active_class_ids  # noqa: E402
from scripts.train_yolo_default_with_inloop_feedback import (  # noqa: E402
    build_probe_decision_from_rows,
    merge_per_class_and_attribution,
)


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs/experiments/multiseed_cp_catf_paper_mode"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/cp_catf_decision_coverage_audit"
DEFAULT_EXEC_FIXED_SEED0 = PROJECT_ROOT / "outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only"
FEEDBACK_EPOCHS = (5, 10, 15, 20, 25, 30, 35, 40, 45)
SEEDS = (0, 1, 2)
IMAGE_ACTIONS = {"roi_image_aug"}
PRECISION_REASONS = {
    "estimated_precision_drop_too_high",
    "non_active_fp_delta_too_high",
    "high_confidence_fp_delta_too_high",
}
REQUESTED_REASON_KEYS = (
    "estimated_precision_drop_too_high",
    "non_active_fp_delta_too_high",
    "high_confidence_fp_delta_too_high",
    "fp_increase_rate_too_high",
    "high_fp_spillover_rate_too_high",
    "non_active_regression_too_high",
    "ok_class_false_activation",
    "bbox_instability_too_high",
    "no_positive_benefit",
    "insufficient_evidence",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def f4(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def pct(numer: int, denom: int) -> float:
    return 0.0 if denom <= 0 else float(numer) / float(denom)


def active_rows_for_epoch(rows: list[dict[str, Any]], policy: dict[str, Any], top_k: int = 2) -> list[dict[str, Any]]:
    ids = set(active_class_ids(policy))
    active_rows = [row for row in rows if int(row.get("class_id", -1)) in ids]
    if active_rows:
        return active_rows
    return [
        row
        for row in rows
        if bool(row.get("strong_update_allowed", False)) and not bool(row.get("no_aug_class", False))
    ][:top_k]


def event_by_epoch(events: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    for event in events:
        if int(event.get("epoch", -1)) == int(epoch):
            return event
    return {}


def original_decision_without_precision_gate(item: dict[str, Any]) -> dict[str, Any]:
    probe_set = item.get("probe_set") or {}
    return decide_candidate_acceptance(
        candidate_policy=item.get("candidate_policy") or {},
        benefit_metrics=item.get("benefit_metrics") or {},
        risk_metrics=item.get("risk_metrics") or {},
        evidence_count=int(probe_set.get("evidence_count", 0) or 0),
        diagnosis_confidence=float(probe_set.get("diagnosis_confidence", 0.0) or 0.0),
        audit_priors=list(probe_set.get("audit_priors") or []),
        max_estimated_precision_drop=math.inf,
        max_non_active_fp_delta=math.inf,
        max_high_confidence_fp_delta=math.inf,
    )


def decision_with_thresholds(item: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    probe_set = item.get("probe_set") or {}
    return decide_candidate_acceptance(
        candidate_policy=item.get("candidate_policy") or {},
        benefit_metrics=item.get("benefit_metrics") or {},
        risk_metrics=item.get("risk_metrics") or {},
        evidence_count=int(probe_set.get("evidence_count", 0) or 0),
        diagnosis_confidence=float(probe_set.get("diagnosis_confidence", 0.0) or 0.0),
        audit_priors=list(probe_set.get("audit_priors") or []),
        max_estimated_precision_drop=float(thresholds["estimated_precision_drop"]),
        max_non_active_fp_delta=float(thresholds["non_active_fp_delta"]),
        max_high_confidence_fp_delta=float(thresholds["high_confidence_fp_delta"]),
    )


def relaxed_acceptance_flags(item: dict[str, Any]) -> dict[str, bool]:
    base = dict(DEFAULT_PRECISION_GATE_THRESHOLDS)
    relaxed = {}
    for key in DEFAULT_PRECISION_GATE_THRESHOLDS:
        thresholds = dict(base)
        thresholds[key] = math.inf
        relaxed[f"accepted_if_relax_{key}"] = bool(decision_with_thresholds(item, thresholds).get("accepted", False))
    relaxed["accepted_if_relax_all_precision_gates"] = bool(original_decision_without_precision_gate(item).get("accepted", False))
    return relaxed


def requested_reason_counts(reasons: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    reason_set = set(reasons)
    for reason in reason_set:
        if reason in REQUESTED_REASON_KEYS:
            counts[reason] += 1
    if "causal_score_not_positive" in reason_set or "insufficient_active_class_benefit" in reason_set:
        counts["no_positive_benefit"] += 1
    if "insufficient_evidence_count" in reason_set or "diagnosis_confidence_too_low" in reason_set:
        counts["insufficient_evidence"] += 1
    if "non_active_regression_rate_too_high" in reason_set:
        counts["non_active_regression_too_high"] += 1
    if "bbox_instability_rate_too_high" in reason_set:
        counts["bbox_instability_too_high"] += 1
    return counts


def classify_final_action(selected: dict[str, Any]) -> tuple[str, bool, bool, bool]:
    decision = selected.get("decision") or {}
    image_allowed = bool(decision.get("image_modification_allowed", False))
    sampler_allowed = bool(decision.get("sample_weighting_allowed", False))
    sampler_effective = bool(decision.get("sample_weighting_effective", False))
    strict_noop = not image_allowed and (not sampler_allowed or not sampler_effective)
    if image_allowed:
        return "image_aug", sampler_allowed, sampler_effective, False
    if sampler_allowed and sampler_effective:
        return "sampler_only", sampler_allowed, sampler_effective, False
    return "strict_noop", sampler_allowed, sampler_effective, strict_noop


def risk_attenuation_candidate(record: dict[str, Any]) -> bool:
    if not bool(record.get("is_image_candidate")):
        return False
    if not bool(record.get("original_causal_accept")):
        return False
    if bool(record.get("final_accepted")):
        return False
    reasons = set(record.get("rejection_reasons_list") or [])
    hard_reasons = {
        "fp_increase_rate_too_high",
        "ok_class_false_activation",
        "insufficient_evidence_count",
        "diagnosis_confidence_too_low",
    }
    if reasons & hard_reasons:
        return False
    precision_values = [
        float(record.get("estimated_precision_drop", 0.0) or 0.0),
        float(record.get("non_active_fp_delta", 0.0) or 0.0),
        float(record.get("high_confidence_fp_delta", 0.0) or 0.0),
    ]
    return max(precision_values) <= 0.05


def collect_seed_records(input_root: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    run_root = input_root / f"cp_catf_seed_{seed}"
    reports = run_root / "reports"
    leakage = read_json(reports / "paper_probe_leakage_audit.json")
    events = read_json(reports / "causal_probe_events.json").get("events", [])
    enriched_events: list[dict[str, Any]] = []
    if int(seed) == 0:
        enriched_path = DEFAULT_EXEC_FIXED_SEED0 / "reports" / "causal_probe_events.json"
        if enriched_path.exists():
            enriched_events = read_json(enriched_path).get("events", [])
    policy_selection_data = str(leakage.get("policy_selection_data_yaml") or leakage.get("policy_selection_data") or "")
    candidate_records: list[dict[str, Any]] = []
    epoch_records: list[dict[str, Any]] = []
    missing_epochs: list[int] = []

    for epoch in FEEDBACK_EPOCHS:
        per_class_path = reports / f"per_class_diagnosis_epoch_{epoch}.json"
        attribution_path = reports / f"issue_attribution_epoch_{epoch}.json"
        policy_path = reports / f"policy_matrix_epoch_{epoch}_after.json"
        if not per_class_path.exists() or not attribution_path.exists() or not policy_path.exists():
            missing_epochs.append(epoch)
            continue

        per_class = read_json(per_class_path)
        attribution = read_json(attribution_path)
        policy = read_json(policy_path)
        class_rows = merge_per_class_and_attribution(per_class, attribution)
        active_rows = active_rows_for_epoch(class_rows, policy, top_k=2)
        decision = build_probe_decision_from_rows(
            active_rows=active_rows,
            context_rows=class_rows,
            policy=policy,
            epoch_num=epoch,
            mode="paper",
            policy_selection_data=policy_selection_data,
            policy_selection_source="probe_split",
        )
        selected = decision.get("selected_candidate") or {}
        selected_decision = selected.get("decision") or {}
        final_action, sampler_allowed, sampler_effective, strict_noop = classify_final_action(selected)
        selected_id = str(decision.get("selected_candidate_policy_id"))
        selected_policy = selected.get("candidate_policy") or {}
        selected_class_id = int((selected.get("probe_set") or {}).get("class_id", -1) or -1)
        original_event = event_by_epoch(events, epoch)
        enriched_event = event_by_epoch(enriched_events, epoch)
        logged_original_selected_id = str(original_event.get("selected_candidate_policy_id") or "")
        logged_original_image_allowed = bool(original_event.get("image_modification_allowed", False))

        epoch_records.append(
            {
                "seed": int(seed),
                "epoch": int(epoch),
                "selected_candidate_policy_id": selected_id,
                "selected_candidate_action": decision.get("selected_candidate_action"),
                "selected_active_class": selected_class_id,
                "selected_op_list": list(selected_policy.get("op_list") or []),
                "selected_causal_score": selected.get("causal_score"),
                "final_action": final_action,
                "final_image_aug": final_action == "image_aug",
                "final_sampler_only_selected": bool(sampler_allowed),
                "final_sampler_only_effective": bool(sampler_effective),
                "final_strict_noop": bool(strict_noop),
                "original_logged_selected_candidate": original_event.get("selected_candidate_policy_id"),
                "original_logged_action": original_event.get("selected_candidate_action"),
                "original_logged_image_modification_allowed": bool(original_event.get("image_modification_allowed", False)),
                "original_logged_causal_score": enriched_event.get("selected_causal_score", original_event.get("selected_causal_score")),
                "final_val_leakage": bool(leakage.get("final_val_leakage", False)),
                "final_val_used_for_policy_selection": bool(leakage.get("final_val_used_for_policy_selection", False)),
            }
        )

        active_row_by_class = {int(row.get("class_id", -1)): row for row in active_rows}
        for item in decision.get("candidate_evaluations", []):
            candidate_policy = item.get("candidate_policy") or {}
            candidate_id = str(item.get("candidate_policy_id"))
            final_decision = item.get("decision") or {}
            original_decision = original_decision_without_precision_gate(item)
            probe_set = item.get("probe_set") or {}
            active_class = int(probe_set.get("class_id", -1) or -1)
            active_row = active_row_by_class.get(active_class, {})
            risk_metrics = item.get("risk_metrics") or {}
            weighted_risk = item.get("weighted_risk_metrics") or {}
            precision_gate = final_decision.get("precision_gate") or item.get("precision_gate_metrics") or {}
            reasons = list(final_decision.get("rejection_reasons") or [])
            legacy_original_accept = bool(
                candidate_id == logged_original_selected_id
                and logged_original_image_allowed
                and candidate_policy.get("image_modification", False)
            )
            logged_original_score = None
            logged_original_class = None
            if candidate_id == logged_original_selected_id:
                logged_original_score = enriched_event.get("selected_causal_score", original_event.get("selected_causal_score"))
                logged_original_class = enriched_event.get("candidate_class_id", original_event.get("candidate_class_id"))
            precision_reject = (
                bool(candidate_policy.get("image_modification", False))
                and legacy_original_accept
                and not bool(final_decision.get("accepted", False))
                and bool(set(reasons) & PRECISION_REASONS)
            )
            relaxed_flags = relaxed_acceptance_flags(item)
            row = {
                "seed": int(seed),
                "epoch": int(epoch),
                "candidate_policy_id": candidate_id,
                "active_class": active_class,
                "dominant_issue": active_row.get("dominant_issue"),
                "op_list": list(candidate_policy.get("op_list") or []),
                "candidate_action": candidate_policy.get("action"),
                "is_image_candidate": bool(candidate_policy.get("image_modification", False)),
                "is_sampler_candidate": bool(candidate_policy.get("sample_weighting", False)),
                "original_causal_score": logged_original_score if logged_original_score is not None else item.get("causal_score"),
                "current_replay_causal_score": item.get("causal_score"),
                "original_causal_accept": bool(legacy_original_accept),
                "current_replay_causal_accept": bool(original_decision.get("accepted", False)),
                "original_causal_decision": original_decision.get("decision"),
                "legacy_original_selected_candidate": logged_original_selected_id,
                "legacy_original_image_modification_allowed": logged_original_image_allowed,
                "legacy_original_candidate_class_id": logged_original_class,
                "estimated_precision_drop": precision_gate.get("estimated_precision_drop", 0.0),
                "non_active_fp_delta": precision_gate.get("non_active_fp_delta", 0.0),
                "high_confidence_fp_delta": precision_gate.get("high_confidence_fp_delta", 0.0),
                "fp_increase_rate": risk_metrics.get("fp_increase_rate", weighted_risk.get("fp_increase_rate", 0.0)),
                "high_fp_spillover_rate": risk_metrics.get("high_fp_spillover_rate", weighted_risk.get("high_fp_spillover_rate", 0.0)),
                "non_active_regression_rate": risk_metrics.get(
                    "non_active_regression_rate", weighted_risk.get("non_active_regression_rate", 0.0)
                ),
                "ok_class_false_activation": risk_metrics.get(
                    "ok_class_false_activation", weighted_risk.get("ok_class_false_activation", 0.0)
                ),
                "bbox_instability_rate": risk_metrics.get(
                    "bbox_instability_rate", weighted_risk.get("bbox_instability_rate", 0.0)
                ),
                "final_decision_under_precision_gate": final_decision.get("decision"),
                "final_accepted": bool(final_decision.get("accepted", False)),
                "precision_gate_reject": bool(precision_reject),
                "rejection_reasons_list": reasons,
                "rejection_reasons": ";".join(reasons),
                "final_action": final_action if selected_id == candidate_id else "not_selected",
                "selected_by_precision_gate": selected_id == candidate_id,
                "executable_policy_would_be_generated": bool(
                    selected_id == candidate_id
                    and final_decision.get("image_modification_allowed", False)
                    and active_class >= 0
                ),
                "roi_augmentation_allowed": bool(
                    final_decision.get("image_modification_allowed", False)
                    and set(candidate_policy.get("op_list") or []).issubset({"sharpen_mild", "local_contrast", "gamma", "clahe"})
                ),
                "sampler_only_pending": bool(
                    final_decision.get("sample_weighting_allowed", False)
                    and not final_decision.get("sample_weighting_effective", False)
                ),
                "sampler_only_effective": bool(final_decision.get("sample_weighting_effective", False)),
                "candidate_score_rank_selected": selected_id == candidate_id,
                "final_val_leakage": bool(leakage.get("final_val_leakage", False)),
                "policy_selection_source": "probe_split",
                "policy_selection_data": policy_selection_data,
                **relaxed_flags,
            }
            row["attenuation_candidate"] = risk_attenuation_candidate(row)
            candidate_records.append(row)

    metadata = {
        "seed": int(seed),
        "run_root": str(run_root),
        "missing_epochs": missing_epochs,
        "final_val_leakage": bool(leakage.get("final_val_leakage", False)),
        "final_val_used_for_policy_selection": bool(leakage.get("final_val_used_for_policy_selection", False)),
        "train_core_image_count": leakage.get("train_core_image_count"),
        "probe_image_count": leakage.get("probe_image_count"),
        "final_val_image_count": leakage.get("final_val_image_count"),
    }
    return candidate_records, epoch_records, metadata


def coverage_from_records(records: list[dict[str, Any]], epoch_records: list[dict[str, Any]]) -> dict[str, Any]:
    image_records = [row for row in records if row.get("is_image_candidate")]
    image_original_accepts = [row for row in image_records if row.get("original_causal_accept")]
    image_current_replay_accepts = [row for row in image_records if row.get("current_replay_causal_accept")]
    precision_rejects = [row for row in image_records if row.get("precision_gate_reject")]
    final_image_accept = [row for row in epoch_records if row.get("final_image_aug")]
    final_sampler_selected = [row for row in epoch_records if row.get("final_sampler_only_selected")]
    final_sampler_effective = [row for row in epoch_records if row.get("final_sampler_only_effective")]
    final_noop = [row for row in epoch_records if row.get("final_strict_noop")]
    epoch_count = len(epoch_records)
    return {
        "total_candidates": len(records),
        "total_image_candidates": len(image_records),
        "feedback_epoch_count": epoch_count,
        "causal_score_accept_count": len(image_original_accepts),
        "current_replay_causal_accept_count": len(image_current_replay_accepts),
        "causal_or_sampler_accept_count": sum(
            1 for row in records if row.get("original_causal_accept") or row.get("is_sampler_candidate")
        ),
        "precision_gate_reject_count": len(precision_rejects),
        "final_image_accept_count": len(final_image_accept),
        "final_sampler_only_count": len(final_sampler_selected),
        "final_effective_sampler_only_count": len(final_sampler_effective),
        "final_noop_count": len(final_noop),
        "image_accept_rate": pct(len(final_image_accept), epoch_count),
        "sampler_only_rate": pct(len(final_sampler_selected), epoch_count),
        "effective_sampler_only_rate": pct(len(final_sampler_effective), epoch_count),
        "noop_rate": pct(len(final_noop), epoch_count),
        "attenuation_candidate_count": sum(1 for row in image_records if row.get("attenuation_candidate")),
    }


def reason_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter({key: 0 for key in REQUESTED_REASON_KEYS})
    raw_counts: Counter[str] = Counter()
    for row in records:
        if not row.get("is_image_candidate"):
            continue
        if bool(row.get("final_accepted")):
            continue
        reasons = list(row.get("rejection_reasons_list") or [])
        raw_counts.update(set(reasons))
        counts.update(requested_reason_counts(reasons))
    rows = []
    for reason in REQUESTED_REASON_KEYS:
        rows.append({"rejection_reason": reason, "count": int(counts.get(reason, 0))})
    for reason, count in sorted(raw_counts.items()):
        if reason not in REQUESTED_REASON_KEYS and reason not in {
            "causal_score_not_positive",
            "insufficient_active_class_benefit",
            "insufficient_evidence_count",
            "diagnosis_confidence_too_low",
            "non_active_regression_rate_too_high",
            "bbox_instability_rate_too_high",
        }:
            rows.append({"rejection_reason": f"raw:{reason}", "count": int(count)})
    return rows


def blocker_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    image_records = [row for row in records if row.get("is_image_candidate")]
    original_accept_rejects = [
        row for row in image_records if row.get("original_causal_accept") and not row.get("final_accepted")
    ]
    precision_rejects = [row for row in original_accept_rejects if row.get("precision_gate_reject")]
    by_reason = Counter()
    sole_precision_blockers = Counter()
    for row in precision_rejects:
        reasons = set(row.get("rejection_reasons_list") or [])
        precision_reasons = reasons & PRECISION_REASONS
        by_reason.update(precision_reasons)
        non_precision_reasons = reasons - PRECISION_REASONS
        if not non_precision_reasons and len(precision_reasons) == 1:
            sole_precision_blockers.update(precision_reasons)

    relax_counts = {
        "accepted_if_relax_estimated_precision_drop": sum(
            1
            for row in original_accept_rejects
            if row.get("accepted_if_relax_estimated_precision_drop")
        ),
        "accepted_if_relax_non_active_fp_delta": sum(
            1
            for row in original_accept_rejects
            if row.get("accepted_if_relax_non_active_fp_delta")
        ),
        "accepted_if_relax_high_confidence_fp_delta": sum(
            1
            for row in original_accept_rejects
            if row.get("accepted_if_relax_high_confidence_fp_delta")
        ),
        "accepted_if_relax_all_precision_gates": sum(
            1
            for row in original_accept_rejects
            if row.get("accepted_if_relax_all_precision_gates")
        ),
    }
    top_rejected = sorted(
        precision_rejects,
        key=lambda row: float(row.get("original_causal_score", 0.0) or 0.0),
        reverse=True,
    )[:10]
    return {
        "all_image_candidates_rejected": not any(row.get("final_accepted") for row in image_records),
        "image_candidates_final_accepted": sum(1 for row in image_records if row.get("final_accepted")),
        "original_causal_accept_then_precision_reject": len(precision_rejects),
        "precision_rejects_by_reason": dict(sorted(by_reason.items())),
        "sole_precision_blockers": dict(sorted(sole_precision_blockers.items())),
        "relaxation_accept_counts": relax_counts,
        "high_score_rejected_candidates": [
            {
                "seed": row["seed"],
                "epoch": row["epoch"],
                "candidate_policy_id": row["candidate_policy_id"],
                "active_class": row["active_class"],
                "dominant_issue": row.get("dominant_issue"),
                "causal_score": row.get("original_causal_score"),
                "current_replay_causal_score": row.get("current_replay_causal_score"),
                "estimated_precision_drop": row.get("estimated_precision_drop"),
                "non_active_fp_delta": row.get("non_active_fp_delta"),
                "high_confidence_fp_delta": row.get("high_confidence_fp_delta"),
                "rejection_reasons": row.get("rejection_reasons_list"),
                "attenuation_candidate": bool(row.get("attenuation_candidate")),
            }
            for row in top_rejected
        ],
        "attenuation_candidate_count": sum(1 for row in image_records if row.get("attenuation_candidate")),
    }


def build_report(
    *,
    output_root: Path,
    total: dict[str, Any],
    seed_coverage: list[dict[str, Any]],
    reason_rows: list[dict[str, Any]],
    blocker: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    reason_text = ", ".join(
        f"{row['rejection_reason']}={row['count']}"
        for row in reason_rows
        if row["count"] and not str(row["rejection_reason"]).startswith("raw:")
    )
    primary_reason = max(
        (row for row in reason_rows if not str(row["rejection_reason"]).startswith("raw:")),
        key=lambda row: int(row["count"]),
    )
    all_rejected = bool(blocker["all_image_candidates_rejected"])
    hcf_relax = blocker["relaxation_accept_counts"]["accepted_if_relax_high_confidence_fp_delta"]
    naf_relax = blocker["relaxation_accept_counts"]["accepted_if_relax_non_active_fp_delta"]
    epd_relax = blocker["relaxation_accept_counts"]["accepted_if_relax_estimated_precision_drop"]
    all_relax = blocker["relaxation_accept_counts"]["accepted_if_relax_all_precision_gates"]
    recommend_hierarchical = all_rejected and int(blocker.get("attenuation_candidate_count", 0)) > 0
    recommend_sampler = all_rejected and int(total.get("final_sampler_only_count", 0)) > 0
    if recommend_hierarchical and recommend_sampler:
        next_lever = "sampler_only implementation first; graded attenuation before image rerun"
    elif recommend_hierarchical:
        next_lever = "graded risk gate / precision-aware attenuation"
    elif recommend_sampler:
        next_lever = "sampler_only implementation"
    else:
        next_lever = "keep strict gate"

    lines = [
        "# CP-CATF Paper-Mode Decision Coverage Audit",
        "",
        "## Scope",
        "",
        "- No training was run.",
        "- No seed1/seed2 execution was launched.",
        "- Replay used existing paper-mode diagnostic JSON from `outputs/experiments/multiseed_cp_catf_paper_mode`.",
        "- Precision-aware decisions reuse current `build_probe_decision_from_rows` and `decide_candidate_acceptance` logic.",
        "- `causal_score_accept_count` is the logged pre-precision image accept count from existing paper-mode events; `current_replay_causal_accept_count` is the stricter current-code replay count.",
        "- Feedback epochs audited: `5, 10, 15, 20, 25, 30, 35, 40, 45` for seeds `0, 1, 2`.",
        "",
        "## Leakage Status",
        "",
        f"- train_core images: `{metadata.get('train_core_image_count')}`.",
        f"- probe images: `{metadata.get('probe_image_count')}`.",
        f"- final val images: `{metadata.get('final_val_image_count')}`.",
        f"- final val leakage detected: `{str(metadata.get('final_val_leakage')).lower()}`.",
        f"- final val used for policy selection: `{str(metadata.get('final_val_used_for_policy_selection')).lower()}`.",
        "",
        "## Coverage Summary",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "total_candidates",
        "total_image_candidates",
        "feedback_epoch_count",
        "causal_score_accept_count",
        "current_replay_causal_accept_count",
        "precision_gate_reject_count",
        "final_image_accept_count",
        "final_sampler_only_count",
        "final_effective_sampler_only_count",
        "final_noop_count",
    ]:
        lines.append(f"| {key} | {total.get(key)} |")
    for key in ["image_accept_rate", "sampler_only_rate", "effective_sampler_only_rate", "noop_rate"]:
        lines.append(f"| {key} | {float(total.get(key, 0.0)):.4f} |")

    lines.extend(
        [
            "",
            "## Seed Coverage",
            "",
            "| seed | total candidates | image candidates | causal accept | precision reject | final image accept | sampler selected | effective sampler | strict no-op |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in seed_coverage:
        lines.append(
            "| "
            f"{row['seed']} | {row['total_candidates']} | {row['total_image_candidates']} | "
            f"{row['causal_score_accept_count']} | {row['precision_gate_reject_count']} | "
            f"{row['final_image_accept_count']} | {row['final_sampler_only_count']} | "
            f"{row['final_effective_sampler_only_count']} | {row['final_noop_count']} |"
        )

    lines.extend(
        [
            "",
            "## Rejection Reasons",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
    )
    for row in reason_rows:
        lines.append(f"| `{row['rejection_reason']}` | {row['count']} |")

    lines.extend(
        [
            "",
            "## Gate Conservatism Analysis",
            "",
            f"- All image candidates rejected under current precision-aware gate: `{str(all_rejected).lower()}`.",
            f"- Main requested rejection bucket: `{primary_reason['rejection_reason']}` with `{primary_reason['count']}` hits.",
            f"- Precision-gate rejects after original causal accept: `{blocker['original_causal_accept_then_precision_reject']}`.",
            "- Note: current replay rejects those legacy image accepts for the full risk stack; precision thresholds are present but not sole blockers.",
            f"- Precision reject reasons among those candidates: `{json.dumps(blocker['precision_rejects_by_reason'], ensure_ascii=False, sort_keys=True)}`.",
            f"- Sole precision blockers: `{json.dumps(blocker['sole_precision_blockers'], ensure_ascii=False, sort_keys=True)}`.",
            f"- Candidates accepted if only high_confidence_fp_delta is relaxed: `{hcf_relax}`.",
            f"- Candidates accepted if only non_active_fp_delta is relaxed: `{naf_relax}`.",
            f"- Candidates accepted if only estimated_precision_drop is relaxed: `{epd_relax}`.",
            f"- Candidates accepted if all three precision gates are relaxed: `{all_relax}`.",
            f"- Attenuation candidates: `{blocker['attenuation_candidate_count']}`.",
            "",
            "## High-Score Rejections",
            "",
            "| seed | epoch | candidate | class | issue | logged/original score | current replay score | est P drop | non-active FP | high-conf FP | attenuation | reasons |",
            "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in blocker["high_score_rejected_candidates"][:8]:
        lines.append(
            "| "
            f"{row['seed']} | {row['epoch']} | `{row['candidate_policy_id']}` | {row['active_class']} | "
            f"`{row.get('dominant_issue')}` | {f4(row.get('causal_score'))} | "
            f"{f4(row.get('current_replay_causal_score'))} | "
            f"{f4(row.get('estimated_precision_drop'))} | {f4(row.get('non_active_fp_delta'))} | "
            f"{f4(row.get('high_confidence_fp_delta'))} | `{str(row.get('attenuation_candidate')).lower()}` | "
            f"`{';'.join(row.get('rejection_reasons') or [])}` |"
        )

    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"1. Current precision-aware decision stack is over-conservative for paper-mode image augmentation coverage: `{str(all_rejected).lower()}`.",
            "2. Current CP-CATF image augmentation is zero because every replayed image candidate is rejected under the current risk stack; the selected fallback is sampler_only, but sampler weighting is still pending dataloader support, so it becomes strict no-op in execution.",
            f"3. Safe executable image candidate under current gate: `{str(total.get('final_image_accept_count', 0) > 0).lower()}`.",
            f"4. Recommended next lever: `{next_lever}`.",
            "5. Continue training now: `false`.",
            "6. Run seed1/seed2 now: `false`.",
            "7. Modify gate then rerun seed0: `true` if the next change is a graded attenuation gate; otherwise implement sampler_only first.",
            "8. Use current result as paper method result: `false`.",
            "",
            "## Recommendation Options",
            "",
            "- Scheme A, strict gate: defensible only if zero image augmentation is an acceptable conclusion and the method pivots to sampler_only.",
            "- Scheme B, graded risk gate: recommended for candidates with positive causal score and medium FP risk; reduce probability/strength and top-m before rejecting.",
            "- Scheme C, precision-aware attenuation: recommended for positive-benefit candidates with mild precision risk; keep one low-risk op and re-evaluate at the next feedback epoch.",
            "- Scheme D, implement sampler_only: recommended if image augmentation remains unsafe after attenuation analysis.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{output_root / 'reports' / 'decision_coverage_audit.json'}`.",
            f"- Candidate records: `{output_root / 'decision_records.csv'}`.",
            f"- Seed coverage: `{output_root / 'seed_level_coverage.csv'}`.",
            f"- Rejection reasons: `{output_root / 'rejection_reason_summary.csv'}`.",
        ]
    )
    report_path = output_root / "reports/decision_coverage_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit(input_root: Path, output_root: Path) -> dict[str, Any]:
    all_candidate_records: list[dict[str, Any]] = []
    all_epoch_records: list[dict[str, Any]] = []
    seed_metadata: dict[str, Any] = {}
    seed_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        records, epoch_records, metadata = collect_seed_records(input_root, seed)
        all_candidate_records.extend(records)
        all_epoch_records.extend(epoch_records)
        seed_metadata[str(seed)] = metadata
        coverage = coverage_from_records(records, epoch_records)
        seed_rows.append({"seed": int(seed), **coverage})

    total = coverage_from_records(all_candidate_records, all_epoch_records)
    reason_rows = reason_summary(all_candidate_records)
    blocker = blocker_analysis(all_candidate_records)
    split_metadata = next(iter(seed_metadata.values()), {})
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "training_executed": False,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "feedback_epochs": list(FEEDBACK_EPOCHS),
        "seeds": list(SEEDS),
        "precision_gate_thresholds": dict(DEFAULT_PRECISION_GATE_THRESHOLDS),
        "seed_metadata": seed_metadata,
        "train_core_image_count": split_metadata.get("train_core_image_count"),
        "probe_image_count": split_metadata.get("probe_image_count"),
        "final_val_image_count": split_metadata.get("final_val_image_count"),
        "final_val_leakage": any(bool(item.get("final_val_leakage", False)) for item in seed_metadata.values()),
        "final_val_used_for_policy_selection": any(
            bool(item.get("final_val_used_for_policy_selection", False)) for item in seed_metadata.values()
        ),
    }
    payload = {
        "metadata": metadata,
        "coverage": total,
        "seed_level_coverage": seed_rows,
        "rejection_reason_summary": reason_rows,
        "blocker_analysis": blocker,
        "answers": {
            "all_image_candidates_rejected": blocker["all_image_candidates_rejected"],
            "primary_rejection_reason": max(
                (row for row in reason_rows if not str(row["rejection_reason"]).startswith("raw:")),
                key=lambda row: int(row["count"]),
            ),
            "high_confidence_fp_delta_over_strict": blocker["relaxation_accept_counts"][
                "accepted_if_relax_high_confidence_fp_delta"
            ]
            > 0,
            "non_active_fp_delta_over_strict": blocker["relaxation_accept_counts"][
                "accepted_if_relax_non_active_fp_delta"
            ]
            > 0,
            "estimated_precision_drop_over_strict": blocker["relaxation_accept_counts"][
                "accepted_if_relax_estimated_precision_drop"
            ]
            > 0,
            "safe_image_candidate_exists_current_gate": total["final_image_accept_count"] > 0,
            "attenuation_candidate_exists": blocker["attenuation_candidate_count"] > 0,
            "recommend_graded_gate": blocker["all_image_candidates_rejected"] and blocker["attenuation_candidate_count"] > 0,
            "recommend_sampler_only": blocker["all_image_candidates_rejected"] and total["final_sampler_only_count"] > 0,
            "recommend_continue_training": False,
            "recommend_run_seed1_seed2": False,
            "recommend_modify_gate_then_seed0": True,
            "recommend_as_paper_method_result": False,
        },
    }

    candidate_fields = [
        "seed",
        "epoch",
        "candidate_policy_id",
        "active_class",
        "dominant_issue",
        "op_list",
        "candidate_action",
        "is_image_candidate",
        "is_sampler_candidate",
        "original_causal_score",
        "current_replay_causal_score",
        "original_causal_accept",
        "current_replay_causal_accept",
        "original_causal_decision",
        "legacy_original_selected_candidate",
        "legacy_original_image_modification_allowed",
        "legacy_original_candidate_class_id",
        "estimated_precision_drop",
        "non_active_fp_delta",
        "high_confidence_fp_delta",
        "fp_increase_rate",
        "high_fp_spillover_rate",
        "non_active_regression_rate",
        "ok_class_false_activation",
        "bbox_instability_rate",
        "final_decision_under_precision_gate",
        "final_accepted",
        "precision_gate_reject",
        "rejection_reasons",
        "final_action",
        "selected_by_precision_gate",
        "executable_policy_would_be_generated",
        "roi_augmentation_allowed",
        "sampler_only_pending",
        "sampler_only_effective",
        "attenuation_candidate",
        "accepted_if_relax_estimated_precision_drop",
        "accepted_if_relax_non_active_fp_delta",
        "accepted_if_relax_high_confidence_fp_delta",
        "accepted_if_relax_all_precision_gates",
        "final_val_leakage",
        "policy_selection_source",
        "policy_selection_data",
    ]
    seed_fields = [
        "seed",
        "total_candidates",
        "total_image_candidates",
        "feedback_epoch_count",
        "causal_score_accept_count",
        "current_replay_causal_accept_count",
        "causal_or_sampler_accept_count",
        "precision_gate_reject_count",
        "final_image_accept_count",
        "final_sampler_only_count",
        "final_effective_sampler_only_count",
        "final_noop_count",
        "image_accept_rate",
        "sampler_only_rate",
        "effective_sampler_only_rate",
        "noop_rate",
        "attenuation_candidate_count",
    ]
    reason_fields = ["rejection_reason", "count"]

    write_csv(output_root / "decision_records.csv", all_candidate_records, candidate_fields)
    write_csv(output_root / "seed_level_coverage.csv", seed_rows, seed_fields)
    write_csv(output_root / "rejection_reason_summary.csv", reason_rows, reason_fields)
    write_json(output_root / "reports/decision_coverage_audit.json", payload)
    build_report(
        output_root=output_root,
        total=total,
        seed_coverage=seed_rows,
        reason_rows=reason_rows,
        blocker=blocker,
        metadata=metadata,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CP-CATF paper-mode decision coverage without training.")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    payload = build_audit(input_root=input_root, output_root=output_root)
    print(json.dumps(payload["coverage"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
