from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.catf_v2.policy_matrix import active_class_ids
from scripts.train_yolo_default_with_inloop_feedback import (
    build_probe_decision_from_rows,
    merge_per_class_and_attribution,
)


DEFAULT_SOURCE = PROJECT_ROOT / "outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/experiments/cp_catf_precision_gate_dry_run_seed0"
DEFAULT_EPOCH = 25


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f4(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def find_epoch_event(events: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    for event in events:
        if int(event.get("epoch", -1)) == int(epoch):
            return event
    return {}


def active_rows_for_epoch(rows: list[dict[str, Any]], policy: dict[str, Any], top_k: int = 2) -> list[dict[str, Any]]:
    active_ids = set(active_class_ids(policy))
    active_rows = [row for row in rows if int(row.get("class_id", -1)) in active_ids]
    if active_rows:
        return active_rows
    return [
        row
        for row in rows
        if bool(row.get("strong_update_allowed", False)) and not bool(row.get("no_aug_class", False))
    ][:top_k]


def build_report_payload(source_root: Path, output_root: Path, epoch: int) -> dict[str, Any]:
    reports = source_root / "reports"
    per_class = read_json(reports / f"per_class_diagnosis_epoch_{epoch}.json")
    attribution = read_json(reports / f"issue_attribution_epoch_{epoch}.json")
    policy = read_json(reports / f"policy_matrix_epoch_{epoch}_after.json")
    events = read_json(reports / "causal_probe_events.json").get("events", [])
    flow = read_json(reports / "seed0_execution_flow_report.json")
    leakage = read_json(reports / "paper_probe_leakage_audit.json")

    class_rows = merge_per_class_and_attribution(per_class, attribution)
    active_rows = active_rows_for_epoch(class_rows, policy, top_k=2)
    original_event = find_epoch_event(events, epoch)
    original_flow_event = find_epoch_event(flow.get("events", []), epoch)
    policy_selection_data = str(original_event.get("policy_selection_data") or leakage.get("policy_selection_data") or "")
    decision = build_probe_decision_from_rows(
        active_rows=active_rows,
        context_rows=class_rows,
        policy=policy,
        epoch_num=epoch,
        mode="paper",
        policy_selection_data=policy_selection_data,
        policy_selection_source="probe_split",
    )
    image_eval = next(
        item
        for item in decision.get("candidate_evaluations", [])
        if item.get("candidate_policy_id") == "candidate_policy_1_roi_texture"
    )
    image_decision = image_eval.get("decision") or {}
    selected = decision.get("selected_candidate") or {}
    selected_decision = selected.get("decision") or {}
    precision_gate = image_decision.get("precision_gate") or image_eval.get("precision_gate_metrics") or {}
    precision_reasons = [
        reason
        for reason in image_decision.get("rejection_reasons", [])
        if reason
        in {
            "estimated_precision_drop_too_high",
            "non_active_fp_delta_too_high",
            "high_confidence_fp_delta_too_high",
        }
    ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_root": str(source_root),
        "output_root": str(output_root),
        "dry_run_only": True,
        "training_executed": False,
        "seed": 0,
        "epoch": int(epoch),
        "policy_selection_source": "probe_split",
        "policy_selection_data": policy_selection_data,
        "final_val_used_for_policy_selection": False,
        "final_val_leakage": bool(leakage.get("final_val_leakage", False)),
        "leakage_audit": {
            "train_core_probe_overlap_count": leakage.get("train_core_probe_overlap_count"),
            "probe_final_val_overlap_count": leakage.get("probe_final_val_overlap_count"),
            "train_core_final_val_overlap_count": leakage.get("train_core_final_val_overlap_count"),
        },
        "original_epoch25": {
            "candidate_policy_id": original_event.get("selected_candidate_policy_id"),
            "candidate_action": original_event.get("selected_candidate_action"),
            "accepted": bool(original_event.get("image_modification_allowed", False)),
            "active_class": original_event.get("candidate_class_id"),
            "op_list": list(original_event.get("op_whitelist") or []),
            "causal_score": original_event.get("selected_causal_score"),
            "ops_injected": deepcopy(original_event.get("candidate_ops_injected") or []),
            "execution_status": original_flow_event.get("final_execution_status"),
            "roi_applied": original_flow_event.get("roi_aug_applied_count"),
            "industrial_image_augmented": original_flow_event.get("industrial_aug_applied_count"),
            "router_random_draw_count": original_flow_event.get("router_random_draw_count"),
        },
        "active_rows_replayed": [
            {
                "class_id": int(row.get("class_id", -1)),
                "dominant_issue": row.get("dominant_issue"),
                "evidence_count": row.get("evidence_count"),
                "diagnosis_confidence": row.get("diagnosis_confidence"),
                "strong_update_allowed": bool(row.get("strong_update_allowed", False)),
            }
            for row in active_rows
        ],
        "new_epoch25_roi_texture_evaluation": {
            "candidate_policy_id": image_eval.get("candidate_policy_id"),
            "decision": image_decision.get("decision"),
            "accepted": bool(image_decision.get("accepted", False)),
            "image_modification_allowed": bool(image_decision.get("image_modification_allowed", False)),
            "causal_score": image_eval.get("causal_score"),
            "benefit_metrics": image_eval.get("benefit_metrics"),
            "risk_metrics": image_eval.get("risk_metrics"),
            "weighted_risk_metrics": image_eval.get("weighted_risk_metrics"),
            "precision_gate_metrics": precision_gate,
            "precision_gate_thresholds": image_decision.get("precision_gate_thresholds"),
            "rejection_reasons": image_decision.get("rejection_reasons", []),
            "precision_gate_rejection_reasons": precision_reasons,
            "audit_prior_only": bool(image_decision.get("audit_prior_only", False)),
            "riskguard_used_as_final_rule": False,
        },
        "new_selected_candidate": {
            "candidate_policy_id": decision.get("selected_candidate_policy_id"),
            "candidate_action": decision.get("selected_candidate_action"),
            "decision": selected_decision.get("decision"),
            "image_modification_allowed": bool(selected_decision.get("image_modification_allowed", False)),
            "sample_weighting_allowed": bool(selected_decision.get("sample_weighting_allowed", False)),
            "sample_weighting_effective": bool(selected_decision.get("sample_weighting_effective", False)),
            "sample_weighting_status": selected_decision.get("sample_weighting_status"),
            "strict_image_noop": bool(decision.get("strict_image_noop", False)),
            "sampler_only_selected": bool(decision.get("sampler_only_selected", False)),
        },
        "answers": {
            "epoch25_original_candidate_is_roi_texture": original_event.get("selected_candidate_policy_id")
            == "candidate_policy_1_roi_texture",
            "epoch25_original_candidate_was_accepted": bool(original_event.get("image_modification_allowed", False)),
            "new_precision_gate_rejects_roi_texture": image_decision.get("decision") == "reject",
            "estimated_precision_drop": precision_gate.get("estimated_precision_drop", 0.0),
            "non_active_fp_delta": precision_gate.get("non_active_fp_delta", 0.0),
            "high_confidence_fp_delta": precision_gate.get("high_confidence_fp_delta", 0.0),
            "converted_to_sampler_only": decision.get("selected_candidate_policy_id")
            == "candidate_policy_3_sampler_only",
            "sampler_only_effective": bool(selected_decision.get("sample_weighting_effective", False)),
            "strict_noop_if_sampler_pending": decision.get("selected_candidate_policy_id")
            == "candidate_policy_3_sampler_only"
            and not bool(selected_decision.get("sample_weighting_effective", False)),
            "decision_depends_on_seed_id": False,
            "decision_depends_on_fixed_class_id": False,
            "final_val_leakage": bool(leakage.get("final_val_leakage", False)),
            "recommend_seed0_50ep_rerun": image_decision.get("decision") == "reject",
        },
        "candidate_evaluations": decision.get("candidate_evaluations", []),
    }
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    original = payload["original_epoch25"]
    evaluation = payload["new_epoch25_roi_texture_evaluation"]
    selected = payload["new_selected_candidate"]
    answers = payload["answers"]
    precision = evaluation.get("precision_gate_metrics") or {}
    reasons = evaluation.get("rejection_reasons") or []
    lines = [
        "# CP-CATF Precision Gate Dry Run - Seed0",
        "",
        "## Scope",
        "",
        "- No training was run.",
        "- The dry run replays epoch 25 from the existing paper-mode seed0 execution-fixed run.",
        "- Policy selection source is the paper-mode probe split.",
        "- Final validation is not used for candidate decision.",
        "",
        "## Original Epoch 25",
        "",
        f"- Candidate: `{original.get('candidate_policy_id')}`.",
        f"- Original accepted: `{str(original.get('accepted')).lower()}`.",
        f"- Active class: `{original.get('active_class')}`.",
        f"- Ops: `{', '.join(original.get('op_list') or [])}`.",
        f"- ROI applied in original run: `{original.get('roi_applied')}`.",
        f"- Industrial image augmented in original run: `{original.get('industrial_image_augmented')}`.",
        f"- Router random draw count in original run: `{original.get('router_random_draw_count')}`.",
        "",
        "## New Precision-Aware Decision",
        "",
        f"- ROI texture decision: `{evaluation.get('decision')}`.",
        f"- Image modification allowed: `{str(evaluation.get('image_modification_allowed')).lower()}`.",
        f"- Rejection reasons: `{', '.join(reasons)}`.",
        f"- estimated_precision_drop: `{f4(precision.get('estimated_precision_drop'))}`.",
        f"- non_active_fp_delta: `{f4(precision.get('non_active_fp_delta'))}`.",
        f"- high_confidence_fp_delta: `{f4(precision.get('high_confidence_fp_delta'))}`.",
        f"- Selected candidate after replay: `{selected.get('candidate_policy_id')}` / `{selected.get('candidate_action')}`.",
        f"- Sampler-only effective: `{str(selected.get('sample_weighting_effective')).lower()}`.",
        f"- Strict no-op if sampler pending: `{str(answers.get('strict_noop_if_sampler_pending')).lower()}`.",
        "",
        "## Leakage And Specificity",
        "",
        f"- Final val leakage: `{str(payload.get('final_val_leakage')).lower()}`.",
        f"- Decision depends on seed id: `{str(answers.get('decision_depends_on_seed_id')).lower()}`.",
        f"- Decision depends on fixed class id: `{str(answers.get('decision_depends_on_fixed_class_id')).lower()}`.",
        f"- RiskGuard used as final rule: `{str(evaluation.get('riskguard_used_as_final_rule')).lower()}`.",
        "",
        "## Recommendation",
        "",
        f"- Recommend seed0 50ep rerun: `{str(answers.get('recommend_seed0_50ep_rerun')).lower()}`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay seed0 CP-CATF epoch25 with precision-aware gate.")
    parser.add_argument("--source-run-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_run_root).resolve()
    output_root = Path(args.output_dir).resolve()
    payload = build_report_payload(source_root, output_root, int(args.epoch))
    report_dir = output_root / "reports"
    write_json(report_dir / "precision_gate_dry_run_report.json", payload)
    write_markdown(report_dir / "precision_gate_dry_run_report.md", payload)


if __name__ == "__main__":
    main()
