from __future__ import annotations

import argparse
import json
import sys
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


DEFAULT_RERUN_ROOT = PROJECT_ROOT / "outputs/experiments/cp_catf_paper_mode_precision_gate_seed0_rerun"
DEFAULT_DRY_RUN_ROOT = PROJECT_ROOT / "outputs/experiments/cp_catf_precision_gate_dry_run_seed0"

PAPER_CLEAN_SEED0 = {
    "precision": 0.7513,
    "recall": 0.6763,
    "map50": 0.7566,
    "map50_95": 0.5114,
}


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


def event_epochs(events: list[dict[str, Any]]) -> list[int]:
    return sorted({int(event.get("epoch", -1)) for event in events if int(event.get("epoch", -1)) >= 0})


def find_event(events: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    for event in events:
        if int(event.get("epoch", -1)) == int(epoch):
            return event
    return {}


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


def load_metrics(reports: Path) -> dict[str, Any]:
    online = read_json(reports / "online_aug_stats.json")
    metrics = online.get("val_metrics") or {}
    return {
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "map50": float(metrics.get("map50", 0.0)),
        "map50_95": float(metrics.get("map50_95", 0.0)),
        "images": metrics.get("images"),
        "instances": metrics.get("instances"),
    }


def paper_deltas(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(metrics.get(key, 0.0)) - float(PAPER_CLEAN_SEED0[key])
        for key in ("precision", "recall", "map50", "map50_95")
    }


def paper_constraint_failed(deltas: dict[str, float]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if deltas["precision"] < -0.01:
        reasons.append("precision_drop_gt_0.01")
    if deltas["map50"] < -0.01:
        reasons.append("map50_drop_gt_0.01")
    if deltas["map50_95"] < -0.01:
        reasons.append("map50_95_drop_gt_0.01")
    return bool(reasons), reasons


def reconstruct_epoch_decisions(reports: Path, events: list[dict[str, Any]], leakage: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy_selection_data = str(leakage.get("policy_selection_data_yaml") or leakage.get("policy_selection_data") or "")
    for epoch in event_epochs(events):
        per_class_path = reports / f"per_class_diagnosis_epoch_{epoch}.json"
        attribution_path = reports / f"issue_attribution_epoch_{epoch}.json"
        policy_path = reports / f"policy_matrix_epoch_{epoch}_after.json"
        if not per_class_path.exists() or not attribution_path.exists() or not policy_path.exists():
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
        selected_event = find_event(events, epoch)
        image_rejections = []
        image_accepts = []
        for item in decision.get("candidate_evaluations", []):
            policy_def = item.get("candidate_policy") or {}
            item_decision = item.get("decision") or {}
            if not bool(policy_def.get("image_modification", False)):
                continue
            record = {
                "candidate_policy_id": item.get("candidate_policy_id"),
                "op_list": list(policy_def.get("op_list") or []),
                "decision": item_decision.get("decision"),
                "accepted": bool(item_decision.get("accepted", False)),
                "causal_score": item.get("causal_score"),
                "rejection_reasons": list(item_decision.get("rejection_reasons") or []),
                "precision_gate": item_decision.get("precision_gate") or item.get("precision_gate_metrics") or {},
                "precision_gate_thresholds": item_decision.get("precision_gate_thresholds") or {},
            }
            if item_decision.get("decision") == "accept":
                image_accepts.append(record)
            else:
                image_rejections.append(record)
        rows.append(
            {
                "epoch": epoch,
                "selected_candidate_policy_id": selected_event.get("selected_candidate_policy_id"),
                "selected_candidate_action": selected_event.get("selected_candidate_action"),
                "candidate_class_id": selected_event.get("candidate_class_id"),
                "causal_score": selected_event.get("selected_causal_score"),
                "image_modification_allowed": bool(selected_event.get("image_modification_allowed", False)),
                "sample_weighting_allowed": bool(selected_event.get("sample_weighting_allowed", False)),
                "sample_weighting_effective": bool(selected_event.get("sample_weighting_effective", False)),
                "sample_weighting_status": selected_event.get("sample_weighting_status"),
                "probe_reject_image_aug": bool(selected_event.get("probe_reject_image_aug", False)),
                "final_val_used_for_policy_selection": bool(selected_event.get("final_val_used_for_policy_selection", False)),
                "no_op_reason": selected_event.get("reason"),
                "active_rows": [
                    {
                        "class_id": int(row.get("class_id", -1)),
                        "dominant_issue": row.get("dominant_issue"),
                        "evidence_count": row.get("evidence_count"),
                        "diagnosis_confidence": row.get("diagnosis_confidence"),
                    }
                    for row in active_rows
                ],
                "image_candidate_accepts": image_accepts,
                "image_candidate_rejections": image_rejections,
            }
        )
    return rows


def build_payload(rerun_root: Path, dry_run_root: Path) -> dict[str, Any]:
    reports = rerun_root / "reports"
    dry_report = read_json(dry_run_root / "reports/precision_gate_dry_run_report.json")
    online = read_json(reports / "online_aug_stats.json")
    roi = read_json(reports / "roi_aug_stats.json")
    leakage = read_json(reports / "paper_probe_leakage_audit.json")
    events = read_json(reports / "causal_probe_events.json").get("events", [])
    legacy_constraint = read_json(reports / "constraint_scoring.json")
    metrics = load_metrics(reports)
    deltas = paper_deltas(metrics)
    constraint_failed, failure_reasons = paper_constraint_failed(deltas)
    epoch_decisions = reconstruct_epoch_decisions(reports, events, leakage)
    image_accept_epochs = [
        row["epoch"]
        for row in epoch_decisions
        if row.get("image_modification_allowed") or row.get("image_candidate_accepts")
    ]
    precision_gate_reject_epochs = [
        {
            "epoch": row["epoch"],
            "candidate_rejections": [
                item
                for item in row.get("image_candidate_rejections", [])
                if any(str(reason).endswith("_too_high") for reason in item.get("rejection_reasons", []))
            ],
        }
        for row in epoch_decisions
    ]
    precision_gate_reject_epochs = [
        row for row in precision_gate_reject_epochs if row.get("candidate_rejections")
    ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": 0,
        "rerun_root": str(rerun_root),
        "dry_run_root": str(dry_run_root),
        "dry_run": {
            "epoch25_roi_texture_rejected": dry_report.get("answers", {}).get("new_precision_gate_rejects_roi_texture"),
            "estimated_precision_drop": dry_report.get("answers", {}).get("estimated_precision_drop"),
            "non_active_fp_delta": dry_report.get("answers", {}).get("non_active_fp_delta"),
            "high_confidence_fp_delta": dry_report.get("answers", {}).get("high_confidence_fp_delta"),
            "selected_candidate_after_replay": dry_report.get("new_selected_candidate", {}).get("candidate_policy_id"),
            "final_val_leakage": dry_report.get("answers", {}).get("final_val_leakage"),
        },
        "paper_clean_seed0_reference": PAPER_CLEAN_SEED0,
        "seed0_precision_gate_metrics": metrics,
        "delta_vs_paper_clean_seed0": deltas,
        "constraint_failed_vs_paper_clean_seed0": constraint_failed,
        "constraint_failure_reasons_vs_paper_clean_seed0": failure_reasons,
        "legacy_trainer_constraint_scoring": {
            "note": "This trainer artifact uses the older clean_native_yolo_default reference, not the paper clean seed0 baseline requested for this rerun.",
            "constraint_failed": legacy_constraint.get("constraint_failed"),
            "failure_reasons": legacy_constraint.get("failure_reasons", []),
            "baseline_metrics": legacy_constraint.get("baseline_metrics", {}),
            "deltas": legacy_constraint.get("deltas", {}),
        },
        "execution_stats": {
            "roi_applied": int(roi.get("roi_aug_applied", 0)),
            "industrial_image_augmented": int(online.get("samples_augmented", 0)),
            "router_random_draw_count": int(online.get("router_random_draw_count", 0)),
            "samples_seen": int(online.get("samples_seen", 0)),
            "bbox_class_legal": bool(
                int(online.get("invalid_bbox_count", 0)) == 0
                and int(online.get("bbox_oob_count", 0)) == 0
                and int(online.get("class_id_oob_count", 0)) == 0
            ),
        },
        "leakage": {
            "final_val_leakage": bool(leakage.get("final_val_leakage", False)),
            "final_val_used_for_policy_selection": bool(leakage.get("final_val_used_for_policy_selection", False)),
            "train_core_probe_overlap_count": int(leakage.get("train_core_probe_overlap_count", 0)),
            "probe_final_val_overlap_count": int(leakage.get("probe_final_val_overlap_count", 0)),
            "train_core_final_val_overlap_count": int(leakage.get("train_core_final_val_overlap_count", 0)),
            "policy_selection_source": leakage.get("policy_selection_source"),
        },
        "epoch_decisions": epoch_decisions,
        "image_accept_epochs": image_accept_epochs,
        "precision_gate_image_reject_epochs": precision_gate_reject_epochs,
        "answers": {
            "dry_run_rejected_epoch25_roi_texture": dry_report.get("answers", {}).get("new_precision_gate_rejects_roi_texture"),
            "seed0_rerun_executed": True,
            "any_image_candidate_accepted_in_rerun": bool(image_accept_epochs),
            "roi_applied_gt_zero": int(roi.get("roi_aug_applied", 0)) > 0,
            "industrial_image_augmented_gt_zero": int(online.get("samples_augmented", 0)) > 0,
            "router_random_draw_count_gt_zero": int(online.get("router_random_draw_count", 0)) > 0,
            "final_val_leakage": bool(leakage.get("final_val_leakage", False)),
            "recommend_continue_seed1_seed2": False,
            "recommendation_reason": (
                "seed0 passes the requested paper-clean constraint, but the rerun is strict image no-op because all image candidates were rejected and sampler-only is pending; do not proceed to formal seed1/seed2 enhancement validation until an accepted image candidate can pass the precision gate or sampler-only is actually wired into the dataloader."
                if not constraint_failed
                else "seed0 still fails the requested paper-clean constraint; do not continue seed1/seed2."
            ),
        },
    }
    return payload


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["seed0_precision_gate_metrics"]
    deltas = payload["delta_vs_paper_clean_seed0"]
    dry = payload["dry_run"]
    exec_stats = payload["execution_stats"]
    leak = payload["leakage"]
    lines = [
        "# Seed0 CP-CATF Precision-Aware Gate Rerun",
        "",
        "## Dry Run Result",
        "",
        f"- Epoch 25 roi_texture rejected: `{str(dry.get('epoch25_roi_texture_rejected')).lower()}`.",
        f"- estimated_precision_drop: `{f4(dry.get('estimated_precision_drop'))}`.",
        f"- non_active_fp_delta: `{f4(dry.get('non_active_fp_delta'))}`.",
        f"- high_confidence_fp_delta: `{f4(dry.get('high_confidence_fp_delta'))}`.",
        f"- Selected after replay: `{dry.get('selected_candidate_after_replay')}`.",
        "",
        "## Rerun Candidate Decisions",
        "",
        "| epoch | selected candidate | action | class | score | image aug | sampler effective | reason |",
        "|---:|---|---|---:|---:|---|---|---|",
    ]
    for row in payload["epoch_decisions"]:
        lines.append(
            "| "
            f"{row.get('epoch')} | "
            f"`{row.get('selected_candidate_policy_id')}` | "
            f"`{row.get('selected_candidate_action')}` | "
            f"{row.get('candidate_class_id')} | "
            f"{f4(row.get('causal_score'))} | "
            f"`{str(row.get('image_modification_allowed')).lower()}` | "
            f"`{str(row.get('sample_weighting_effective')).lower()}` | "
            f"`{row.get('no_op_reason')}` |"
        )
    lines.extend(
        [
            "",
            "## Execution Stats",
            "",
            f"- ROI applied: `{exec_stats.get('roi_applied')}`.",
            f"- Industrial image augmented: `{exec_stats.get('industrial_image_augmented')}`.",
            f"- Router random draw count: `{exec_stats.get('router_random_draw_count')}`.",
            f"- BBox/class legal: `{str(exec_stats.get('bbox_class_legal')).lower()}`.",
            "",
            "## Final Val Leakage",
            "",
            f"- Final val leakage: `{str(leak.get('final_val_leakage')).lower()}`.",
            f"- Final val used for policy selection: `{str(leak.get('final_val_used_for_policy_selection')).lower()}`.",
            f"- train_core/probe overlap: `{leak.get('train_core_probe_overlap_count')}`.",
            f"- probe/final val overlap: `{leak.get('probe_final_val_overlap_count')}`.",
            f"- train_core/final val overlap: `{leak.get('train_core_final_val_overlap_count')}`.",
            "",
            "## Metrics Against Paper Clean Seed0",
            "",
            "| metric | clean paper seed0 | CP-CATF precision gate seed0 | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("precision", "P"),
        ("recall", "R"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
    ]:
        lines.append(
            f"| {label} | {f4(PAPER_CLEAN_SEED0[key])} | {f4(metrics.get(key))} | {f4(deltas.get(key))} |"
        )
    lines.extend(
        [
            "",
            f"- constraint_failed vs paper clean seed0: `{str(payload.get('constraint_failed_vs_paper_clean_seed0')).lower()}`.",
            f"- failure reasons vs paper clean seed0: `{', '.join(payload.get('constraint_failure_reasons_vs_paper_clean_seed0') or ['none'])}`.",
            "",
            "## Interpretation",
            "",
            "- The precision-aware gate blocks the known epoch 25 roi_texture risk before training execution.",
            "- The rerun does not execute image augmentation: all feedback points select sampler_only, and sampler weighting is still pending dataloader support, so the image path is strict no-op.",
            "- The trainer's legacy constraint artifact still compares against an older clean_native_yolo_default reference; this report uses the requested paper clean seed0 baseline.",
            f"- Recommendation to continue seed1/seed2: `{str(payload.get('answers', {}).get('recommend_continue_seed1_seed2')).lower()}`.",
            f"- Recommendation reason: {payload.get('answers', {}).get('recommendation_reason')}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize seed0 CP-CATF precision-gate rerun.")
    parser.add_argument("--rerun-root", default=str(DEFAULT_RERUN_ROOT))
    parser.add_argument("--dry-run-root", default=str(DEFAULT_DRY_RUN_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rerun_root = Path(args.rerun_root).resolve()
    dry_run_root = Path(args.dry_run_root).resolve()
    payload = build_payload(rerun_root, dry_run_root)
    reports = rerun_root / "reports"
    write_json(reports / "seed0_precision_gate_rerun_report.json", payload)
    write_markdown(reports / "seed0_precision_gate_rerun_report.md", payload)
    print(json.dumps(payload["answers"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
