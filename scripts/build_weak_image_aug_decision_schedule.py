from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "catf_v2_image_only_weak_aug_replay"
    / "weak_candidate_records.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "catf_v2_image_only_weak_aug_seed2_50ep"
    / "configs"
    / "weak_image_aug_offline_decisions_seed2.json"
)

FEEDBACK_EPOCHS = (5, 10, 15, 20, 25, 30, 35, 40, 45)
ATTENUATION_RATIO = 0.25
MAX_AUG_SAMPLES_PER_INTERVAL = 16


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.replay_csv), seed=int(args.seed))
    payload = build_schedule(records, seed=int(args.seed))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build epoch-specific offline weak image augmentation decisions.")
    parser.add_argument("--replay-csv", default=str(DEFAULT_REPLAY_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def load_records(path: Path, *, seed: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"weak replay CSV not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row.get("seed", -1) or -1) == int(seed):
                out.append(row)
    by_epoch = {int(row["epoch"]): row for row in out}
    return [by_epoch[epoch] for epoch in FEEDBACK_EPOCHS if epoch in by_epoch]


def build_schedule(records: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    epoch_decisions: dict[str, dict[str, Any]] = {}
    weak_epochs: list[int] = []
    strict_noop_epochs: list[int] = []
    for record in records:
        epoch = int(record["epoch"])
        if record.get("final_decision") == "weak_roi_texture":
            decision = weak_decision_payload(record)
            weak_epochs.append(epoch)
        else:
            decision = strict_noop_payload(record)
            strict_noop_epochs.append(epoch)
        epoch_decisions[str(epoch)] = decision
    return {
        "schema_version": 1,
        "decision_schedule_type": "image_only_weak_augmentation",
        "seed": int(seed),
        "source_replay": str(DEFAULT_REPLAY_CSV),
        "sampler_only_involved": False,
        "sample_weighting_allowed": False,
        "weighted_index_list_allowed": False,
        "final_val_used_for_policy_selection": False,
        "policy_selection_source": "weak_image_aug_replay",
        "candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
        "attenuation_ratio": ATTENUATION_RATIO,
        "epoch_decisions": epoch_decisions,
        "summary": {
            "feedback_epochs": list(FEEDBACK_EPOCHS),
            "weak_epoch_count": len(weak_epochs),
            "weak_epochs": weak_epochs,
            "strict_noop_epoch_count": len(strict_noop_epochs),
            "strict_noop_epochs": strict_noop_epochs,
            "sampler_only_involved": False,
            "final_val_leakage": False,
        },
    }


def weak_decision_payload(record: dict[str, Any]) -> dict[str, Any]:
    epoch = int(record["epoch"])
    class_id = int(record["active_class"])
    retained_op = str(record.get("retained_op") or "local_contrast")
    original_prob = original_op_prob(retained_op)
    original_strength = original_op_strength(retained_op)
    weak_prob = round(original_prob * ATTENUATION_RATIO, 6)
    weak_strength = round(original_strength * ATTENUATION_RATIO, 6)
    candidate = {
        "policy_id": "candidate_policy_1b_weak_roi_texture",
        "action": "roi_image_aug",
        "op_list": [retained_op],
        "image_modification": True,
        "sample_weighting": False,
        "weak_image_aug": True,
        "attenuation_ratio": ATTENUATION_RATIO,
        "max_aug_samples_per_interval": MAX_AUG_SAMPLES_PER_INTERVAL,
        "derived_from_policy_id": "candidate_policy_1_roi_texture",
        "original_prob": original_prob,
        "original_strength": original_strength,
        "weak_prob": weak_prob,
        "weak_strength": weak_strength,
    }
    decision = {
        "candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
        "accepted": True,
        "decision": "accept",
        "causal_score": float(record.get("current_replay_causal_score") or 0.0),
        "rejection_reasons": [],
        "image_modification_allowed": True,
        "sample_weighting_allowed": False,
        "strict_noop": False,
        "precision_aware_gate_passed": True,
        "non_active_regression_gate_passed": True,
    }
    return {
        "epoch": epoch,
        "mode": "image_only_weak_replay",
        "decision_basis": "offline_weak_image_aug_replay_no_final_val",
        "policy_selection_source": "weak_image_aug_replay",
        "policy_selection_data": "outputs/experiments/catf_v2_image_only_weak_aug_replay/weak_candidate_records.csv",
        "selected_candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
        "selected_candidate_action": "weak_roi_texture",
        "original_candidate_policy_id": "candidate_policy_1_roi_texture",
        "downgraded_to_weak_roi_texture": True,
        "attenuation_ratio": ATTENUATION_RATIO,
        "retained_op": retained_op,
        "original_prob": original_prob,
        "original_strength": original_strength,
        "weak_prob": weak_prob,
        "weak_strength": weak_strength,
        "max_aug_samples_per_interval": MAX_AUG_SAMPLES_PER_INTERVAL,
        "precision_aware_gate_passed": True,
        "non_active_regression_gate_passed": True,
        "sampler_only_involved": False,
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
            "candidate_policy": candidate,
            "probe_set": {
                "class_id": class_id,
                "candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
                "mode": "image_only_weak_replay",
                "development_probe_uses_existing_val_diagnostics": False,
            },
            "causal_score": float(record.get("current_replay_causal_score") or 0.0),
            "decision": decision,
        },
        "candidate_evaluations": [],
        "image_augmentation_rejected": False,
        "strict_image_noop": False,
        "paper_probe_mode": False,
        "development_probe_uses_existing_val_diagnostics": False,
        "final_val_used_for_policy_selection": False,
        "forbid_final_val_policy_selection": False,
        "riskguard_interpretation": {
            "riskguard_used_as_final_rule": False,
            "riskguard_role": "audit_debug_prior_only",
        },
        "seed_specific_rule": False,
        "fixed_class_id_specific_rule": False,
        "dataset_specific_rule": False,
    }


def strict_noop_payload(record: dict[str, Any]) -> dict[str, Any]:
    epoch = int(record["epoch"])
    return {
        "epoch": epoch,
        "mode": "image_only_weak_replay",
        "decision_basis": "offline_weak_image_aug_replay_no_final_val",
        "policy_selection_source": "weak_image_aug_replay",
        "policy_selection_data": "outputs/experiments/catf_v2_image_only_weak_aug_replay/weak_candidate_records.csv",
        "selected_candidate_policy_id": "candidate_policy_0_noop",
        "selected_candidate_action": "strict_noop",
        "original_candidate_policy_id": "candidate_policy_1_roi_texture",
        "downgraded_to_weak_roi_texture": False,
        "attenuation_ratio": ATTENUATION_RATIO,
        "retained_op": None,
        "precision_aware_gate_passed": False,
        "non_active_regression_gate_passed": False,
        "weak_rejection_reasons": str(record.get("ratio_0_25_rejection_reasons") or record.get("original_rejection_reasons") or ""),
        "sampler_only_involved": False,
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_0_noop",
            "candidate_policy": {
                "policy_id": "candidate_policy_0_noop",
                "action": "no_op",
                "op_list": [],
                "image_modification": False,
                "sample_weighting": False,
            },
            "probe_set": {
                "class_id": int(record.get("active_class") or -1),
                "candidate_policy_id": "candidate_policy_0_noop",
                "mode": "image_only_weak_replay",
                "development_probe_uses_existing_val_diagnostics": False,
            },
            "causal_score": 0.0,
            "decision": {
                "candidate_policy_id": "candidate_policy_0_noop",
                "accepted": False,
                "decision": "no_op",
                "causal_score": 0.0,
                "rejection_reasons": ["weak_candidate_not_safe"],
                "image_modification_allowed": False,
                "sample_weighting_allowed": False,
                "strict_noop": True,
            },
        },
        "candidate_evaluations": [],
        "image_augmentation_rejected": True,
        "strict_image_noop": True,
        "paper_probe_mode": False,
        "development_probe_uses_existing_val_diagnostics": False,
        "final_val_used_for_policy_selection": False,
        "forbid_final_val_policy_selection": False,
        "riskguard_interpretation": {
            "riskguard_used_as_final_rule": False,
            "riskguard_role": "audit_debug_prior_only",
        },
        "seed_specific_rule": False,
        "fixed_class_id_specific_rule": False,
        "dataset_specific_rule": False,
    }


def original_op_prob(op_name: str) -> float:
    return {"local_contrast": 0.18, "sharpen_mild": 0.20, "gamma": 0.15, "clahe": 0.15}.get(str(op_name), 0.10)


def original_op_strength(op_name: str) -> float:
    return {"local_contrast": 0.20, "sharpen_mild": 0.22, "gamma": 0.18, "clahe": 0.18}.get(str(op_name), 0.10)


if __name__ == "__main__":
    main()
