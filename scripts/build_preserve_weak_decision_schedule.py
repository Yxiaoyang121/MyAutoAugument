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
    / "catf_v2_image_only_preserve_weak_replay"
    / "preserve_weak_decision_records.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "catf_v2_image_only_preserve_weak_seed0_sanity"
    / "configs"
    / "preserve_weak_offline_decisions_seed0.json"
)
ATTENUATION_RATIO = 0.25
MAX_AUG_SAMPLES_PER_INTERVAL = 16


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.replay_csv), seed=int(args.seed))
    fixed_history_path = Path(args.fixed_policy_history) if args.fixed_policy_history else default_fixed_policy_history(int(args.seed))
    fixed_by_epoch = load_fixed_policy_by_epoch(fixed_history_path) if fixed_history_path else {}
    payload = build_schedule(
        records,
        seed=int(args.seed),
        replay_csv=Path(args.replay_csv),
        fixed_policy_history=fixed_history_path,
        fixed_by_epoch=fixed_by_epoch,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline preserve/weak image CATF epoch decisions.")
    parser.add_argument("--replay-csv", default=str(DEFAULT_REPLAY_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixed-policy-history",
        default="",
        help="Optional fixed CATF-v2 policy_history.json used for epoch-exact preserve_original payloads.",
    )
    return parser.parse_args()


def default_fixed_policy_history(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "experiments"
        / "multiseed_clean_yolo_default_vs_catf_v2_fixed"
        / f"seed_{int(seed)}"
        / "catf_v2"
        / "reports"
        / "policy_history.json"
    )


def load_records(path: Path, *, seed: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"preserve/weak replay CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row.get("seed") or -1)) == int(seed):
                rows.append(row)
    rows.sort(key=lambda row: int(float(row.get("epoch") or 0)))
    return rows


def load_fixed_policy_by_epoch(path: Path | None) -> dict[int, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for item in payload.get("history", []):
        epoch = int(float(item.get("epoch") or 0))
        by_epoch[epoch] = active_policy_rows(item.get("accepted_policy"))
    return by_epoch


def build_schedule(
    records: list[dict[str, Any]],
    *,
    seed: int,
    replay_csv: Path,
    fixed_policy_history: Path | None = None,
    fixed_by_epoch: dict[int, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    epoch_decisions: dict[str, dict[str, Any]] = {}
    counts = {"preserve_original": 0, "weak_roi_texture": 0, "strict_noop": 0}
    fixed_by_epoch = fixed_by_epoch or {}
    for record in records:
        action = str(record.get("final_replay_action") or "strict_noop")
        epoch = int(float(record["epoch"]))
        if action == "preserve_original":
            decision = preserve_original_payload(record, replay_csv=replay_csv, epoch_exact_rows=fixed_by_epoch.get(epoch))
        elif action == "weak_roi_texture":
            decision = weak_roi_payload(record, replay_csv=replay_csv)
        else:
            decision = strict_noop_payload(record, replay_csv=replay_csv)
            action = "strict_noop"
        counts[action] += 1
        epoch_decisions[str(epoch)] = decision
    return {
        "schema_version": 2,
        "decision_schedule_type": "image_only_preserve_weak",
        "seed": int(seed),
        "source_replay": str(replay_csv),
        "fixed_policy_history": str(fixed_policy_history) if fixed_policy_history else "",
        "epoch_exact_preserve_original": bool(fixed_by_epoch),
        "preserve_original_enabled": True,
        "weak_only_for_moderate_risk": True,
        "sampler_only_involved": False,
        "sample_weighting_allowed": False,
        "weighted_index_list_allowed": False,
        "final_val_used_for_policy_selection": False,
        "policy_selection_source": "preserve_weak_image_catf_replay",
        "attenuation_ratio": ATTENUATION_RATIO,
        "epoch_decisions": epoch_decisions,
        "summary": {
            "feedback_epochs": [int(float(row["epoch"])) for row in records],
            "preserve_original_count": counts["preserve_original"],
            "weak_roi_texture_count": counts["weak_roi_texture"],
            "strict_noop_count": counts["strict_noop"],
            "sampler_only_involved": False,
            "final_val_leakage": False,
            "epoch_exact_preserve_original": bool(fixed_by_epoch),
        },
    }


def preserve_original_payload(
    record: dict[str, Any],
    *,
    replay_csv: Path,
    epoch_exact_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    epoch = int(float(record["epoch"]))
    exact_rows = list(epoch_exact_rows or [])
    exact_active_class = active_class_text(exact_rows)
    exact_op_list = format_ops(exact_rows)
    exact_prob_strength = original_prob_strength(exact_rows)
    active_classes = parse_active_classes(exact_active_class if epoch_exact_rows is not None else record.get("original_fixed_active_class"))
    candidate = {
        "policy_id": "candidate_policy_preserve_original",
        "action": "preserve_original",
        "op_list": [],
        "image_modification": True,
        "sample_weighting": False,
        "preserve_original": True,
    }
    decision = {
        "candidate_policy_id": "candidate_policy_preserve_original",
        "accepted": True,
        "decision": "preserve_original",
        "causal_score": float(record.get("original_causal_score") or 0.0),
        "rejection_reasons": [],
        "image_modification_allowed": True,
        "sample_weighting_allowed": False,
        "strict_noop": False,
        "preserve_original": True,
    }
    payload = base_payload(record, replay_csv=replay_csv) | {
        "selected_candidate_policy_id": "candidate_policy_preserve_original",
        "selected_candidate_action": "preserve_original",
        "final_replay_action": "preserve_original",
        "preserve_original": True,
        "image_augmentation_rejected": False,
        "strict_image_noop": False,
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_preserve_original",
            "candidate_policy": candidate,
            "probe_set": {
                "class_id": active_classes[0] if active_classes else -1,
                "active_classes": active_classes,
                "candidate_policy_id": "candidate_policy_preserve_original",
                "mode": "image_only_preserve_weak_replay",
                "development_probe_uses_existing_val_diagnostics": False,
            },
            "causal_score": float(record.get("original_causal_score") or 0.0),
            "decision": decision,
        },
    }
    if epoch_exact_rows is not None:
        payload |= {
            "epoch_exact_fixed_active_class": exact_active_class,
            "epoch_exact_fixed_op_list": exact_op_list,
            "epoch_exact_fixed_prob_strength": exact_prob_strength,
            "epoch_exact_fixed_policy_empty": not bool(exact_rows),
            "epoch_exact_fixed_policy_source": "fixed_catf_v2_policy_history",
        }
    return payload


def weak_roi_payload(record: dict[str, Any], *, replay_csv: Path) -> dict[str, Any]:
    epoch = int(float(record["epoch"]))
    class_id = int(float(record.get("weak_replay_active_class") or -1))
    retained_op = str(record.get("selected_op") or "local_contrast")
    original_prob, original_strength = split_prob_strength(record.get("original_prob_strength_for_weak"), defaults=(0.18, 0.20))
    weak_prob, weak_strength = split_prob_strength(record.get("weak_prob_strength"), defaults=(original_prob * ATTENUATION_RATIO, original_strength * ATTENUATION_RATIO))
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
        "causal_score": float(record.get("original_causal_score") or 0.0),
        "rejection_reasons": [],
        "image_modification_allowed": True,
        "sample_weighting_allowed": False,
        "strict_noop": False,
        "precision_aware_gate_passed": True,
        "non_active_regression_gate_passed": True,
    }
    return base_payload(record, replay_csv=replay_csv) | {
        "selected_candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
        "selected_candidate_action": "weak_roi_texture",
        "final_replay_action": "weak_roi_texture",
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
        "image_augmentation_rejected": False,
        "strict_image_noop": False,
        "selected_candidate": {
            "candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
            "candidate_policy": candidate,
            "probe_set": {
                "class_id": class_id,
                "candidate_policy_id": "candidate_policy_1b_weak_roi_texture",
                "mode": "image_only_preserve_weak_replay",
                "development_probe_uses_existing_val_diagnostics": False,
            },
            "causal_score": float(record.get("original_causal_score") or 0.0),
            "decision": decision,
        },
    }


def strict_noop_payload(record: dict[str, Any], *, replay_csv: Path) -> dict[str, Any]:
    class_id = int(float(record.get("weak_replay_active_class") or -1))
    return base_payload(record, replay_csv=replay_csv) | {
        "selected_candidate_policy_id": "candidate_policy_0_noop",
        "selected_candidate_action": "strict_noop",
        "final_replay_action": "strict_noop",
        "original_candidate_policy_id": "candidate_policy_1_roi_texture",
        "image_augmentation_rejected": True,
        "strict_image_noop": True,
        "weak_rejection_reasons": str(record.get("noop_reason") or record.get("ratio_0_25_rejection_reasons") or record.get("original_rejection_reasons") or ""),
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
                "class_id": class_id,
                "candidate_policy_id": "candidate_policy_0_noop",
                "mode": "image_only_preserve_weak_replay",
                "development_probe_uses_existing_val_diagnostics": False,
            },
            "causal_score": 0.0,
            "decision": {
                "candidate_policy_id": "candidate_policy_0_noop",
                "accepted": False,
                "decision": "no_op",
                "causal_score": 0.0,
                "rejection_reasons": ["preserve_weak_candidate_not_safe"],
                "image_modification_allowed": False,
                "sample_weighting_allowed": False,
                "strict_noop": True,
            },
        },
    }


def base_payload(record: dict[str, Any], *, replay_csv: Path) -> dict[str, Any]:
    return {
        "epoch": int(float(record["epoch"])),
        "mode": "image_only_preserve_weak_replay",
        "decision_basis": "offline_preserve_weak_replay_no_final_val",
        "policy_selection_source": "preserve_weak_image_catf_replay",
        "policy_selection_data": str(replay_csv),
        "risk_level": str(record.get("risk_level") or ""),
        "decision_reason": str(record.get("decision_reason") or ""),
        "risk_basis": str(record.get("risk_basis") or ""),
        "original_fixed_candidate": str(record.get("original_fixed_candidate") or ""),
        "original_fixed_active_class": str(record.get("original_fixed_active_class") or ""),
        "original_fixed_op_list": str(record.get("original_fixed_op_list") or ""),
        "original_fixed_prob_strength": str(record.get("original_fixed_prob_strength") or ""),
        "sampler_only_involved": False,
        "sample_weighting_allowed": False,
        "weighted_index_list_allowed": False,
        "paper_probe_mode": False,
        "development_probe_uses_existing_val_diagnostics": False,
        "final_val_used_for_policy_selection": False,
        "forbid_final_val_policy_selection": False,
        "weak_only_for_moderate_risk": True,
        "riskguard_interpretation": {
            "riskguard_used_as_final_rule": False,
            "riskguard_role": "audit_debug_prior_only",
        },
        "seed_specific_rule": False,
        "fixed_class_id_specific_rule": False,
        "dataset_specific_rule": False,
    }


def parse_active_classes(value: Any) -> list[int]:
    out: list[int] = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def active_policy_rows(policy: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not policy:
        return []
    rows: list[dict[str, Any]] = []
    for cid_text, class_policy in sorted(policy.get("classes", {}).items(), key=lambda item: int(item[0])):
        if not is_router_executable_class_policy(class_policy):
            continue
        ops = []
        for op_name, op_cfg in sorted((class_policy.get("ops") or {}).items()):
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


def is_router_executable_class_policy(class_policy: dict[str, Any]) -> bool:
    if bool(class_policy.get("no_aug_class", False)):
        return False
    guards = class_policy.get("guards") or {}
    if bool(guards.get("high_fp_guarded", False)) or bool(guards.get("precision_guard", False)):
        return False
    status = str(class_policy.get("status") or "")
    state = str(class_policy.get("state") or "")
    return status in {"active", "accepted"} or state == "accepted"


def active_class_text(active_rows: list[dict[str, Any]]) -> str:
    return ";".join(str(row["class_id"]) for row in active_rows)


def format_ops(active_rows: list[dict[str, Any]]) -> str:
    parts = []
    for row in active_rows:
        op_text = ",".join(
            f"{op['op']}@p={op['prob']:.4g}/s={op['strength']:.4g}"
            for op in row["ops"]
        )
        parts.append(f"c{row['class_id']}:{row.get('dominant_issue') or 'unknown'}:{op_text}")
    return "; ".join(parts)


def original_prob_strength(active_rows: list[dict[str, Any]]) -> str:
    values = []
    for row in active_rows:
        for op in row["ops"]:
            values.append(f"c{row['class_id']}:{op['op']}:{op['prob']:.4g}/{op['strength']:.4g}")
    return "; ".join(values)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def split_prob_strength(value: Any, *, defaults: tuple[float, float]) -> tuple[float, float]:
    text = str(value or "").strip()
    if "/" not in text:
        return defaults
    left, right = text.split("/", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return defaults


if __name__ == "__main__":
    main()
