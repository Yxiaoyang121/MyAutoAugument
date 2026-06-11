from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("outputs/experiments/multiseed_cp_catf_paper_mode")
REPORT_DIR = ROOT / "reports"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def active_ops(policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for class_id, class_policy in (policy.get("classes") or {}).items():
        ops = []
        for op_name, op in (class_policy.get("ops") or {}).items():
            prob = float(op.get("prob", 0.0) or 0.0)
            strength = float(op.get("strength", 0.0) or 0.0)
            if prob > 0.0 and strength > 0.0:
                ops.append({"op_name": str(op_name), "prob": prob, "strength": strength})
        if ops:
            out[str(class_id)] = ops
    return out


def history_record(run_dir: Path, epoch: int) -> dict[str, Any]:
    payload = read_json(run_dir / "reports" / "policy_history.json", {})
    records = payload.get("history", []) if isinstance(payload, dict) else payload
    for record in records or []:
        if int(record.get("epoch", -1) or -1) == int(epoch):
            return record
    return {}


def audit_seed(seed: int) -> dict[str, Any]:
    run_dir = ROOT / f"cp_catf_seed_{seed}"
    reports = run_dir / "reports"
    events_payload = read_json(reports / "causal_probe_events.json", {})
    events = events_payload.get("events", []) if isinstance(events_payload, dict) else events_payload
    online_stats = read_json(reports / "online_aug_stats.json", {})
    roi_stats = read_json(reports / "roi_aug_stats.json", {})
    rows = []
    for event in events or []:
        epoch = int(event.get("epoch", -1) or -1)
        candidate = str(event.get("selected_candidate_policy_id") or "")
        action = str(event.get("selected_candidate_action") or event.get("action") or "")
        accepted = candidate == "candidate_policy_1_roi_texture" and action == "accept"
        after_path = reports / f"policy_matrix_epoch_{epoch}_after_causal_probe.json"
        after_policy = read_json(after_path, {})
        after_active_ops = active_ops(after_policy)
        record = history_record(run_dir, epoch)
        class_policy_active_ops = active_ops(record.get("accepted_policy") or record.get("new_policy") or {})
        policy_matrix_written = after_path.exists()
        class_policy_written = bool(record)
        router_eligible = bool(after_active_ops)
        final_status = (
            "accepted_but_no_executable_policy"
            if accepted and not router_eligible
            else ("accepted_and_router_eligible" if accepted else "rejected_or_sampler_only_no_image_aug")
        )
        noop_reason = None
        if accepted and not router_eligible:
            noop_reason = "candidate accept was recorded, but no target class/op was injected into after_causal_probe policy"
        elif not accepted:
            noop_reason = str(event.get("reason") or "candidate rejected or sampler-only")
        rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "candidate_decision": candidate,
                "accepted": bool(accepted),
                "action": action,
                "active_class": event.get("candidate_class_id"),
                "op_list": event.get("op_whitelist", []),
                "policy_matrix_written": policy_matrix_written,
                "policy_matrix_active_ops": after_active_ops,
                "class_policy_written": class_policy_written,
                "class_policy_active_ops": class_policy_active_ops,
                "history_active_classes": record.get("active_classes", []),
                "history_guard_triggered": record.get("guard_triggered", []),
                "router_called": "transform_level_not_recorded",
                "router_policy_eligible": router_eligible,
                "router_candidate_samples": 0 if not router_eligible else "not_recorded_per_epoch",
                "router_selected_samples": 0 if not router_eligible else "not_recorded_per_epoch",
                "roi_aug_called": bool(router_eligible),
                "roi_aug_applied_count": int((roi_stats or {}).get("roi_aug_applied", 0) or 0),
                "industrial_aug_applied_count": int((online_stats or {}).get("samples_augmented", 0) or 0),
                "random_draw_count": int((online_stats or {}).get("router_random_draw_count", 0) or 0),
                "final_execution_status": final_status,
                "no_op_reason": noop_reason,
            }
        )
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "online_aug_stats": {
            "samples_seen": int((online_stats or {}).get("samples_seen", 0) or 0),
            "samples_augmented": int((online_stats or {}).get("samples_augmented", 0) or 0),
            "router_random_draw_count": int((online_stats or {}).get("router_random_draw_count", 0) or 0),
            "industrial_aug_enabled": bool((online_stats or {}).get("industrial_online_augmentation", False)),
            "paper_probe_mode": bool((online_stats or {}).get("paper_probe_mode", False)),
            "policy_selection_source": (online_stats or {}).get("policy_selection_source"),
            "final_val_used_for_policy_selection": (online_stats or {}).get("final_val_used_for_policy_selection"),
        },
        "roi_aug_stats": roi_stats or {},
        "events": rows,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CP-CATF Paper-Mode Accept-to-Execution Audit",
        "",
        "## Conclusion",
        "",
        f"- Accept events found: `{payload['summary']['accept_event_count']}`",
        f"- Accept events with executable policy matrix: `{payload['summary']['accept_with_executable_policy_count']}`",
        f"- Industrial samples augmented across paper-mode runs: `{payload['summary']['industrial_samples_augmented_total']}`",
        f"- ROI applied across paper-mode runs: `{payload['summary']['roi_applied_total']}`",
        f"- Router random draw count across paper-mode runs: `{payload['summary']['router_random_draw_total']}`",
        "",
        "The historical paper-mode runs recorded `candidate_policy_1_roi_texture` accept events, but the accepted candidate was not injected into the training policy matrix. The after-causal-probe policy files contain no active class-op entries, so the sample router had no executable target and bypassed before drawing any augmentation random numbers.",
        "",
        "## Per-Seed Events",
        "",
    ]
    for seed_result in payload["seeds"]:
        lines.append(f"### Seed {seed_result['seed']}")
        lines.append("")
        lines.append("| epoch | candidate | accepted | op list | policy matrix active ops | class history active classes | final status | no-op reason |")
        lines.append("|---:|---|---|---|---|---|---|---|")
        for row in seed_result["events"]:
            if row["accepted"] or row["candidate_decision"] == "candidate_policy_1_roi_texture":
                lines.append(
                    "| {epoch} | `{candidate_decision}` | `{accepted}` | `{op_list}` | `{policy_matrix_active_ops}` | `{history_active_classes}` | `{final_execution_status}` | {no_op_reason} |".format(
                        **row
                    )
                )
        lines.append("")
    lines.extend(
        [
            "## Root Cause",
            "",
            "- `apply_offline_probe_decision_to_policy()` accepted the causal-probe candidate but only applied an op whitelist.",
            "- The whitelist removed non-candidate ops, but it did not set a target class to active and did not assign nonzero prob/strength for the candidate ops.",
            "- `policy_history.json` therefore shows `active_classes=[]` for accepted epochs.",
            "- `SampleAwareAugmentationRouter.apply()` saw no `_has_active_ops(...)` target and bypassed before random draws.",
            "- RiskGuard was not the blocker, and paper-mode did not globally disable industrial augmentation; `industrial_online_augmentation=true` but no executable policy reached the router.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    seeds = [audit_seed(seed) for seed in (0, 1, 2)]
    all_events = [row for seed in seeds for row in seed["events"]]
    accept_events = [row for row in all_events if row["accepted"]]
    payload = {
        "run_root": str(ROOT),
        "summary": {
            "accept_event_count": len(accept_events),
            "accept_with_executable_policy_count": sum(1 for row in accept_events if row["router_policy_eligible"]),
            "industrial_samples_augmented_total": sum(seed["online_aug_stats"]["samples_augmented"] for seed in seeds),
            "roi_applied_total": sum(int((seed["roi_aug_stats"] or {}).get("roi_aug_applied", 0) or 0) for seed in seeds),
            "router_random_draw_total": sum(seed["online_aug_stats"]["router_random_draw_count"] for seed in seeds),
            "root_cause": "candidate accept was not materialized into active class-op policy entries",
        },
        "seeds": seeds,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "cp_catf_accept_to_execution_audit.json", payload)
    (REPORT_DIR / "cp_catf_accept_to_execution_audit.md").write_text(render_markdown(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
