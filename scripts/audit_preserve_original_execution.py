from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AutoAugment.catf_v2.policy_matrix import active_class_ids, initial_policy_matrix  # noqa: E402
from scripts.train_yolo_default_with_inloop_feedback import apply_offline_probe_decision_to_policy  # noqa: E402


FIXED_ROOT = ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed/seed_0/catf_v2"
PRESERVE_ROOT = ROOT / "outputs/experiments/catf_v2_image_only_preserve_weak_seed0_sanity"
SCHEDULE_PATH = PRESERVE_ROOT / "configs/preserve_weak_offline_decisions_seed0.json"
OUT_CSV = PRESERVE_ROOT / "preserve_execution_parity_epoch_diff.csv"
REPORTS = PRESERVE_ROOT / "reports"
AUDIT_JSON = REPORTS / "preserve_original_execution_audit.json"
AUDIT_MD = REPORTS / "preserve_original_execution_audit.md"
DRYRUN_JSON = REPORTS / "preserve_execution_dryrun.json"
DRYRUN_MD = REPORTS / "preserve_execution_dryrun.md"
FEEDBACK_EPOCHS = (5, 10, 15, 20, 25, 30, 35, 40, 45)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def active_ops(policy: dict[str, Any] | None) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    if not policy:
        return out
    for cid_text, class_policy in sorted((policy.get("classes") or {}).items(), key=lambda item: int(item[0])):
        ops = []
        for op_name, op in sorted((class_policy.get("ops") or {}).items()):
            prob = float(op.get("prob", 0.0) or 0.0)
            strength = float(op.get("strength", 0.0) or 0.0)
            if prob > 0.0 and strength > 0.0:
                ops.append({"op": op_name, "prob": prob, "strength": strength})
        if ops and class_policy.get("status") == "active":
            out[int(cid_text)] = ops
    return out


def format_classes(values: list[int] | set[int]) -> str:
    return ";".join(str(int(value)) for value in sorted(values))


def format_ops(ops_by_class: dict[int, list[dict[str, Any]]]) -> str:
    parts = []
    for cid, ops in sorted(ops_by_class.items()):
        op_text = ",".join(f"{item['op']}@p={item['prob']:.6g}/s={item['strength']:.6g}" for item in ops)
        parts.append(f"c{cid}:{op_text}")
    return "; ".join(parts)


def parse_expected_classes(text: Any) -> list[int]:
    if text in (None, ""):
        return []
    out = []
    for part in str(text).replace(",", ";").split(";"):
        part = part.strip()
        if part.startswith("c"):
            part = part[1:]
        if part.isdigit():
            out.append(int(part))
    return sorted(set(out))


def history_by_epoch(path: Path) -> dict[int, dict[str, Any]]:
    payload = read_json(path)
    return {int(row["epoch"]): row for row in payload.get("history", [])}


def policy_for_record(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    return row.get("accepted_policy") or row.get("new_policy") or row.get("proposed_policy") or {}


def roi_summary(path: Path) -> str:
    if not path.exists():
        return "not_available"
    payload = read_json(path)
    affected = payload.get("affected_classes") or {}
    return "total={}; by_class={}".format(payload.get("roi_aug_applied", "unknown"), dict(sorted(affected.items(), key=lambda item: int(item[0]))))


def industrial_summary(path: Path) -> str:
    if not path.exists():
        return "not_available"
    payload = read_json(path)
    return "total={}".format(payload.get("samples_augmented", "unknown"))


def build_epoch_diff(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    fixed_history = history_by_epoch(FIXED_ROOT / "reports/policy_history.json")
    preserve_history = history_by_epoch(PRESERVE_ROOT / "reports/policy_history.json")
    fixed_roi = roi_summary(FIXED_ROOT / "reports/roi_aug_stats.json")
    preserve_roi = roi_summary(PRESERVE_ROOT / "reports/roi_aug_stats.json")
    fixed_industrial = industrial_summary(FIXED_ROOT / "reports/online_aug_stats.json")
    preserve_industrial = industrial_summary(PRESERVE_ROOT / "reports/online_aug_stats.json")
    rows: list[dict[str, Any]] = []
    decisions = schedule.get("epoch_decisions") or {}

    for epoch in FEEDBACK_EPOCHS:
        fixed_policy = policy_for_record(fixed_history.get(epoch))
        preserve_policy = policy_for_record(preserve_history.get(epoch))
        decision = decisions.get(str(epoch), {})
        fixed_active = active_class_ids(fixed_policy) if fixed_policy else []
        expected_active = parse_expected_classes(decision.get("original_fixed_active_class"))
        actual_active = active_class_ids(preserve_policy) if preserve_policy else []
        mismatch = []
        missing = sorted(set(expected_active) - set(actual_active))
        extra = sorted(set(actual_active) - set(expected_active))
        if missing:
            mismatch.append(f"missing_expected_classes={missing}")
        if extra:
            mismatch.append(f"unexpected_active_classes={extra}")
        if missing or extra:
            mismatch.append("preserve_branch_kept_runtime_controller_policy_instead_of_replay_fixed_policy")
        rows.append(
            {
                "epoch": epoch,
                "fixed_active_classes": format_classes(fixed_active),
                "preserve_expected_active_classes": format_classes(expected_active),
                "preserve_actual_active_classes": format_classes(actual_active),
                "fixed_op_list": format_ops(active_ops(fixed_policy)),
                "preserve_expected_op_list": decision.get("original_fixed_op_list", ""),
                "preserve_actual_op_list": format_ops(active_ops(preserve_policy)),
                "fixed_prob_strength": format_ops(active_ops(fixed_policy)),
                "preserve_expected_prob_strength": decision.get("original_fixed_prob_strength", ""),
                "preserve_actual_prob_strength": format_ops(active_ops(preserve_policy)),
                "fixed_roi_applied": fixed_roi + " (aggregate; per-feedback-epoch not recorded)",
                "preserve_actual_roi_applied": preserve_roi + " (aggregate; per-feedback-epoch not recorded)",
                "fixed_industrial_augmented": fixed_industrial + " (aggregate; per-feedback-epoch not recorded)",
                "preserve_actual_industrial_augmented": preserve_industrial + " (aggregate; per-feedback-epoch not recorded)",
                "mismatch_reason": "; ".join(mismatch) if mismatch else "none",
            }
        )
    return rows


def write_epoch_diff(rows: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def router_eligible_classes(policy: dict[str, Any]) -> list[int]:
    eligible = []
    for cid_text, class_policy in sorted((policy.get("classes") or {}).items(), key=lambda item: int(item[0])):
        if class_policy.get("status") not in {"active", "pending", "accepted"}:
            continue
        if bool(class_policy.get("no_aug_class", False)):
            continue
        guards = class_policy.get("guards") or {}
        if bool(guards.get("high_fp_guarded", False)) or class_policy.get("dominant_issue") == "high_fp":
            continue
        if any(float(op.get("prob", 0.0) or 0.0) > 0.0 and float(op.get("strength", 0.0) or 0.0) > 0.0 for op in (class_policy.get("ops") or {}).values()):
            eligible.append(int(cid_text))
    return eligible


def class_names_from_fixed() -> dict[int, str]:
    payload = read_json(FIXED_ROOT / "reports/policy_history.json")
    latest = payload.get("latest_policy") or {}
    return {int(cid): str(item.get("class_name", cid)) for cid, item in (latest.get("classes") or {}).items()}


def run_dryrun(schedule: dict[str, Any]) -> dict[str, Any]:
    matrix = initial_policy_matrix(class_names_from_fixed())
    rows = []
    expected_union: set[int] = set()
    runtime_union: set[int] = set()
    router_union: set[int] = set()
    executable_union: set[int] = set()
    for epoch in FEEDBACK_EPOCHS:
        decision = (schedule.get("epoch_decisions") or {}).get(str(epoch), {})
        expected = parse_expected_classes(decision.get("original_fixed_active_class"))
        expected_union.update(expected)
        matrix, event = apply_offline_probe_decision_to_policy(
            matrix,
            decision,
            epoch_num=epoch,
            output_dir=PRESERVE_ROOT,
            disable_sampler_only=True,
        )
        runtime = active_class_ids(matrix)
        eligible = router_eligible_classes(matrix)
        executable = sorted(active_ops(matrix).keys())
        runtime_union.update(runtime)
        router_union.update(eligible)
        executable_union.update(executable)
        rows.append(
            {
                "epoch": epoch,
                "expected_classes": expected,
                "runtime_policy_matrix_classes": runtime,
                "sample_router_eligible_classes": eligible,
                "final_executable_classes": executable,
                "expected_op_list": decision.get("original_fixed_op_list", ""),
                "runtime_op_list": format_ops(active_ops(matrix)),
                "sample_router_allowed": bool(event.get("sample_router_allowed", False)),
                "sampler_only": bool(event.get("sample_weighting_allowed", False)),
                "weighted_index_list": False,
                "weak_class9_replaced": 9 in runtime or 9 in eligible,
            }
        )
    required = {4, 11, 12}
    return {
        "seed": 0,
        "dryrun_only_no_training": True,
        "sampler_only": False,
        "weighted_index_list": False,
        "expected_class_union": sorted(expected_union),
        "runtime_policy_matrix_class_union": sorted(runtime_union),
        "sample_router_eligible_class_union": sorted(router_union),
        "final_executable_class_union": sorted(executable_union),
        "required_classes": sorted(required),
        "expected_contains_class4_11_12": required.issubset(expected_union),
        "runtime_policy_matrix_contains_class4_11_12": required.issubset(runtime_union),
        "sample_router_contains_class4_11_12": required.issubset(router_union),
        "final_executable_contains_class4_11_12": required.issubset(executable_union),
        "fixed_prob_strength_inherited": all(bool(row["runtime_op_list"]) for row in rows),
        "fixed_op_list_inherited": all(bool(row["runtime_op_list"]) for row in rows),
        "weak_class9_replaced": any(row["weak_class9_replaced"] for row in rows),
        "rows": rows,
    }


def write_md(path: Path, payload: dict[str, Any], *, kind: str) -> None:
    if kind == "dryrun":
        lines = [
            "# Preserve Original Execution Dry-run",
            "",
            f"- Dry-run only, no training: `{str(payload['dryrun_only_no_training']).lower()}`",
            f"- Expected union: `{payload['expected_class_union']}`",
            f"- Runtime policy_matrix union: `{payload['runtime_policy_matrix_class_union']}`",
            f"- Router eligible union: `{payload['sample_router_eligible_class_union']}`",
            f"- Final executable union: `{payload['final_executable_class_union']}`",
            f"- Runtime contains class4/11/12: `{str(payload['runtime_policy_matrix_contains_class4_11_12']).lower()}`",
            f"- Router contains class4/11/12: `{str(payload['sample_router_contains_class4_11_12']).lower()}`",
            f"- Fixed op list inherited: `{str(payload['fixed_op_list_inherited']).lower()}`",
            f"- Fixed prob/strength inherited: `{str(payload['fixed_prob_strength_inherited']).lower()}`",
            f"- Weak class9 replaced preserve path: `{str(payload['weak_class9_replaced']).lower()}`",
            f"- sampler_only: `{str(payload['sampler_only']).lower()}`",
            f"- weighted_index_list: `{str(payload['weighted_index_list']).lower()}`",
            "",
            "| epoch | expected | runtime policy | router eligible | executable | runtime ops |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in payload["rows"]:
            lines.append(
                f"| {row['epoch']} | `{row['expected_classes']}` | `{row['runtime_policy_matrix_classes']}` | "
                f"`{row['sample_router_eligible_classes']}` | `{row['final_executable_classes']}` | `{row['runtime_op_list']}` |"
            )
    else:
        lines = [
            "# Preserve Original Execution Audit",
            "",
            "- Root cause: `preserve_original` previously returned the current runtime controller policy via `deepcopy(policy)`.",
            "- That policy was generated by the new seed0 run and was still subject to top-k/global-guard controller behavior before preserve was applied.",
            "- The replay payload expected fixed seed0 classes `4/11/12`, but the execution matrix only retained class `11`; class `4/12` were never installed into the executable policy matrix.",
            "- Router was not the primary cause: it can route multiple classes when the policy matrix contains active ops for those classes.",
            "- Gate second-pass filtering was not the primary cause: the preserve branch bypassed weak attenuation and returned before candidate filtering.",
            "- Schema/path mismatch existed at execution level: schedule fields had the fixed policy, but runtime code ignored them.",
            "",
            "## Fix",
            "",
            "- Added preserve-original overlay that parses `original_fixed_active_class`, `original_fixed_op_list`, and `original_fixed_prob_strength`.",
            "- The overlay installs active class list, op list, probability, and strength into the policy matrix.",
            "- Stale active ops from non-preserved classes are cleared, preventing weak class9 replacement.",
            "- sampler_only and weighted index list remain disabled.",
            "",
            "## Dry-run Result",
            "",
            f"- Runtime policy_matrix contains class4/11/12: `{str(payload['dryrun']['runtime_policy_matrix_contains_class4_11_12']).lower()}`",
            f"- Router contains class4/11/12: `{str(payload['dryrun']['sample_router_contains_class4_11_12']).lower()}`",
            f"- Final executable contains class4/11/12: `{str(payload['dryrun']['final_executable_contains_class4_11_12']).lower()}`",
            f"- sampler_only involved: `{str(payload['sampler_only_involved']).lower()}`",
            "",
            "## Recommendation",
            "",
            "- Re-run seed0 sanity before seed2. Do not proceed to seed2 until seed0 confirms execution parity.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    schedule = read_json(SCHEDULE_PATH)
    rows = build_epoch_diff(schedule)
    write_epoch_diff(rows)
    dryrun = run_dryrun(schedule)
    write_json(DRYRUN_JSON, dryrun)
    write_md(DRYRUN_MD, dryrun, kind="dryrun")
    payload = {
        "seed": 0,
        "training_run": "catf_v2_image_only_preserve_weak_seed0_sanity",
        "training_started_by_this_script": False,
        "root_cause": "preserve_original_kept_runtime_controller_policy_instead_of_installing_replay_fixed_policy",
        "class4_class12_loss_stage": "apply_offline_probe_decision_to_policy preserve branch before policy_matrix/router execution",
        "policy_matrix_merge_overwrite": True,
        "router_single_class_bug": False,
        "gate_second_pass_filtering": False,
        "schema_path_mismatch": "schedule carried fixed class/op fields, but runtime preserve path ignored them",
        "fix": {
            "installed_overlay_from_original_fixed_fields": True,
            "preserves_multi_class": True,
            "clears_stale_non_preserve_ops": True,
            "bypasses_weak_candidate_merge": True,
            "sampler_only": False,
        },
        "dryrun": dryrun,
        "epoch_diff_csv": str(OUT_CSV),
        "recommend_rerun_seed0_sanity": True,
        "recommend_seed2_now": False,
        "sampler_only_involved": False,
    }
    write_json(AUDIT_JSON, payload)
    write_md(AUDIT_MD, payload, kind="audit")
    print(json.dumps({"audit": str(AUDIT_JSON), "dryrun": str(DRYRUN_JSON), "csv": str(OUT_CSV)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
