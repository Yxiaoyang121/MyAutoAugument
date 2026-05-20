from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostic_pipeline import (  # noqa: E402
    generate_candidate_policies,
    run_error_diagnosis,
    run_proxy_evaluation,
    run_validation_prediction,
)
from AutoAugment.diagnostic_pipeline.common import write_json, write_markdown  # noqa: E402
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402
from AutoAugment.utils import resolve_yolo_train_val_records  # noqa: E402


RUN_ID = "20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline"
BASELINE_ID = "20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep"
DIAGAUG_ID = "20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep"

DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
DATA_YAML = DATASET_ROOT / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID
BASELINE_DIR = PROJECT_ROOT / "outputs" / "experiments" / BASELINE_ID
DIAGAUG_DIR = PROJECT_ROOT / "outputs" / "experiments" / DIAGAUG_ID
BASELINE_BEST = BASELINE_DIR / "train" / "weights" / "best.pt"
BASELINE_METRICS = BASELINE_DIR / "reports" / "baseline_50ep_metrics.json"
DIAGAUG_SELECTED_POLICY = DIAGAUG_DIR / "short_training" / "selected_policy.json"

SEED = 42
PROXY_SAMPLES = 64
IMGSZ = 1024
WORKERS = 0
DEVICE = "0"
CONF = 0.25
IOU = 0.5
MATCH_IOU = 0.5

ISSUE_VECTOR_KEY = {
    "small_object_low_recall": "small_object_score",
    "low_contrast_missed_defect": "low_contrast_score",
    "class_imbalance": "class_imbalance_score",
    "localization_bias": "localization_score",
    "high_false_positive": "false_positive_score",
    "background_interference": "low_contrast_score",
}


def main() -> None:
    configure_yolo_environment()
    prepare_output_dir(OUTPUT_DIR)
    class_names = load_class_names_from_data_yaml(DATA_YAML)
    split = resolve_yolo_train_val_records(DATASET_ROOT, seed=SEED)
    baseline_metrics = read_json(BASELINE_METRICS)

    run_config = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "single_round_baseline_diagnosis",
        "not_multi_round_optimization": True,
        "short_training_executed": False,
        "final_training_executed": False,
        "baseline_run_id": BASELINE_ID,
        "baseline_best_pt": str(BASELINE_BEST.resolve()),
        "data_yaml": rel(DATA_YAML),
        "prediction_settings": {
            "imgsz": IMGSZ,
            "workers": WORKERS,
            "device": DEVICE,
            "conf": CONF,
            "iou": IOU,
            "match_iou": MATCH_IOU,
        },
        "proxy_settings": {
            "seed": SEED,
            "proxy_samples": PROXY_SAMPLES,
            "hard_reject_rule": "Only class id out-of-range, invalid bboxes, or image save/read failures hard reject a policy.",
            "copy_paste_rule": "copy_paste is retained for ranking; slight bbox_safe_rate or retention weakness is a soft penalty, not a hard rejection.",
        },
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)

    prediction = run_validation_prediction(
        weights=BASELINE_BEST,
        val_images_dir=split.val_images_dir,
        val_labels_dir=split.val_labels_dir,
        output_dir=OUTPUT_DIR / "validation_prediction",
        imgsz=IMGSZ,
        workers=WORKERS,
        device=DEVICE,
        conf=CONF,
        iou=IOU,
        dry_run=False,
    )
    diagnosis = run_error_diagnosis(
        val_images_dir=split.val_images_dir,
        val_labels_dir=split.val_labels_dir,
        predictions_dir=prediction["predictions_dir"],
        output_dir=OUTPUT_DIR / "diagnosis",
        class_names=class_names,
        match_iou=MATCH_IOU,
    )
    diagnosis = enrich_diagnosis(diagnosis, baseline_metrics, prediction)
    write_json(OUTPUT_DIR / "diagnosis" / "diagnosis.json", diagnosis)
    write_diagnosis_summary(OUTPUT_DIR / "diagnosis" / "diagnosis_summary.md", diagnosis)

    policies_payload = generate_candidate_policies(diagnosis, output_dir=OUTPUT_DIR / "policies", seed=SEED)
    policies_payload = enrich_policies(policies_payload)
    write_json(OUTPUT_DIR / "policies" / "candidate_policies.json", policies_payload)
    write_candidate_policies_md(OUTPUT_DIR / "policies" / "candidate_policies.md", policies_payload)

    proxy_payload = run_proxy_evaluation(
        policies_payload=policies_payload,
        train_records=split.train_records,
        output_dir=OUTPUT_DIR / "proxy",
        seed=SEED,
        proxy_samples=PROXY_SAMPLES,
        dry_run=False,
        class_names=class_names,
    )
    proxy_payload = apply_severe_only_hard_filter(proxy_payload)
    write_json(OUTPUT_DIR / "proxy" / "proxy_metrics.json", proxy_payload)
    write_json(OUTPUT_DIR / "proxy" / "proxy_ranking.json", proxy_payload["ranking"])
    write_proxy_ranking_md(OUTPUT_DIR / "proxy" / "proxy_ranking.md", proxy_payload)
    write_proxy_safety_report(OUTPUT_DIR / "proxy" / "proxy_safety_report.md", proxy_payload)

    top3_payload = build_top3_payload(diagnosis, policies_payload, proxy_payload)
    write_json(OUTPUT_DIR / "top3_policies" / "top3_policies.json", top3_payload)
    write_top3_policies_md(OUTPUT_DIR / "top3_policies" / "top3_policies.md", top3_payload)

    trace = build_policy_selection_trace(
        diagnosis=diagnosis,
        policies_payload=policies_payload,
        proxy_payload=proxy_payload,
        top3_payload=top3_payload,
        baseline_metrics=baseline_metrics,
    )
    write_json(OUTPUT_DIR / "reports" / "policy_selection_trace.json", trace)
    write_policy_selection_trace_md(OUTPUT_DIR / "reports" / "policy_selection_trace.md", trace)
    update_state_docs(trace)
    print(json.dumps(trace["summary"], ensure_ascii=False, indent=2))


def configure_yolo_environment() -> None:
    yolo_scripts = Path("D:/Anaconda/envs/pytorch/Scripts")
    if yolo_scripts.exists():
        os.environ["PATH"] = str(yolo_scripts) + os.pathsep + os.environ.get("PATH", "")
    yolo_config = PROJECT_ROOT / "outputs" / "Ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def prepare_output_dir(path: Path) -> None:
    resolved = path.resolve()
    experiments_root = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    if resolved.exists():
        if not str(resolved).startswith(str(experiments_root)):
            raise RuntimeError(f"refusing to remove output outside experiments: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def enrich_diagnosis(
    diagnosis: dict[str, Any],
    baseline_metrics: dict[str, Any],
    prediction: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(diagnosis)
    payload["baseline_run_id"] = BASELINE_ID
    payload["baseline_best_pt"] = str(BASELINE_BEST.resolve())
    payload["dataset"] = rel(DATA_YAML)
    payload["diagnosis_protocol"] = {
        "type": "single_round_baseline_diagnosis",
        "flow": "baseline best.pt -> validation prediction -> diagnosis -> candidate policies -> proxy/safety ranking -> top3",
        "not_multi_round_optimization": True,
        "short_training_executed": False,
    }
    payload["prediction_record_summary"] = {
        "predictions_dir": prediction.get("predictions_dir"),
        "image_count": prediction.get("image_count"),
        "prediction_count": prediction.get("prediction_count"),
        "conf": prediction.get("conf"),
        "iou": prediction.get("iou"),
    }
    validation = baseline_metrics["validation"]
    payload["baseline_validation"] = {
        "overall": validation["overall"],
        "per_class": validation["per_class"],
    }
    per_class_by_id = {int(item["class_id"]): item for item in validation["per_class"]}
    for class_key, item in payload.get("per_class", {}).items():
        class_id = int(item["class_id"])
        validation_item = per_class_by_id.get(class_id, {})
        item["ap50"] = float(validation_item.get("ap50", 0.0))
        item["ap50_95"] = float(validation_item.get("ap50_95", 0.0))
        item["ap"] = item["ap50"]
        item["yolo_val_precision"] = float(validation_item.get("precision", 0.0))
        item["yolo_val_recall"] = float(validation_item.get("recall", item.get("recall", 0.0)))
    vector = payload.get("diagnosis_vector", {})
    for issue in payload.get("issues", []):
        issue_type = str(issue.get("type", ""))
        vector_key = ISSUE_VECTOR_KEY.get(issue_type)
        score = None
        if vector_key:
            vector_item = vector.get(vector_key, {})
            score = vector_item.get("score") if isinstance(vector_item, dict) else vector_item
        if score is None:
            score = severity_label_to_score(str(issue.get("severity", "medium")))
        issue["severity_score"] = float(score)
    return payload


def write_diagnosis_summary(path: Path, diagnosis: dict[str, Any]) -> None:
    global_metrics = diagnosis["global"]
    baseline_overall = diagnosis["baseline_validation"]["overall"]
    lines = [
        "# Baseline Diagnosis Summary",
        "",
        "- Mode: single-round baseline diagnosis.",
        "- Flow: baseline best.pt -> prediction -> diagnosis -> candidate policies -> proxy ranking -> top3.",
        "- No top3 short-training or final training was run.",
        f"- Baseline best.pt: `{diagnosis['baseline_best_pt']}`",
        f"- Dataset: `{diagnosis['dataset']}`",
        "",
        "## Diagnosis TP/FP/FN",
        "",
        f"- TP: `{global_metrics.get('tp')}`",
        f"- FP: `{global_metrics.get('fp')}`",
        f"- FN: `{global_metrics.get('fn')}`",
        f"- Diagnosis Precision: `{float(global_metrics.get('precision', 0.0)):.4f}`",
        f"- Diagnosis Recall: `{float(global_metrics.get('recall', 0.0)):.4f}`",
        f"- Baseline val Precision/Recall: `{baseline_overall['precision']:.3f}` / `{baseline_overall['recall']:.3f}`",
        "",
        "## Diagnosis Vector",
        "",
    ]
    for key, item in diagnosis.get("diagnosis_vector", {}).items():
        lines.append(f"- {key}: `{float(item.get('score', 0.0)):.4f}` evidence=`{json.dumps(item.get('basis', {}), ensure_ascii=False)}`")
    lines.extend(["", "## Triggered Issues", ""])
    for issue in diagnosis.get("issues", []):
        lines.append(
            f"- `{issue.get('type')}` severity=`{issue.get('severity')}` "
            f"severity_score=`{float(issue.get('severity_score', 0.0)):.4f}` "
            f"evidence=`{json.dumps(issue.get('evidence', {}), ensure_ascii=False)}`"
        )
    lines.extend(
        [
            "",
            "## Per-Class AP/Recall",
            "",
            "| class id | class | TP | FP | FN | diagnosis Recall | val Recall | AP50 | AP50-95 |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in diagnosis.get("per_class", {}).values():
        lines.append(
            "| {class_id} | {class_name} | {tp} | {fp} | {fn} | {recall:.3f} | {yolo_val_recall:.3f} | {ap50:.3f} | {ap50_95:.3f} |".format(
                **item
            )
        )
    write_markdown(path, lines)


def enrich_policies(payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload)
    policies = []
    for policy in enriched.get("policies", []):
        row = dict(policy)
        row["contains_copy_paste"] = contains_copy_paste(row)
        row["operation_count"] = len(row.get("operations", []) or [])
        policies.append(row)
    enriched["policies"] = policies
    return enriched


def write_candidate_policies_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Candidate Policies From Baseline Diagnosis",
        "",
        "- Mapping: diagnosis_vector + triggered issues -> severity-driven candidate policies.",
        f"- Candidate count: `{payload.get('policy_count', len(payload.get('policies', [])))}`",
        f"- Probability formula: `{payload.get('mapping_rules', {}).get('prob_formula')}`",
        f"- Strength formula: `{payload.get('mapping_rules', {}).get('strength_formula')}`",
        "",
        "| policy_id | source_issue | severity_score | contains_copy_paste | operations | expected_effect | risk_control |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for policy in payload.get("policies", []):
        lines.append(
            "| {policy_id} | {source_issue} | {severity_score:.4f} | {contains_copy_paste} | {ops} | {effect} | {risk} |".format(
                policy_id=policy.get("policy_id"),
                source_issue=policy.get("source_issue"),
                severity_score=float(policy.get("severity_score", 0.0)),
                contains_copy_paste=bool(policy.get("contains_copy_paste")),
                ops=format_operations(policy),
                effect=escape_pipe(str(policy.get("expected_effect", ""))),
                risk=escape_pipe(str(policy.get("risk_control", ""))),
            )
        )
    lines.extend(["", "## Operation Details", ""])
    for policy in payload.get("policies", []):
        lines.extend(
            [
                f"### {policy.get('policy_id')}",
                "",
                f"- source_issue: `{policy.get('source_issue')}`",
                f"- source_issues: `{', '.join(policy.get('source_issues', []))}`",
                f"- severity_score: `{float(policy.get('severity_score', 0.0)):.4f}`",
                f"- prob_formula: `{policy.get('prob_formula')}`",
                f"- strength_formula: `{policy.get('strength_formula')}`",
                f"- contains_copy_paste: `{str(bool(policy.get('contains_copy_paste'))).lower()}`",
                "",
                "| operation | prob | strength | params |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for op in policy.get("operations", []):
            lines.append(
                f"| `{op.get('name')}` | {float(op.get('prob', 0.0)):.4f} | {float(op.get('strength', 0.0)):.4f} | `{json.dumps(op.get('params', {}), ensure_ascii=False)}` |"
            )
        lines.append("")
    write_markdown(path, lines)


def apply_severe_only_hard_filter(proxy_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(proxy_payload)
    rows = []
    for row in payload.get("metrics", []):
        updated = dict(row)
        updated["pre_relaxation_hard_filter_pass"] = bool(row.get("hard_filter_pass"))
        updated["pre_relaxation_hard_filter_reasons"] = list(row.get("hard_filter_reasons", []) or [])
        severe_reasons = severe_hard_filter_reasons(updated)
        updated["hard_filter_pass"] = not severe_reasons
        updated["hard_filter_reasons"] = severe_reasons
        updated["hard_filter_rule"] = (
            "Only class id out-of-range, invalid bboxes, or image save/read failures hard reject. "
            "bbox_safe_rate, retention, exposure, and strength risks remain soft penalties."
        )
        rows.append(updated)
    ranking = sorted(
        rows,
        key=lambda item: (
            bool(item.get("hard_filter_pass")),
            float(item.get("combined_proxy_safety_score", item.get("proxy_score", 0.0)) or 0.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    payload["metrics"] = rows
    payload["ranking"] = ranking
    payload["filter_rules"] = {
        "hard_reject": "only class id out-of-range, invalid bbox, or image save/read failure",
        "copy_paste": "copy_paste is never hard rejected solely for slight bbox_safe_rate or retention weakness",
        "soft_penalty": "bbox_safe_rate, original bbox retention, new bbox validity, exposure, class distribution, small object retention, and strength are recorded as soft safety/proxy risks",
        "final_ranking": "sort by severe hard_filter_pass then combined_proxy_safety_score",
    }
    return payload


def severe_hard_filter_reasons(row: dict[str, Any]) -> list[str]:
    reasons = []
    class_oob = int(row.get("class_out_of_range_count", 0) or 0)
    invalid_bbox = int(row.get("invalid_bbox_count", 0) or 0)
    invalid_sample = int(row.get("invalid_sample_count", 0) or 0)
    valid_images = int(row.get("valid_image_count", 0) or 0)
    image_count = int(row.get("image_count", 0) or 0)
    if class_oob > 0:
        reasons.append(f"class_out_of_range_count {class_oob} > 0")
    if invalid_bbox > 0:
        reasons.append(f"invalid_bbox_count {invalid_bbox} > 0")
    if invalid_sample > 0 or valid_images < image_count:
        reasons.append(f"image_save_or_read_failure_count {max(invalid_sample, image_count - valid_images)} > 0")
    return reasons


def write_proxy_ranking_md(path: Path, proxy_payload: dict[str, Any]) -> None:
    lines = [
        "# Proxy / Safety Ranking",
        "",
        "- No YOLO training was run.",
        "- Ranking uses proxy_score and safety_score only.",
        "- Hard rejection is limited to severe structural errors: class id out-of-range, invalid bbox, or image save/read failure.",
        "- copy_paste is not hard rejected for slight bbox safety or retention weakness; such risks are soft penalties.",
        "",
        "| rank | policy_id | source_issue | contains_copy_paste | proxy_score | safety_score | combined_score | hard_filter_pass | soft_penalties |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in proxy_payload.get("ranking", []):
        policy = row.get("policy", {})
        lines.append(
            "| {rank} | {policy_id} | {source_issue} | {cp} | {proxy:.4f} | {safety:.4f} | {combined:.4f} | {passed} | {soft} |".format(
                rank=row.get("rank"),
                policy_id=row.get("policy_id"),
                source_issue=policy.get("source_issue"),
                cp=contains_copy_paste(policy),
                proxy=float(row.get("proxy_score", 0.0) or 0.0),
                safety=float(row.get("safety_score", 0.0) or 0.0),
                combined=float(row.get("combined_proxy_safety_score", 0.0) or 0.0),
                passed=bool(row.get("hard_filter_pass")),
                soft=escape_pipe("; ".join(row.get("safety_soft_penalty_reasons", []) or [])),
            )
        )
    write_markdown(path, lines)


def write_proxy_safety_report(path: Path, proxy_payload: dict[str, Any]) -> None:
    lines = [
        "# Proxy Safety Report",
        "",
        "## Filter Contract",
        "",
        "- Severe hard reject only: class id out-of-range, invalid bboxes, or image save/read failure.",
        "- Soft penalty only: bbox_safe_rate, original bbox retention, new bbox valid rate, total bbox valid rate, small object retention, exposure, class distribution drift, and strength.",
        "- copy_paste policies are retained unless they trigger a severe structural error.",
        "",
        "## Per-Policy Safety",
        "",
        "| rank | policy_id | bbox_valid_rate | original_bbox_retention | new_bbox_valid_rate | total_bbox_valid_rate | small_object_retention | exposure_score | class_distribution_change | hard_filter_pass | hard_filter_reasons |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in proxy_payload.get("ranking", []):
        lines.append(
            "| {rank} | {policy_id} | {bbox:.4f} | {orig:.4f} | {new:.4f} | {total:.4f} | {small:.4f} | {exposure:.4f} | {dist:.4f} | {passed} | {reasons} |".format(
                rank=row.get("rank"),
                policy_id=row.get("policy_id"),
                bbox=float(row.get("bbox_valid_rate", 0.0) or 0.0),
                orig=float(row.get("original_bbox_retention", 0.0) or 0.0),
                new=float(row.get("new_bbox_valid_rate", 0.0) or 0.0),
                total=float(row.get("total_bbox_valid_rate", 0.0) or 0.0),
                small=float(row.get("small_object_retention", 0.0) or 0.0),
                exposure=float(row.get("exposure_score", 0.0) or 0.0),
                dist=float(row.get("class_distribution_change", 0.0) or 0.0),
                passed=bool(row.get("hard_filter_pass")),
                reasons=escape_pipe("; ".join(row.get("hard_filter_reasons", []) or [])),
            )
        )
        if contains_copy_paste(row.get("policy", {})):
            audit = row.get("copy_paste_audit", {}) or {}
            lines.extend(
                [
                    f"  - copy_paste_audit for `{row.get('policy_id')}`: "
                    f"new_bbox_count=`{audit.get('new_bbox_count', 0)}`, "
                    f"new_bbox_valid_rate=`{float(audit.get('new_bbox_valid_rate', 0.0) or 0.0):.4f}`, "
                    f"total_bbox_valid_rate=`{float(audit.get('total_bbox_valid_rate', 0.0) or 0.0):.4f}`, "
                    f"class_out_of_range_count=`{audit.get('class_out_of_range_count', 0)}`, "
                    f"hard_rejected=`{str(not bool(row.get('hard_filter_pass'))).lower()}`.",
                ]
            )
    write_markdown(path, lines)


def build_top3_payload(
    diagnosis: dict[str, Any],
    policies_payload: dict[str, Any],
    proxy_payload: dict[str, Any],
) -> dict[str, Any]:
    top3 = proxy_payload["ranking"][:3]
    explained = []
    for row in top3:
        policy = row.get("policy", {})
        issue = find_issue(diagnosis, str(policy.get("source_issue")))
        explanation = explain_policy(row, issue)
        explained.append({**row, "top3_reason": explanation})
    previous_selected = read_json(DIAGAUG_SELECTED_POLICY) if DIAGAUG_SELECTED_POLICY.exists() else {}
    return {
        "run_id": RUN_ID,
        "stage": "proxy_top3_policy_selection",
        "status": "completed",
        "selection_basis": "proxy_and_safety_only_no_short_training",
        "candidate_count": len(policies_payload.get("policies", [])),
        "top3": explained,
        "previous_diagaug_selected_policy": {
            "policy_id": previous_selected.get("policy_id"),
            "source_issue": previous_selected.get("source_issue"),
            "operations": previous_selected.get("operations", []),
        },
        "top1_is_diag_policy_001": bool(top3 and top3[0].get("policy_id") == "diag_policy_001"),
        "top1_matches_previous_diagaug_selected_policy": bool(
            top3 and previous_selected and top3[0].get("policy_id") == previous_selected.get("policy_id")
        ),
    }


def explain_policy(row: dict[str, Any], issue: dict[str, Any] | None) -> dict[str, Any]:
    policy = row.get("policy", {})
    source_issue = str(policy.get("source_issue"))
    operations = [str(op.get("name")) for op in policy.get("operations", [])]
    contains_cp = contains_copy_paste(policy)
    risk = policy.get("risk_control", "")
    if issue:
        evidence = issue.get("evidence", {})
        severity = issue.get("severity_score", policy.get("severity_score"))
    else:
        evidence = {"source_issues": policy.get("source_issues", [])}
        severity = policy.get("severity_score")
    audit = row.get("copy_paste_audit", {}) or {}
    return {
        "source_issue": source_issue,
        "source_evidence": evidence,
        "severity_score": severity,
        "why_suitable": why_suitable_text(source_issue, operations),
        "expected_improvement": policy.get("expected_effect"),
        "risk": risk,
        "why_proxy_safety_allows": (
            f"hard_filter_pass={row.get('hard_filter_pass')}, "
            f"combined={float(row.get('combined_proxy_safety_score', 0.0) or 0.0):.4f}, "
            f"safety={float(row.get('safety_score', 0.0) or 0.0):.4f}, "
            f"hard_reasons={row.get('hard_filter_reasons', [])}"
        ),
        "copy_paste": {
            "contains": contains_cp,
            "new_bbox_count": int(audit.get("new_bbox_count", 0) or 0) if contains_cp else 0,
            "new_bbox_valid_rate": float(audit.get("new_bbox_valid_rate", 1.0) or 1.0) if contains_cp else None,
            "hard_rejected": bool(contains_cp and not row.get("hard_filter_pass")),
        },
    }


def why_suitable_text(source_issue: str, operations: list[str]) -> str:
    if source_issue == "low_contrast_missed_defect":
        return "Baseline diagnosis shows many false negatives in low-contrast or dark regions; visibility operators match that error structure."
    if source_issue == "class_imbalance":
        return "Baseline diagnosis shows severe class count and recall imbalance; class-balanced copy_paste and mild geometry can increase minority-class exposure."
    if source_issue == "localization_bias":
        return "Baseline diagnosis contains localization-weak matches; mild translate/scale/rotate improves tolerance to small bbox shifts."
    if source_issue == "combined":
        return "The policy blends dominant diagnosis scores, especially class imbalance and low contrast, while keeping geometry conservative."
    return f"The operators {', '.join(operations)} are mapped from the triggered issue `{source_issue}`."


def write_top3_policies_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Top3 Proxy Policies From Baseline Diagnosis",
        "",
        "- Basis: proxy/safety ranking only.",
        "- No top3 short-training was run.",
        "- These top3 policies are candidates for later short-training validation, not final selected training policy.",
        "",
        "| rank | policy_id | source_issue | operations | contains_copy_paste | proxy_score | safety_score | combined_score | hard_filter_pass | reason |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["top3"]:
        policy = row.get("policy", {})
        reason = row.get("top3_reason", {})
        lines.append(
            "| {rank} | {policy_id} | {source_issue} | {ops} | {cp} | {proxy:.4f} | {safety:.4f} | {combined:.4f} | {passed} | {reason} |".format(
                rank=row.get("rank"),
                policy_id=row.get("policy_id"),
                source_issue=policy.get("source_issue"),
                ops=format_operations(policy),
                cp=contains_copy_paste(policy),
                proxy=float(row.get("proxy_score", 0.0) or 0.0),
                safety=float(row.get("safety_score", 0.0) or 0.0),
                combined=float(row.get("combined_proxy_safety_score", 0.0) or 0.0),
                passed=bool(row.get("hard_filter_pass")),
                reason=escape_pipe(reason.get("why_suitable", "")),
            )
        )
    for index, row in enumerate(payload["top3"], start=1):
        policy = row.get("policy", {})
        reason = row.get("top3_reason", {})
        cp = reason.get("copy_paste", {})
        lines.extend(
            [
                "",
                f"## {index}. {row.get('policy_id')}",
                "",
                f"1. Source diagnosis issue: `{reason.get('source_issue')}`; "
                f"source_issues: `{', '.join(policy.get('source_issues', []))}`; "
                f"evidence: `{json.dumps(reason.get('source_evidence', {}), ensure_ascii=False)}`.",
                f"2. Why suitable: {reason.get('why_suitable')}",
                f"3. Expected improvement: {reason.get('expected_improvement')}",
                f"4. Risk: {reason.get('risk')}",
                f"5. Proxy/safety allowance: {reason.get('why_proxy_safety_allows')}",
            ]
        )
        if contains_copy_paste(policy):
            lines.append(
                "6. copy_paste audit: "
                f"new_bbox_count=`{cp.get('new_bbox_count')}`, "
                f"new_bbox_valid_rate=`{float(cp.get('new_bbox_valid_rate', 0.0) or 0.0):.4f}`, "
                f"hard_rejected=`{str(cp.get('hard_rejected')).lower()}`."
            )
        else:
            lines.append("6. copy_paste audit: this policy does not contain copy_paste.")
    write_markdown(path, lines)


def build_policy_selection_trace(
    *,
    diagnosis: dict[str, Any],
    policies_payload: dict[str, Any],
    proxy_payload: dict[str, Any],
    top3_payload: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    top3 = top3_payload["top3"]
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trace_type": "single_round_baseline_diagnosis",
        "flow": [
            "baseline best.pt",
            "validation prediction",
            "diagnosis",
            "candidate policy generation",
            "proxy/safety ranking",
            "top3 proxy policies",
        ],
        "not_multi_round_optimization": True,
        "top3_short_training_executed": False,
        "final_training_executed": False,
        "selection_contract": (
            "Top3 are proxy candidates only. Formal final-policy selection requires short-training for all top3 "
            "and selection by short_train_score."
        ),
        "baseline": {
            "run_id": BASELINE_ID,
            "best_pt": str(BASELINE_BEST.resolve()),
            "data_yaml": rel(DATA_YAML),
            "metrics": baseline_metrics["validation"]["overall"],
        },
        "diagnosis": {
            "global": diagnosis.get("global", {}),
            "diagnosis_vector": diagnosis.get("diagnosis_vector", {}),
            "issues": diagnosis.get("issues", []),
        },
        "candidate_policy_count": len(policies_payload.get("policies", [])),
        "proxy_ranking_count": len(proxy_payload.get("ranking", [])),
        "top3": top3,
        "summary": {
            "issues": [issue.get("type") for issue in diagnosis.get("issues", [])],
            "candidate_policy_count": len(policies_payload.get("policies", [])),
            "top3_policy_ids": [row.get("policy_id") for row in top3],
            "top3_contains_copy_paste": [
                row.get("policy_id") for row in top3 if contains_copy_paste(row.get("policy", {}))
            ],
            "top1_is_diag_policy_001": top3_payload["top1_is_diag_policy_001"],
            "top1_matches_previous_diagaug_selected_policy": top3_payload[
                "top1_matches_previous_diagaug_selected_policy"
            ],
        },
        "artifacts": {
            "diagnosis_json": rel(OUTPUT_DIR / "diagnosis" / "diagnosis.json"),
            "diagnosis_summary_md": rel(OUTPUT_DIR / "diagnosis" / "diagnosis_summary.md"),
            "candidate_policies_json": rel(OUTPUT_DIR / "policies" / "candidate_policies.json"),
            "candidate_policies_md": rel(OUTPUT_DIR / "policies" / "candidate_policies.md"),
            "proxy_ranking_json": rel(OUTPUT_DIR / "proxy" / "proxy_ranking.json"),
            "proxy_ranking_md": rel(OUTPUT_DIR / "proxy" / "proxy_ranking.md"),
            "proxy_safety_report_md": rel(OUTPUT_DIR / "proxy" / "proxy_safety_report.md"),
            "top3_policies_json": rel(OUTPUT_DIR / "top3_policies" / "top3_policies.json"),
            "top3_policies_md": rel(OUTPUT_DIR / "top3_policies" / "top3_policies.md"),
            "policy_selection_trace_json": rel(OUTPUT_DIR / "reports" / "policy_selection_trace.json"),
            "policy_selection_trace_md": rel(OUTPUT_DIR / "reports" / "policy_selection_trace.md"),
        },
    }


def write_policy_selection_trace_md(path: Path, trace: dict[str, Any]) -> None:
    lines = [
        "# Policy Selection Trace",
        "",
        "## Scope",
        "",
        "- This is a single-round baseline diagnosis.",
        "- Flow: baseline best.pt -> diagnosis -> candidate policies -> proxy ranking -> top3.",
        "- This is not multi-round optimization.",
        "- This run did not perform top3 short-training.",
        "- Formal final strategy selection requires short-training for each top3 policy and selection by short_train_score.",
        "",
        "## Baseline",
        "",
        f"- baseline best.pt: `{trace['baseline']['best_pt']}`",
        f"- data.yaml: `{trace['baseline']['data_yaml']}`",
        f"- metrics: Precision={trace['baseline']['metrics']['precision']:.3f}, Recall={trace['baseline']['metrics']['recall']:.3f}, mAP50={trace['baseline']['metrics']['map50']:.3f}, mAP50-95={trace['baseline']['metrics']['map50_95']:.3f}",
        "",
        "## Triggered Issues",
        "",
    ]
    for issue in trace["diagnosis"]["issues"]:
        lines.append(
            f"- `{issue.get('type')}` severity_score=`{float(issue.get('severity_score', 0.0)):.4f}` evidence=`{json.dumps(issue.get('evidence', {}), ensure_ascii=False)}`"
        )
    lines.extend(
        [
            "",
            "## Top3",
            "",
            "| rank | policy_id | source_issue | operations | contains_copy_paste | combined_score |",
            "| ---: | --- | --- | --- | --- | ---: |",
        ]
    )
    for row in trace["top3"]:
        policy = row.get("policy", {})
        lines.append(
            f"| {row.get('rank')} | {row.get('policy_id')} | {policy.get('source_issue')} | {format_operations(policy)} | {contains_copy_paste(policy)} | {float(row.get('combined_proxy_safety_score', 0.0) or 0.0):.4f} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for key, value in trace["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    write_markdown(path, lines)


def update_state_docs(trace: dict[str, Any]) -> None:
    section = build_state_section(trace)
    for path in [PROJECT_ROOT / "PROJECT_STATE.md", PROJECT_ROOT / "CODEX_HANDOFF.md", PROJECT_ROOT / "EXPERIMENT_LOG.md"]:
        upsert_section(path, "BASELINE_POLICY_TOP3", section)


def build_state_section(trace: dict[str, Any]) -> str:
    top3 = ", ".join(trace["summary"]["top3_policy_ids"])
    issues = ", ".join(trace["summary"]["issues"])
    return "\n".join(
        [
            "## Baseline Diagnosis Top3 Candidate Policies",
            "",
            f"- Run ID: `{RUN_ID}`",
            f"- Baseline best.pt: `{trace['baseline']['best_pt']}`",
            f"- Dataset: `{trace['baseline']['data_yaml']}`",
            f"- Triggered issues: `{issues}`",
            f"- Candidate policies generated: `{trace['candidate_policy_count']}`",
            f"- Top3 proxy policies: `{top3}`",
            f"- Top3 containing copy_paste: `{', '.join(trace['summary']['top3_contains_copy_paste']) or 'none'}`",
            "- Scope: single-round baseline diagnosis only; not multi-round optimization.",
            "- Training status: no YOLO train, no final 50 epoch train, no top3 short-training.",
            "- Next step for final strategy selection: run short-training for all top3 and select by short_train_score.",
            f"- Trace report: `{trace['artifacts']['policy_selection_trace_md']}`",
            f"- Top3 report: `{trace['artifacts']['top3_policies_md']}`",
            f"- Proxy ranking: `{trace['artifacts']['proxy_ranking_json']}`",
        ]
    )


def upsert_section(path: Path, key: str, section: str) -> None:
    start = f"<!-- {key}_START -->"
    end = f"<!-- {key}_END -->"
    text = path.read_text(encoding="utf-8-sig", errors="replace") if path.exists() else ""
    block = f"{start}\n{section.rstrip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _match: block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def find_issue(diagnosis: dict[str, Any], source_issue: str) -> dict[str, Any] | None:
    for issue in diagnosis.get("issues", []):
        if issue.get("type") == source_issue:
            return issue
    return None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def contains_copy_paste(policy: dict[str, Any]) -> bool:
    return any(str(op.get("name", "")).lower() == "copy_paste" for op in policy.get("operations", []) or [])


def format_operations(policy: dict[str, Any]) -> str:
    return ", ".join(
        f"{op.get('name')}(p={float(op.get('prob', 0.0)):.3f}, s={float(op.get('strength', 0.0)):.3f}, params={json.dumps(op.get('params', {}), ensure_ascii=False)})"
        for op in policy.get("operations", []) or []
    )


def escape_pipe(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def severity_label_to_score(label: str) -> float:
    return {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}.get(label.lower(), 0.5)


if __name__ == "__main__":
    main()
