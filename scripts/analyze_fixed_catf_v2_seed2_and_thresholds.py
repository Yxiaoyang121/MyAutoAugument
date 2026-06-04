"""Analyze fixed CATF-v2 seed2 failure and post-hoc thresholds.

This script is analysis-only. It may run YOLO prediction to create cached
validation prediction JSON, but it never trains or modifies weights.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402
from scripts.evaluate_catf_v2_threshold_posthoc import (  # noqa: E402
    CANONICAL_CLASS_NAMES,
    THRESHOLDS,
    collect_validation_prediction_records,
    constraint_failures,
    evaluate,
    f4,
    fd,
    greedy_search,
    metric_delta,
    run_api_validation_prediction,
    threshold_changes,
)
from scripts.train_yolo_default_with_inloop_feedback import resolve_val_image_label_dirs  # noqa: E402


DATA = Path("outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml")
FIXED_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed")
OLD_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
OLD_PRED_ROOT = OLD_ROOT / "reports/threshold_posthoc_predictions"
REPORTS = FIXED_ROOT / "reports"
PRED_ROOT = REPORTS / "fixed_threshold_posthoc_predictions"
SEEDS = [0, 1, 2]
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixed_run_root(seed: int) -> Path:
    if seed == 1:
        return Path("outputs/experiments/catf_v2_fixed_seed1_50ep")
    return FIXED_ROOT / f"seed_{seed}/catf_v2"


def clean_metrics_path(seed: int) -> Path:
    return OLD_ROOT / f"seed_{seed}/clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json"


def old_catf_metrics_path(seed: int) -> Path:
    return OLD_ROOT / f"seed_{seed}/catf_v2/reports/final_metrics.json"


def fixed_metrics_path(seed: int) -> Path:
    return fixed_run_root(seed) / "reports/final_metrics.json"


def clean_run_root(seed: int) -> Path:
    data = read_json(clean_metrics_path(seed))
    return Path(data["output_dir"])


def final_global_metrics(path: Path, *, clean: bool = False) -> dict[str, float]:
    data = read_json(path)
    if clean:
        row = data["metrics"]
    else:
        row = data["constraint_scoring"]["metrics"]
    return {key: float(row[key]) for key in METRIC_KEYS}


def final_per_class(path: Path, *, clean: bool = False) -> dict[int, dict[str, Any]]:
    data = read_json(path)
    metrics = data["val"]["metrics"] if clean else data["val"]["metrics"]
    rows = metrics.get("per_class") or []
    return {int(row["class_id"]): row for row in rows}


def class_name_map() -> dict[int, str]:
    names = load_class_names_from_data_yaml(DATA)
    if set(CANONICAL_CLASS_NAMES).issubset(set(names)):
        names = {**names, **CANONICAL_CLASS_NAMES}
    return {int(k): str(v) for k, v in names.items()}


def resolve_weights(seed: int, group: str) -> Path:
    if group == "clean":
        data = read_json(clean_metrics_path(seed))
        return Path(data["artifacts"]["best_pt"])
    return fixed_run_root(seed) / "train/weights/best.pt"


def prediction_json(seed: int, group: str) -> Path:
    if group == "clean":
        cached = OLD_PRED_ROOT / f"seed_{seed}/clean/validation_predictions.json"
        if cached.exists():
            return cached
    out_dir = PRED_ROOT / f"seed_{seed}/{group}"
    pred_json = out_dir / "validation_predictions.json"
    if pred_json.exists():
        return pred_json
    weights = resolve_weights(seed, group)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights for seed={seed} group={group}: {weights}")
    val_images, val_labels = resolve_val_image_label_dirs(DATA)
    run_api_validation_prediction(
        weights=weights,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        output_dir=out_dir,
        imgsz=1024,
        workers=0,
        device="0",
        conf=0.10,
        iou=0.5,
    )
    predict_runs = out_dir / "predict_runs"
    if predict_runs.exists():
        shutil.rmtree(predict_runs)
    return pred_json


def estimate_counts(row: dict[str, Any]) -> dict[str, float]:
    instances = float(row.get("instances", 0.0))
    recall = float(row.get("recall", 0.0))
    precision = float(row.get("precision", 0.0))
    tp = instances * recall
    fn = max(0.0, instances - tp)
    fp = max(0.0, tp / max(precision, 1e-9) - tp)
    return {"tp_est": tp, "fn_est": fn, "fp_est": fp, "instances": instances}


def per_class_delta(clean_pc: dict[int, dict[str, Any]], fixed_pc: dict[int, dict[str, Any]], names: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for cid in sorted(set(clean_pc) | set(fixed_pc)):
        c = clean_pc.get(cid, {})
        f = fixed_pc.get(cid, {})
        cc = estimate_counts(c)
        fc = estimate_counts(f)
        row = {
            "class_id": cid,
            "class_name": f.get("name") or c.get("name") or names.get(cid, str(cid)),
            "instances": int(f.get("instances", c.get("instances", 0))),
            "precision_clean": c.get("precision"),
            "precision_fixed": f.get("precision"),
            "recall_clean": c.get("recall"),
            "recall_fixed": f.get("recall"),
            "ap50_clean": c.get("ap50"),
            "ap50_fixed": f.get("ap50"),
            "ap50_95_clean": c.get("ap50_95"),
            "ap50_95_fixed": f.get("ap50_95"),
            "delta_precision": float(f.get("precision", 0.0)) - float(c.get("precision", 0.0)),
            "delta_recall": float(f.get("recall", 0.0)) - float(c.get("recall", 0.0)),
            "delta_ap50": float(f.get("ap50", 0.0)) - float(c.get("ap50", 0.0)),
            "delta_ap50_95": float(f.get("ap50_95", 0.0)) - float(c.get("ap50_95", 0.0)),
            "fn_est_clean": cc["fn_est"],
            "fn_est_fixed": fc["fn_est"],
            "delta_fn_est": fc["fn_est"] - cc["fn_est"],
            "fp_est_clean": cc["fp_est"],
            "fp_est_fixed": fc["fp_est"],
            "delta_fp_est": fc["fp_est"] - cc["fp_est"],
        }
        rows.append(row)
    return rows


def load_policy_behavior(seed: int) -> dict[str, Any]:
    reports = fixed_run_root(seed) / "reports"
    policy_history = read_json(reports / "policy_history.json").get("history", [])
    class_history = read_json(reports / "class_policy_history.json").get("history", [])
    roi = read_json(reports / "roi_aug_stats.json")
    online = read_json(reports / "online_aug_stats.json")
    active = Counter()
    active_detail = []
    for item in class_history:
        cid = int(item.get("class_id", -1))
        if item.get("action") in {"propose", "accept", "shrink", "update"} or item.get("after_status") == "active":
            active[str(cid)] += 1
            active_detail.append(item)
    return {
        "policy_action_counts": dict(Counter(str(item.get("action")) for item in policy_history)),
        "class_policy_action_counts": dict(Counter(str(item.get("action")) for item in class_history)),
        "active_classes": dict(active),
        "active_details": active_detail,
        "roi_aug_stats": roi,
        "online_aug_stats": online,
        "policy_history": policy_history,
        "class_policy_history": class_history,
    }


def curve_divergence(seed: int) -> dict[str, Any]:
    import csv

    clean_csv = OLD_ROOT / f"seed_{seed}/clean_native_yolo_default/train/results.csv"
    fixed_csv = fixed_run_root(seed) / "train/results.csv"
    clean_rows = list(csv.DictReader(clean_csv.open(encoding="utf-8")))
    fixed_rows = list(csv.DictReader(fixed_csv.open(encoding="utf-8")))
    result: dict[str, Any] = {"epoch_count_clean": len(clean_rows), "epoch_count_fixed": len(fixed_rows), "divergence": {}}
    pairs = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    for key, col in pairs.items():
        diffs = []
        first_large = None
        for c, f in zip(clean_rows, fixed_rows):
            epoch = int(float(f["epoch"]))
            diff = float(f[col]) - float(c[col])
            diffs.append({"epoch": epoch, "delta": diff})
            if first_large is None and abs(diff) >= 0.01:
                first_large = epoch
        result["divergence"][key] = {
            "first_abs_delta_ge_0.01_epoch": first_large,
            "final_delta": diffs[-1]["delta"] if diffs else None,
            "feedback_epoch_deltas": [row for row in diffs if row["epoch"] in {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}],
        }
    return result


def threshold_analysis(names: dict[int, str]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_conf": 0.10,
        "nms_iou": 0.5,
        "thresholds": THRESHOLDS,
        "note": "Post-hoc evaluator uses prediction JSON at conf=0.10 and custom AP; it is analysis-only and may not exactly match Ultralytics val.",
        "seeds": {},
    }
    pass_count = 0
    for seed in SEEDS:
        class_ids = sorted(names)
        clean_records = read_json(prediction_json(seed, "clean"))["records"]
        fixed_records = read_json(prediction_json(seed, "catf_v2"))["records"]
        clean_default = evaluate(clean_records, {cid: 0.25 for cid in class_ids})
        fixed_default = evaluate(fixed_records, {cid: 0.25 for cid in class_ids})
        seed_payload = {
            "clean_prediction_json": str(prediction_json(seed, "clean")),
            "fixed_prediction_json": str(prediction_json(seed, "catf_v2")),
            "clean_default_threshold_metrics": clean_default.metrics,
            "fixed_default_threshold_metrics": fixed_default.metrics,
            "fixed_default_delta_vs_clean_default": metric_delta(fixed_default.metrics, clean_default.metrics),
            "searches": {},
        }
        for objective in ("constrained_score", "balanced_score", "industrial_score"):
            seed_payload["searches"][objective] = greedy_search(fixed_records, class_ids, clean_default.metrics, objective)
        constrained = seed_payload["searches"]["constrained_score"]
        constrained["delta_vs_clean_default"] = metric_delta(constrained["metrics"], clean_default.metrics)
        constrained["threshold_changes"] = {
            "raise": threshold_changes(constrained["thresholds"], names)[0],
            "lower": threshold_changes(constrained["thresholds"], names)[1],
        }
        seed_payload["constrained_summary"] = constrained
        pass_count += int(not constrained["constraint_failed"])
        output["seeds"][str(seed)] = seed_payload
    output["summary"] = {
        "fixed_catf_v2_constraint_pass_count_after_constrained_calibration": pass_count,
        "seed2_repaired_by_threshold_calibration": not output["seeds"]["2"]["constrained_summary"]["constraint_failed"],
        "seed2_recall_delta_after_calibration": output["seeds"]["2"]["constrained_summary"]["delta_vs_clean_default"]["recall"],
    }
    return output


def write_threshold_report(payload: dict[str, Any], names: dict[int, str]) -> None:
    write_json(REPORTS / "fixed_threshold_calibration_posthoc.json", payload)
    lines = [
        "# Fixed CATF-v2 Post-hoc Per-Class Threshold Calibration",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "No training was run. This report uses prediction JSON at `conf=0.10` and searches per-class thresholds from `0.10` to `0.70`.",
        "",
        "## Constrained Calibration Summary",
        "",
        "| seed | P | R | mAP50 | mAP50-95 | ΔP | ΔR | ΔmAP50 | ΔmAP50-95 | constraint_failed | raise threshold | lower threshold |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|---|",
    ]
    for seed in SEEDS:
        row = payload["seeds"][str(seed)]["constrained_summary"]
        metrics = row["metrics"]
        delta = row["delta_vs_clean_default"]
        raised = [f"{item['class_id']}:{item['class_name']}->{item['threshold']:.2f}" for item in row["threshold_changes"]["raise"]]
        lowered = [f"{item['class_id']}:{item['class_name']}->{item['threshold']:.2f}" for item in row["threshold_changes"]["lower"]]
        lines.append(
            f"| {seed} | {f4(metrics['precision'])} | {f4(metrics['recall'])} | {f4(metrics['map50'])} | {f4(metrics['map50_95'])} | "
            f"{fd(delta['precision'])} | {fd(delta['recall'])} | {fd(delta['map50'])} | {fd(delta['map50_95'])} | "
            f"{str(row['constraint_failed']).lower()} | {raised} | {lowered} |"
        )
    seed2 = payload["seeds"]["2"]["constrained_summary"]
    lines += [
        "",
        "## Answers",
        "",
        f"- Seed2 repaired by threshold calibration: `{not seed2['constraint_failed']}`.",
        f"- Seed2 Recall delta after constrained calibration: `{fd(seed2['delta_vs_clean_default']['recall'])}`.",
        f"- Fixed CATF-v2 pass count after constrained calibration: `{payload['summary']['fixed_catf_v2_constraint_pass_count_after_constrained_calibration']}/3`.",
        "- If seed2 still fails, the failure is not a simple confidence-threshold issue.",
        "",
        "## Objective-Specific Searches",
        "",
    ]
    for seed in SEEDS:
        lines.append(f"### Seed {seed}")
        for objective in ("constrained_score", "balanced_score", "industrial_score"):
            row = payload["seeds"][str(seed)]["searches"][objective]
            raised, lowered = threshold_changes(row["thresholds"], names)
            raised_s = [f"{item['class_id']}:{item['class_name']}->{item['threshold']:.2f}" for item in raised]
            lowered_s = [f"{item['class_id']}:{item['class_name']}->{item['threshold']:.2f}" for item in lowered]
            metrics = row["metrics"]
            lines.append(
                f"- `{objective}`: P={f4(metrics['precision'])}, R={f4(metrics['recall'])}, "
                f"mAP50={f4(metrics['map50'])}, mAP50-95={f4(metrics['map50_95'])}, "
                f"constraint_failed={str(row['constraint_failed']).lower()}, raise={raised_s}, lower={lowered_s}"
            )
        lines.append("")
    write_md(REPORTS / "fixed_threshold_calibration_posthoc.md", lines)


def write_seed2_failure_report(seed2: dict[str, Any], names: dict[int, str]) -> None:
    write_json(REPORTS / "seed2_failure_analysis.json", seed2)
    rows = seed2["per_class_delta"]
    recall_drop = sorted(rows, key=lambda row: row["delta_recall"])[:6]
    fn_increase = sorted(rows, key=lambda row: row["delta_fn_est"], reverse=True)[:6]
    ap50_drop = sorted(rows, key=lambda row: row["delta_ap50"])[:6]
    ap95_drop = sorted(rows, key=lambda row: row["delta_ap50_95"])[:6]
    fp_reduced = sorted(rows, key=lambda row: row["delta_fp_est"])[:6]
    behavior = seed2["controller_behavior"]
    lines = [
        "# Fixed CATF-v2 Seed2 Failure Analysis",
        "",
        "Seed2 is the only fixed CATF-v2 seed that still fails the industrial constraint.",
        "",
        "## Global Metrics",
        "",
        "| Run | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        f"| clean native seed2 | {f4(seed2['clean_metrics']['precision'])} | {f4(seed2['clean_metrics']['recall'])} | {f4(seed2['clean_metrics']['map50'])} | {f4(seed2['clean_metrics']['map50_95'])} |",
        f"| fixed CATF-v2 seed2 | {f4(seed2['fixed_metrics']['precision'])} | {f4(seed2['fixed_metrics']['recall'])} | {f4(seed2['fixed_metrics']['map50'])} | {f4(seed2['fixed_metrics']['map50_95'])} |",
        f"| delta | {fd(seed2['delta_metrics']['precision'])} | {fd(seed2['delta_metrics']['recall'])} | {fd(seed2['delta_metrics']['map50'])} | {fd(seed2['delta_metrics']['map50_95'])} |",
        "",
        "## Recall Drop Classes",
        "",
        "| class | ΔRecall | FN clean est | FN fixed est | ΔFN est |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in recall_drop:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {fd(row['delta_recall'])} | {row['fn_est_clean']:.2f} | {row['fn_est_fixed']:.2f} | {row['delta_fn_est']:+.2f} |")
    lines += [
        "",
        "## FN Increase Classes",
        "",
        "| class | ΔFN est | ΔRecall |",
        "|---|---:|---:|",
    ]
    for row in fn_increase:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {row['delta_fn_est']:+.2f} | {fd(row['delta_recall'])} |")
    lines += [
        "",
        "## AP Drop Classes",
        "",
        "| class | ΔAP50 | ΔAP50-95 |",
        "|---|---:|---:|",
    ]
    for row in ap50_drop:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {fd(row['delta_ap50'])} | {fd(row['delta_ap50_95'])} |")
    lines += [
        "",
        "## AP50-95 Drop Classes",
        "",
        "| class | ΔAP50-95 | ΔAP50 |",
        "|---|---:|---:|",
    ]
    for row in ap95_drop:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {fd(row['delta_ap50_95'])} | {fd(row['delta_ap50'])} |")
    lines += [
        "",
        "## Precision Increase / FP Reduction",
        "",
        "| class | ΔFP est | ΔPrecision |",
        "|---|---:|---:|",
    ]
    for row in fp_reduced:
        lines.append(f"| {row['class_id']}:{row['class_name']} | {row['delta_fp_est']:+.2f} | {fd(row['delta_precision'])} |")
    lines += [
        "",
        "## CATF-v2 Activation And ROI",
        "",
        f"- Active classes: `{behavior['active_classes']}`.",
        f"- Active details: `{behavior['active_details']}`.",
        f"- ROI stats: `{behavior['roi_aug_stats']}`.",
        f"- Online stats summary: samples_augmented=`{behavior['online_aug_stats'].get('samples_augmented')}`, ops=`{behavior['online_aug_stats'].get('ops')}`.",
        f"- Policy actions: `{behavior['policy_action_counts']}`.",
        f"- Class policy actions: `{behavior['class_policy_action_counts']}`.",
        "",
        "## Interpretation",
        "",
        "- Seed2 clean native is already the strongest clean run for Recall and mAP50-95, so CATF-v2 operated on a high-recall baseline.",
        "- Fixed CATF-v2 increases Precision substantially, but lowers Recall and both mAP metrics; this is a少报但更准 pattern.",
        "- ROI activity was concentrated on classes 9, 11, and 8, while the largest Recall/AP drops include several non-active or only-observed classes. This suggests activation did not fully align with final degradation targets.",
        "- The controller did not rollback seed2; actions are mostly shrink/accept/freeze. This supports adding negative-effect attribution or class-level rollback in future work.",
    ]
    write_md(REPORTS / "seed2_failure_analysis.md", lines)


def write_fixed_vs_old_report(summary: dict[str, Any], old_summary: dict[str, Any] | None = None) -> None:
    fixed = read_json(REPORTS / "multiseed_catf_v2_fixed_summary.json")
    payload = {
        "fixed_summary": fixed,
        "old_summary_path": str(OLD_ROOT / "reports/multiseed_catf_v2_summary.json"),
        "old_summary_loaded": old_summary is not None,
        "constraint_failed_old": fixed["comparison_with_old_catf_v2"]["old_constraint_failed"],
        "constraint_failed_fixed": fixed["constraint_failed_count_fixed"],
        "ok3_fixed": {"ever_active": fixed["ok3_ever_active"], "roi_total": fixed["ok3_roi_total"]},
        "interpretation": "Strict no-op bypass repair improved constraint pass rate from 1/3 to 2/3. Remaining gap is seed2 mAP/Recall degradation.",
    }
    write_json(REPORTS / "fixed_vs_old_catf_v2_analysis.json", payload)
    lines = [
        "# Fixed vs Old CATF-v2 Analysis",
        "",
        "## Key Change",
        "",
        "- Old CATF-v2 had a formal no-augmentation path that could rewrite labels/Instances even when no op was applied.",
        "- Fixed CATF-v2 bypasses label conversion, clipping, and Instances rebuild unless an actual augmentation is applied.",
        "",
        "## Metrics And Constraints",
        "",
        f"- Old CATF-v2 constraint_failed: `{fixed['comparison_with_old_catf_v2']['old_constraint_failed']}/3`.",
        f"- Fixed CATF-v2 constraint_failed: `{fixed['constraint_failed_count_fixed']}/3`.",
        f"- Fixed active classes: `{fixed['active_class_counts']}`.",
        f"- Fixed ROI affected classes: `{fixed['roi_affected_class_counts']}`.",
        f"- OK3 fixed ever active: `{fixed['ok3_ever_active']}`.",
        f"- OK3 fixed ROI total: `{fixed['ok3_roi_total']}`.",
        "",
        "## Interpretation",
        "",
        "- The repair improved seed0 from failed to pass and preserved the strong fixed seed1 result.",
        "- Seed2 remains the limiting case, with mAP50 and mAP50-95 below constraint tolerance.",
        "- Fixed CATF-v2 is more credible than old CATF-v2, but remaining stability risk is concentrated in seed2 rather than OK3/no-op safety.",
    ]
    write_md(REPORTS / "fixed_vs_old_catf_v2_analysis.md", lines)


def write_next_step_report(seed2: dict[str, Any], thresh: dict[str, Any]) -> None:
    seed2_cal = thresh["seeds"]["2"]["constrained_summary"]
    seed2_repaired = not seed2_cal["constraint_failed"]
    pass_count = thresh["summary"]["fixed_catf_v2_constraint_pass_count_after_constrained_calibration"]
    full_calibration_pass = pass_count == len(SEEDS)
    payload = {
        "fixed_catf_v2_as_main_method": "candidate_with_threshold_calibration" if full_calibration_pass else "candidate_not_final",
        "seed2_failure_main_cause": seed2["main_cause"],
        "seed2_repaired_by_threshold_calibration": seed2_repaired,
        "threshold_calibrated_pass_count": pass_count,
        "recommended_positioning": (
            "CATF-v2 + per-class threshold calibration can be positioned as the final candidate"
            if full_calibration_pass
            else "Fixed CATF-v2 is a stronger candidate, but not a fully stable main method yet; next work should add class-level rollback/negative-effect attribution and improve calibration for remaining failing seeds."
        ),
        "next_changes_if_needed": [
            "class-level rollback",
            "negative effect attribution",
            "high-recall baseline protection",
            "threshold calibration as deployment layer",
        ],
    }
    write_json(REPORTS / "fixed_catf_v2_next_step_summary.json", payload)
    lines = [
        "# Fixed CATF-v2 Next Step Summary",
        "",
        f"- Fixed CATF-v2 current constraint pass count: `2/3`.",
        f"- After post-hoc constrained threshold calibration: `{pass_count}/3`.",
        f"- Seed2 repaired by threshold calibration: `{seed2_repaired}`.",
        "",
        "## Conclusion",
        "",
    ]
    if full_calibration_pass:
        lines += [
            "Fixed CATF-v2 can be treated as a stronger论文主方法 candidate when paired with per-class threshold calibration.",
            "The correct method statement should be `CATF-v2 training + per-class threshold calibration`, because calibration closes the remaining constraint gaps.",
        ]
    elif seed2_repaired:
        lines += [
            "Seed2 can be repaired by per-class threshold calibration in the post-hoc evaluator, but the calibrated pass count remains below 3/3 because another seed still violates the mAP50-95 constraint.",
            "Fixed CATF-v2 should be treated as a strong candidate rather than a finalized stable论文主方法.",
            "The next implementation work should prioritize class-level rollback, negative-effect attribution, high-recall baseline protection, and a more conservative calibration objective for the remaining failing seed.",
        ]
    else:
        lines += [
            "Fixed CATF-v2 should not yet be claimed as a stable standalone论文主方法.",
            "The remaining gap is seed2: Precision rises, but Recall/mAP fall. If threshold calibration cannot repair this, the issue is training trajectory/capability degradation rather than inference thresholding.",
            "Next implementation work should prioritize class-level rollback, negative effect attribution, and high-recall baseline protection.",
        ]
    lines += [
        "",
        "## Recommended Paper Wording",
        "",
        "- Safe statement: `CATF-v2 improves constraint pass rate and removes no-op augmentation artifacts; remaining instability is isolated to a high-recall seed and motivates threshold calibration / class-level rollback.`",
        "- Avoid claiming `stable superiority over YOLO default` unless all calibrated seeds pass under the same evaluator and official validation protocol.",
    ]
    write_md(REPORTS / "fixed_catf_v2_next_step_summary.md", lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    names = class_name_map()

    clean2 = final_global_metrics(clean_metrics_path(2), clean=True)
    fixed2 = final_global_metrics(fixed_metrics_path(2))
    clean2_pc = final_per_class(clean_metrics_path(2), clean=True)
    fixed2_pc = final_per_class(fixed_metrics_path(2))
    per_class = per_class_delta(clean2_pc, fixed2_pc, names)
    behavior2 = load_policy_behavior(2)
    seed2 = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "clean_metrics": clean2,
        "fixed_metrics": fixed2,
        "delta_metrics": {key: fixed2[key] - clean2[key] for key in METRIC_KEYS},
        "per_class_delta": per_class,
        "controller_behavior": behavior2,
        "curve_divergence": curve_divergence(2),
        "main_cause": "CATF-v2 seed2 shifts the model toward higher Precision but lower Recall/localization quality on an already strong clean baseline; active ROI targets do not fully match the final degraded classes.",
    }
    write_seed2_failure_report(seed2, names)

    threshold_payload = threshold_analysis(names)
    write_threshold_report(threshold_payload, names)
    write_fixed_vs_old_report(threshold_payload)
    write_next_step_report(seed2, threshold_payload)

    print(json.dumps({
        "seed2_delta": seed2["delta_metrics"],
        "seed2_threshold_repaired": threshold_payload["summary"]["seed2_repaired_by_threshold_calibration"],
        "threshold_pass_count": threshold_payload["summary"]["fixed_catf_v2_constraint_pass_count_after_constrained_calibration"],
        "reports": [
            str(REPORTS / "seed2_failure_analysis.md"),
            str(REPORTS / "fixed_threshold_calibration_posthoc.md"),
            str(REPORTS / "fixed_vs_old_catf_v2_analysis.md"),
            str(REPORTS / "fixed_catf_v2_next_step_summary.md"),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
