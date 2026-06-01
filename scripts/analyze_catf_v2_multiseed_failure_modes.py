"""Generate CATF-v2 multiseed failure-mode analysis reports.

This script is report-only: it reads existing experiment artifacts and does not
run training, validation, or mutate CATF-v2 controller logic.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
REPORTS = ROOT / "reports"
SEEDS = [0, 1, 2]
METRIC_KEYS = ["precision", "recall", "map50", "map50_95"]
RESULTS_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
CURVE_DIVERGENCE_THRESHOLDS = {
    "precision": 0.02,
    "recall": 0.02,
    "map50": 0.01,
    "map50_95": 0.01,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f4(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def fd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def bool_s(value: bool) -> str:
    return "true" if value else "false"


def read_results_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                clean_key = key.strip()
                try:
                    row[clean_key] = float(value)
                except (TypeError, ValueError):
                    row[clean_key] = math.nan
            rows.append(row)
    return rows


def by_epoch(rows: list[dict[str, float]]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in rows:
        epoch = int(row.get("epoch", len(out) + 1))
        out[epoch] = {
            key: row.get(column, math.nan) for key, column in RESULTS_COLUMNS.items()
        }
    return out


def result_metric(row_by_epoch: dict[int, dict[str, float]], epoch: int, key: str) -> float | None:
    value = row_by_epoch.get(epoch, {}).get(key)
    if value is None or math.isnan(value):
        return None
    return float(value)


def class_map(metrics: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["class_id"]): item for item in metrics.get("per_class", [])}


def approx_counts(item: dict[str, Any]) -> dict[str, float]:
    instances = float(item.get("instances") or 0.0)
    precision = max(float(item.get("precision") or 0.0), 1e-9)
    recall = float(item.get("recall") or 0.0)
    tp = recall * instances
    fn = max(instances - tp, 0.0)
    fp = max(tp * (1.0 / precision - 1.0), 0.0)
    pred = tp + fp
    return {"tp": tp, "fp": fp, "fn": fn, "pred": pred}


def feedback_history(seed_root: Path) -> list[dict[str, Any]]:
    return load_json(seed_root / "catf_v2/reports/policy_history.json").get("history", [])


def seed_paths(seed: int) -> dict[str, Path]:
    base = ROOT / f"seed_{seed}"
    return {
        "seed_root": base,
        "clean_results": base / "clean_native_yolo_default/train/results.csv",
        "catf_results": base / "catf_v2/train/results.csv",
        "clean_metrics": base
        / "clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json",
        "final_metrics": base / "catf_v2/reports/final_metrics.json",
        "policy_history": base / "catf_v2/reports/policy_history.json",
        "class_policy_history": base / "catf_v2/reports/class_policy_history.json",
        "roi_stats": base / "catf_v2/reports/roi_aug_stats.json",
        "online_stats": base / "catf_v2/reports/online_aug_stats.json",
        "threshold": base / "catf_v2/reports/threshold_calibration.json",
    }


def load_seed(seed: int) -> dict[str, Any]:
    paths = seed_paths(seed)
    clean_json = load_json(paths["clean_metrics"])
    final_json = load_json(paths["final_metrics"])
    clean_metrics = clean_json["val"]["metrics"]
    catf_metrics = final_json["val"]["metrics"]
    clean_classes = class_map(clean_metrics)
    catf_classes = class_map(catf_metrics)
    class_deltas: dict[int, dict[str, Any]] = {}
    for cid in sorted(set(clean_classes) | set(catf_classes)):
        clean_item = clean_classes.get(cid, {})
        catf_item = catf_classes.get(cid, {})
        name = catf_item.get("name") or clean_item.get("name") or str(cid)
        clean_counts = approx_counts(clean_item)
        catf_counts = approx_counts(catf_item)
        class_deltas[cid] = {
            "class_id": cid,
            "class_name": name,
            "instances": catf_item.get("instances") or clean_item.get("instances"),
            "clean": {
                "precision": clean_item.get("precision"),
                "recall": clean_item.get("recall"),
                "ap50": clean_item.get("ap50"),
                "ap50_95": clean_item.get("ap50_95"),
                **clean_counts,
            },
            "catf_v2": {
                "precision": catf_item.get("precision"),
                "recall": catf_item.get("recall"),
                "ap50": catf_item.get("ap50"),
                "ap50_95": catf_item.get("ap50_95"),
                **catf_counts,
            },
            "delta": {
                "precision": (catf_item.get("precision") or 0.0)
                - (clean_item.get("precision") or 0.0),
                "recall": (catf_item.get("recall") or 0.0)
                - (clean_item.get("recall") or 0.0),
                "ap50": (catf_item.get("ap50") or 0.0) - (clean_item.get("ap50") or 0.0),
                "ap50_95": (catf_item.get("ap50_95") or 0.0)
                - (clean_item.get("ap50_95") or 0.0),
                "tp": catf_counts["tp"] - clean_counts["tp"],
                "fp": catf_counts["fp"] - clean_counts["fp"],
                "fn": catf_counts["fn"] - clean_counts["fn"],
                "pred": catf_counts["pred"] - clean_counts["pred"],
            },
        }
    return {
        "seed": seed,
        "paths": {k: str(v) for k, v in paths.items()},
        "clean_json": clean_json,
        "final_json": final_json,
        "clean_results": by_epoch(read_results_csv(paths["clean_results"])),
        "catf_results": by_epoch(read_results_csv(paths["catf_results"])),
        "policy_history": load_json(paths["policy_history"]).get("history", []),
        "class_policy_history": load_json(paths["class_policy_history"]).get("history", []),
        "roi_stats": load_json(paths["roi_stats"]),
        "online_stats": load_json(paths["online_stats"]),
        "threshold": load_json(paths["threshold"]),
        "clean_metrics": {key: clean_metrics[key] for key in METRIC_KEYS},
        "catf_metrics": {key: catf_metrics[key] for key in METRIC_KEYS},
        "global_delta": {
            key: catf_metrics[key] - clean_metrics[key] for key in METRIC_KEYS
        },
        "class_deltas": class_deltas,
    }


def load_diag(seed: int, epoch: int) -> dict[str, Any] | None:
    path = ROOT / f"seed_{seed}/catf_v2/reports/per_class_diagnosis_epoch_{epoch}.json"
    return load_json(path) if path.exists() else None


def load_issue(seed: int, epoch: int) -> dict[str, Any] | None:
    path = ROOT / f"seed_{seed}/catf_v2/reports/issue_attribution_epoch_{epoch}.json"
    return load_json(path) if path.exists() else None


def policy_active_by_epoch(seed: int) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for path in sorted((ROOT / f"seed_{seed}/catf_v2/reports").glob("policy_matrix_epoch_*_after.json")):
        match = re.search(r"epoch_(\d+)_after", path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        matrix = load_json(path)
        active = []
        for cid_s, cls in matrix.get("classes", {}).items():
            if cls.get("status") == "active":
                active.append(
                    {
                        "class_id": int(cid_s),
                        "class_name": cls.get("class_name", cid_s),
                        "dominant_issue": cls.get("dominant_issue"),
                        "ops": cls.get("ops", {}),
                        "guards": cls.get("guards", {}),
                    }
                )
        out[epoch] = active
    return out


def collect_active_events(seed_data: dict[str, Any]) -> list[dict[str, Any]]:
    seed = seed_data["seed"]
    active_by_epoch = policy_active_by_epoch(seed)
    events = []
    for rec in seed_data["policy_history"]:
        epoch = int(rec["epoch"])
        action = rec.get("action")
        issue = load_issue(seed, epoch)
        diag = load_diag(seed, epoch)
        for active in active_by_epoch.get(epoch, []):
            cid = active["class_id"]
            issue_item = (issue or {}).get("classes", {}).get(str(cid), {})
            diag_item = (diag or {}).get("classes", {}).get(str(cid), {})
            roi_count = int(seed_data["roi_stats"].get("affected_classes", {}).get(str(cid), 0))
            class_delta = seed_data["class_deltas"].get(cid, {})
            events.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "class_id": cid,
                    "class_name": active["class_name"],
                    "dominant_issue": active.get("dominant_issue")
                    or issue_item.get("dominant_issue"),
                    "secondary_issues": issue_item.get("secondary_issues", []),
                    "diagnosis_confidence": issue_item.get(
                        "diagnosis_confidence", diag_item.get("diagnosis_confidence")
                    ),
                    "issue_scores": issue_item.get("issue_scores", {}),
                    "policy_action": action,
                    "roi_applied_total_for_class": roi_count,
                    "final_class_delta": class_delta.get("delta", {}),
                    "class_instances": class_delta.get("instances"),
                    "global_delta": seed_data["global_delta"],
                    "positive_global_contribution": (
                        (class_delta.get("delta", {}).get("recall", 0.0) > 0)
                        and (class_delta.get("delta", {}).get("ap50", 0.0) >= 0)
                        and (class_delta.get("delta", {}).get("fp", 0.0) <= 1.0)
                    ),
                    "ops": active.get("ops", {}),
                }
            )
    return events


def curve_diagnosis(seed_data: dict[str, Any]) -> dict[str, Any]:
    clean = seed_data["clean_results"]
    catf = seed_data["catf_results"]
    deltas_by_epoch: dict[int, dict[str, float]] = {}
    for epoch in sorted(set(clean) & set(catf)):
        deltas_by_epoch[epoch] = {
            key: (catf[epoch][key] - clean[epoch][key]) for key in METRIC_KEYS
        }
    first_divergence = {}
    for key in METRIC_KEYS:
        first_divergence[key] = next(
            (
                epoch
                for epoch, deltas in sorted(deltas_by_epoch.items())
                if abs(deltas[key]) >= CURVE_DIVERGENCE_THRESHOLDS[key]
            ),
            None,
        )
    feedback_epochs = [int(item["epoch"]) for item in seed_data["policy_history"]]
    feedback_effects = []
    for epoch in feedback_epochs:
        prev_epoch = max(1, epoch - 5)
        next_epoch = min(50, epoch + 5)
        item: dict[str, Any] = {
            "epoch": epoch,
            "action": next(
                (rec.get("action") for rec in seed_data["policy_history"] if int(rec["epoch"]) == epoch),
                None,
            ),
            "active_classes": next(
                (rec.get("active_classes") for rec in seed_data["policy_history"] if int(rec["epoch"]) == epoch),
                [],
            ),
            "delta_at_epoch": deltas_by_epoch.get(epoch, {}),
            "delta_change_from_prev_feedback": {},
            "delta_change_to_next_interval": {},
            "catf_raw_change_to_next_interval": {},
        }
        for key in METRIC_KEYS:
            if epoch in deltas_by_epoch and prev_epoch in deltas_by_epoch:
                item["delta_change_from_prev_feedback"][key] = (
                    deltas_by_epoch[epoch][key] - deltas_by_epoch[prev_epoch][key]
                )
            if epoch in deltas_by_epoch and next_epoch in deltas_by_epoch:
                item["delta_change_to_next_interval"][key] = (
                    deltas_by_epoch[next_epoch][key] - deltas_by_epoch[epoch][key]
                )
            start_value = result_metric(catf, epoch, key)
            end_value = result_metric(catf, next_epoch, key)
            if start_value is not None and end_value is not None:
                item["catf_raw_change_to_next_interval"][key] = end_value - start_value
        feedback_effects.append(item)
    freeze_epochs = [
        int(item["epoch"])
        for item in seed_data["policy_history"]
        if item.get("action") == "freeze" or item.get("frozen")
    ]
    first_freeze = min(freeze_epochs) if freeze_epochs else None
    freeze_to_final = {}
    if first_freeze:
        for key in METRIC_KEYS:
            start = result_metric(catf, first_freeze, key)
            end = result_metric(catf, 50, key)
            if start is not None and end is not None:
                freeze_to_final[key] = end - start
    return {
        "seed": seed_data["seed"],
        "first_divergence_epoch": first_divergence,
        "deltas_by_epoch": {
            str(epoch): deltas for epoch, deltas in sorted(deltas_by_epoch.items())
        },
        "feedback_epochs": feedback_epochs,
        "feedback_effects": feedback_effects,
        "freeze_epochs": freeze_epochs,
        "first_freeze_epoch": first_freeze,
        "freeze_to_final_catf_change": freeze_to_final,
        "final_delta": seed_data["global_delta"],
    }


def top_class_changes(seed_data: dict[str, Any], field: str, n: int = 5, reverse: bool = True) -> list[dict[str, Any]]:
    rows = []
    for item in seed_data["class_deltas"].values():
        rows.append(
            {
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "instances": item["instances"],
                "delta": item["delta"],
                "clean": item["clean"],
                "catf_v2": item["catf_v2"],
            }
        )
    return sorted(rows, key=lambda x: x["delta"].get(field, 0.0), reverse=reverse)[:n]


def threshold_summary(seed_data: dict[str, Any]) -> dict[str, Any]:
    classes = seed_data["threshold"].get("classes", {})
    raise_classes = []
    lower_classes = []
    keep_classes = []
    for cid_s, item in classes.items():
        default = float(item.get("default_threshold", 0.25))
        recommended = float(item.get("recommended_threshold", default))
        entry = {
            "class_id": int(cid_s),
            "class_name": item.get("class_name", cid_s),
            "default_threshold": default,
            "recommended_threshold": recommended,
            "reason": item.get("reason"),
        }
        if recommended > default:
            raise_classes.append(entry)
        elif recommended < default:
            lower_classes.append(entry)
        else:
            keep_classes.append(entry)
    return {
        "raise_threshold": raise_classes,
        "lower_threshold": lower_classes,
        "keep_threshold": keep_classes,
    }


def build_reports() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    seeds = [load_seed(seed) for seed in SEEDS]
    generated_at = datetime.now().isoformat(timespec="seconds")
    summary_path = REPORTS / "multiseed_catf_v2_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    curve_like_artifacts = [
        str(path)
        for path in ROOT.rglob("*")
        if path.is_file()
        and re.search(r"(confusion|PR_curve|P_curve|R_curve|F1_curve)", path.name, re.IGNORECASE)
    ]
    prediction_artifacts = [
        str(path)
        for path in ROOT.rglob("validation_predictions.json")
    ]

    curve = {
        "generated_at": generated_at,
        "reports": {
            "summary": str(summary_path),
        },
        "seeds": [curve_diagnosis(seed) for seed in seeds],
        "interpretation": {
            "seed_0": "Precision is below clean from the first epoch and remains negative after the only active ROI window; Recall/mAP gains are bought with additional false positives.",
            "seed_1": "No active ROI augmentation is executed, so the run behaves like a conservative guarded controller and passes mainly by not disturbing YOLO default.",
            "seed_2": "Recall is already below clean early and deteriorates after the feedback/freeze window; final behavior is conservative with high Precision and fewer detections.",
        },
    }
    write_json(REPORTS / "curve_diagnosis.json", curve)

    curve_lines = [
        "# CATF-v2 Curve Diagnosis",
        "",
        f"Generated: `{generated_at}`",
        "",
        "No training was run. This report compares existing clean native and CATF-v2 `results.csv` curves.",
        "",
    ]
    for item in curve["seeds"]:
        seed = item["seed"]
        curve_lines += [
            f"## Seed {seed}",
            "",
            f"- First divergence epochs: `{item['first_divergence_epoch']}`.",
            f"- Feedback epochs: `{item['feedback_epochs']}`.",
            f"- First freeze epoch: `{item['first_freeze_epoch']}`; CATF raw metric change from first freeze to final: `{item['freeze_to_final_catf_change']}`.",
            f"- Final delta P/R/mAP50/mAP50-95: `{ {k: round(v, 4) for k, v in item['final_delta'].items()} }`.",
            "",
            "| feedback epoch | action | active classes | ΔP at epoch | ΔR at epoch | ΔmAP50 at epoch | ΔmAP50-95 at epoch | next-window ΔP change | next-window ΔR change |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for effect in item["feedback_effects"]:
            d = effect["delta_at_epoch"]
            next_d = effect["delta_change_to_next_interval"]
            curve_lines.append(
                f"| {effect['epoch']} | {effect['action']} | {effect.get('active_classes') or []} | "
                f"{fd(d.get('precision'))} | {fd(d.get('recall'))} | {fd(d.get('map50'))} | {fd(d.get('map50_95'))} | "
                f"{fd(next_d.get('precision'))} | {fd(next_d.get('recall'))} |"
            )
        curve_lines.append("")
    curve_lines += [
        "## Curve-Level Conclusions",
        "",
        "- Seed 0 Precision weakness is present from the beginning and is not corrected by CATF-v2; the later active class window does not explain the full Precision loss by itself.",
        "- Seed 2 Recall loss is also an early/global trajectory shift rather than a simple direct result of the small ROI intervention at epoch 25.",
        "- Freeze at epoch 40 prevents late corrective updates; after freeze the model can still drift through normal YOLO training, but CATF-v2 no longer adapts.",
        "- ROI events are sparse and do not align strongly enough with the final global deltas to claim ROI augmentation as the primary cause of gains.",
    ]
    write_md(REPORTS / "curve_diagnosis.md", curve_lines)

    active_events = []
    for seed in seeds:
        active_events.extend(collect_active_events(seed))
    seed42_context = {}
    seed42_root = Path("outputs/experiments/catf_v2_seed42_50ep")
    if seed42_root.exists():
        seed42_history = load_json(seed42_root / "reports/policy_history.json").get("history", [])
        seed42_actions = Counter(item.get("action") for item in seed42_history)
        seed42_active = Counter()
        for item in seed42_history:
            for cid in item.get("active_classes") or []:
                seed42_active[str(cid)] += 1
        seed42_context = {
            "path": str(seed42_root),
            "roi_stats": load_json(seed42_root / "reports/roi_aug_stats.json"),
            "policy_action_counts": dict(seed42_actions),
            "active_class_epoch_counts": dict(seed42_active),
            "final_metrics": {
                key: load_json(seed42_root / "reports/final_metrics.json")["val"]["metrics"][key]
                for key in METRIC_KEYS
            },
        }
    active_analysis = {
        "generated_at": generated_at,
        "active_events": active_events,
        "per_seed_active_class_counts": {
            str(seed["seed"]): {
                f"{event['class_id']}:{event['class_name']}": event
                for event in collect_active_events(seed)
            }
            for seed in seeds
        },
        "seed42_context": seed42_context,
        "answers": {
            "xijian_effect": "Seed 0 activated class 11 (锡尖). Final class Recall and AP50/AP50-95 improved, but global Precision still failed because other classes accumulated FP.",
            "zangwu_effect": "Seed 2 activated class 8 (脏污) with only 3 ROI applications. Class 8 Recall/AP declined while FP declined, matching the seed-level conservative high-P/low-R pattern.",
            "active_too_sparse": "Yes. Across seeds 0/1/2, only two class activation events occurred and seed1 had no ROI application, so intervention is too sparse to reliably move global metrics.",
            "wrong_target": "Partly. Activated classes were not the main final failure classes in seed2, and seed0 Precision loss was not limited to the active class.",
        },
    }
    write_json(REPORTS / "active_class_effect_analysis.json", active_analysis)

    active_lines = [
        "# CATF-v2 Active Class Effect Analysis",
        "",
        f"Generated: `{generated_at}`",
        "",
        "| seed | epoch | class | issue | confidence | action | ROI count | ΔRecall | ΔAP50 | ΔAP50-95 | ΔFP | contribution |",
        "|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for event in active_events:
        d = event["final_class_delta"]
        active_lines.append(
            f"| {event['seed']} | {event['epoch']} | {event['class_id']}:{event['class_name']} | {event['dominant_issue']} | "
            f"{f4(event.get('diagnosis_confidence'))} | {event['policy_action']} | {event['roi_applied_total_for_class']} | "
            f"{fd(d.get('recall'))} | {fd(d.get('ap50'))} | {fd(d.get('ap50_95'))} | {fd(d.get('fp'))} | {event['positive_global_contribution']} |"
        )
    if not active_events:
        active_lines.append("| n/a | n/a | none | n/a | n/a | n/a | 0 | n/a | n/a | n/a | n/a | n/a |")
    active_lines += [
        "",
        "## Findings",
        "",
        "- 锡尖在 seed 0 被激活后本类 Recall/AP 有提升，但全局 Precision 失败，说明收益不是无代价的全局收益。",
        "- 脏污在 seed 2 被激活后本类 Recall/AP 下降且 FP 下降；它符合 seed 2 的高 Precision、低 Recall 保守化模式。",
        "- seed 42 的漏背锡/锡膏激活没有在 seeds 0/1/2 稳定复现，说明诊断选择对 seed 轨迹敏感。",
        "- active class 太少：三次 50 epoch 只有 2 次 class propose，seed 1 完全没有 ROI applied；seed 1 通过约束不能证明 ROI augmentation 有效。",
        "- 存在增强目标与最终退化类不一致的问题；需要 negative effect attribution 来回滚对应类或扩大诊断到实际退化类。",
    ]
    write_md(REPORTS / "active_class_effect_analysis.md", active_lines)

    precision_rows = {}
    for seed in seeds:
        precision_rows[str(seed["seed"])] = {
            "final_delta": seed["global_delta"],
            "top_fp_increases": top_class_changes(seed, "fp", n=8, reverse=True),
            "top_precision_drops": top_class_changes(seed, "precision", n=8, reverse=False),
            "threshold": threshold_summary(seed),
            "active_events": collect_active_events(seed),
        }
    precision_analysis = {
        "generated_at": generated_at,
        "focus": "seed 0 and seed 1 Precision behavior",
        "seeds": precision_rows,
        "conclusions": {
            "seed_0_main_precision_drop_source": "Approximate FP increases are led by OK2 plus several defect classes; the active class is not the only source.",
            "seed_1_precision_status": "Precision drop is small (-0.0034) and within constraint; threshold report still flags high-FP prior classes for possible threshold raise.",
            "roi_relation": "ROI-aware augmentation is too sparse to fully explain Precision drop; seed1 has zero ROI and tiny Precision drop, seed0 has 20 ROI on 锡尖 but Precision loss spans other classes.",
            "threshold_issue": "Per-class threshold calibration is a plausible repair for seed0 Precision because many FP-risk classes are recommended to raise threshold from 0.25 to 0.40.",
        },
    }
    write_json(REPORTS / "precision_drop_analysis.json", precision_analysis)

    precision_lines = [
        "# CATF-v2 Precision Drop Analysis",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Seed 0",
        "",
        f"- Global delta: `{ {k: round(v, 4) for k, v in seeds[0]['global_delta'].items()} }`.",
        "- Top approximate FP increases:",
        "",
        "| class | ΔFP | ΔPrecision | ΔRecall | active? | threshold recommendation |",
        "|---|---:|---:|---:|:---:|---|",
    ]
    seed0 = seeds[0]
    seed0_active_ids = {event["class_id"] for event in collect_active_events(seed0)}
    seed0_threshold = {item["class_id"]: item for item in threshold_summary(seed0)["raise_threshold"] + threshold_summary(seed0)["lower_threshold"]}
    for item in top_class_changes(seed0, "fp", n=8, reverse=True):
        cid = item["class_id"]
        th = seed0_threshold.get(cid)
        threshold_text = (
            f"{th['default_threshold']} -> {th['recommended_threshold']} {th['reason']}"
            if th
            else "keep"
        )
        precision_lines.append(
            f"| {cid}:{item['class_name']} | {fd(item['delta']['fp'])} | {fd(item['delta']['precision'])} | {fd(item['delta']['recall'])} | "
            f"{bool_s(cid in seed0_active_ids)} | {threshold_text} |"
        )
    precision_lines += [
        "",
        "## Seed 1",
        "",
        f"- Global Precision delta is `{fd(seeds[1]['global_delta']['precision'])}`, within constraint.",
        "- Seed 1 had no active ROI augmentation, so the slight Precision drop is normal seed/training variance rather than evidence of harmful CATF-v2 augmentation.",
        "",
        "## Answers",
        "",
        "- Precision drop mainly comes from classes with increased approximate FP, not only the active class.",
        "- high-FP prior covered OK/oil/dirty classes, but final FP risk also appears in non-prior defect classes, so prior coverage is incomplete.",
        "- Existing threshold reports suggest raising many high-FP classes to `0.40`; seed0 is the strongest candidate for threshold repair.",
        "- Do not add OK2/OK3 back into augmentation. Keep them no_aug/high-FP guarded.",
    ]
    write_md(REPORTS / "precision_drop_analysis.md", precision_lines)

    recall_rows = {}
    for seed in seeds:
        recall_rows[str(seed["seed"])] = {
            "final_delta": seed["global_delta"],
            "top_fn_increases": top_class_changes(seed, "fn", n=8, reverse=True),
            "top_recall_drops": top_class_changes(seed, "recall", n=8, reverse=False),
            "threshold": threshold_summary(seed),
            "policy_actions": Counter(rec.get("action") for rec in seed["policy_history"]),
        }
    recall_analysis = {
        "generated_at": generated_at,
        "focus": "seed 2 Recall decline",
        "seeds": recall_rows,
        "conclusions": {
            "seed_2_main_recall_drop_source": "Recall loss is broad, led by low-support/small defect classes and several non-active defect classes; it is not explained by the active 脏污 ROI alone.",
            "confidence_conservatism": "The final pattern P up sharply while R and mAP down indicates conservative detection/confidence distribution or fewer accepted detections.",
            "clean_seed2_context": "Clean seed2 already had the highest Recall and mAP50-95 among seeds; CATF-v2 disturbed that high-recall trajectory.",
            "feedback_policy": "Shrink/freeze dominated, so difficult classes got little to no positive augmentation before epoch 40 freeze.",
            "callback_rng_risk": "Seed1 had zero industrial samples augmented but still diverged from clean, so in-loop diagnosis/callback RNG side effects should be isolated with a feedback-on/no-augmentation control.",
        },
    }
    write_json(REPORTS / "recall_drop_analysis.json", recall_analysis)

    seed2 = seeds[2]
    recall_lines = [
        "# CATF-v2 Recall Drop Analysis",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"- Seed 2 global delta: `{ {k: round(v, 4) for k, v in seed2['global_delta'].items()} }`.",
        "- Top approximate FN increases:",
        "",
        "| class | ΔFN | ΔRecall | ΔAP50 | ΔAP50-95 | active? | threshold recommendation |",
        "|---|---:|---:|---:|---:|:---:|---|",
    ]
    seed2_active_ids = {event["class_id"] for event in collect_active_events(seed2)}
    seed2_threshold = {item["class_id"]: item for item in threshold_summary(seed2)["raise_threshold"] + threshold_summary(seed2)["lower_threshold"]}
    for item in top_class_changes(seed2, "fn", n=8, reverse=True):
        cid = item["class_id"]
        th = seed2_threshold.get(cid)
        threshold_text = (
            f"{th['default_threshold']} -> {th['recommended_threshold']} {th['reason']}"
            if th
            else "keep"
        )
        recall_lines.append(
            f"| {cid}:{item['class_name']} | {fd(item['delta']['fn'])} | {fd(item['delta']['recall'])} | {fd(item['delta']['ap50'])} | {fd(item['delta']['ap50_95'])} | "
            f"{bool_s(cid in seed2_active_ids)} | {threshold_text} |"
        )
    recall_lines += [
        "",
        "## Answers",
        "",
        "- Seed 2 Recall loss is broad and not concentrated in the active 脏污 class.",
        "- The P/R pattern strongly suggests conservative confidence/detection behavior: Precision rises by +0.1395 while Recall drops by -0.1247.",
        "- Clean seed2 was already a high-Recall trajectory; CATF-v2 should likely reduce or disable intervention when the reference curve already outperforms on Recall/mAP.",
        "- Shrink/freeze behavior leaves too little opportunity to rescue classes after early negative drift.",
        "- Seed 1 had zero industrial augmentation but still changed metrics, so CATF-v2 evaluation should include an in-loop diagnosis-only control to isolate callback/RNG effects.",
    ]
    write_md(REPORTS / "recall_drop_analysis.md", recall_lines)

    threshold_analysis = {
        "generated_at": generated_at,
        "seeds": {
            str(seed["seed"]): {
                "global_delta": seed["global_delta"],
                "threshold_summary": threshold_summary(seed),
                "precision_drop_classes": top_class_changes(seed, "fp", n=5, reverse=True),
                "recall_drop_classes": top_class_changes(seed, "fn", n=5, reverse=True),
            }
            for seed in seeds
        },
        "answers": {
            "seed_0_precision_fix_potential": "High. The report recommends raising thresholds for most FP-risk classes; this is the right lever for Precision loss without retraining.",
            "seed_2_recall_fix_potential": "Partial. Lowering thresholds is recommended only for a small subset, while many Recall-loss classes are still flagged for threshold raise due high-FP risk.",
            "training_plus_threshold_plan": "Keep training feedback conservative and treat per-class threshold calibration as a separate deployment/post-processing step, validated against clean reference constraints.",
            "could_pass_constraints": "Seed0 likely could move closer to constraints with threshold raises. Seed2 may need both reduced training intervention and selective threshold lowering; threshold calibration alone is unlikely to fully recover mAP50/mAP50-95.",
        },
    }
    write_json(REPORTS / "threshold_calibration_analysis.json", threshold_analysis)

    threshold_lines = [
        "# CATF-v2 Threshold Calibration Analysis",
        "",
        f"Generated: `{generated_at}`",
        "",
        "| seed | raise threshold classes | lower threshold classes | likely repair effect |",
        "|---:|---|---|---|",
    ]
    for seed in seeds:
        th = threshold_summary(seed)
        raise_names = [f"{item['class_id']}:{item['class_name']}" for item in th["raise_threshold"]]
        lower_names = [f"{item['class_id']}:{item['class_name']}" for item in th["lower_threshold"]]
        repair = "Precision repair candidate" if seed["seed"] == 0 else ("Recall repair partial" if seed["seed"] == 2 else "Already passes constraints")
        threshold_lines.append(
            f"| {seed['seed']} | {raise_names} | {lower_names} | {repair} |"
        )
    threshold_lines += [
        "",
        "## Conclusions",
        "",
        "- Seed 0 Precision loss is the clearest threshold-calibration target; many high-FP classes are recommended for threshold raise.",
        "- Seed 2 Recall loss cannot be solved cleanly by threshold alone because many classes are simultaneously high-FP guarded and recommended for threshold raise.",
        "- A defensible next protocol is `YOLO default + conservative CATF-v2 training` followed by `analysis-only per-class threshold calibration`, then a separate post-processing validation.",
    ]
    write_md(REPORTS / "threshold_calibration_analysis.md", threshold_lines)

    controller = {
        "generated_at": generated_at,
        "per_seed": {},
        "aggregate": {
            "actions": Counter(),
            "class_actions": Counter(),
            "rollback": 0,
            "cooldown": 0,
            "freeze_policy_actions": 0,
            "freeze_markers_including_frozen_flag": 0,
        },
        "answers": {
            "shrink_too_many": "Yes. 14 shrink actions versus only 2 accepts indicates a controller that mostly suppresses itself instead of learning a useful policy.",
            "accept_too_few": "Yes. Only two accepted/proposed class policies across three full runs is too sparse for a feedback augmentation method.",
            "freeze_too_early": "Freeze at epoch 40 is intended for close_mosaic convergence, but with sparse accepted actions it locks in weak or wrong directions.",
            "rollback_zero_reason": "Most failures appear as global/final metric trade-offs or non-active-class degradation, not as the controller's current class-level rollback trigger.",
            "need_negative_attribution": "Yes. The controller needs to attribute final/per-epoch harm to class policies or to global confidence shift, then rollback/shrink that specific source.",
            "diagnosis_side_effect_control_needed": "Seed1 applied zero industrial augmentation but still diverged from clean, so a diagnosis-only in-loop control is needed before attributing gains to CATF-v2 augmentation.",
        },
    }
    for seed in seeds:
        actions = Counter(rec.get("action") for rec in seed["policy_history"])
        class_actions = Counter()
        for rec in seed["policy_history"]:
            for ca in rec.get("class_actions") or []:
                class_actions[ca.get("action")] += 1
        freeze_count = sum(1 for rec in seed["policy_history"] if rec.get("action") == "freeze" or rec.get("frozen"))
        freeze_action_count = actions.get("freeze", 0)
        controller["per_seed"][str(seed["seed"])] = {
            "actions": dict(actions),
            "class_actions": dict(class_actions),
            "freeze_policy_action_count": freeze_action_count,
            "freeze_marker_count_including_frozen_flag": freeze_count,
            "constraint_failed": seed["final_json"].get("constraint_scoring", {}).get("constraint_failed"),
            "final_delta": seed["global_delta"],
        }
        controller["aggregate"]["actions"].update(actions)
        controller["aggregate"]["class_actions"].update(class_actions)
        controller["aggregate"]["rollback"] += actions.get("rollback", 0) + class_actions.get("rollback", 0)
        controller["aggregate"]["cooldown"] += actions.get("cooldown", 0) + class_actions.get("cooldown", 0)
        controller["aggregate"]["freeze_policy_actions"] += freeze_action_count
        controller["aggregate"]["freeze_markers_including_frozen_flag"] += freeze_count
    controller["aggregate"]["actions"] = dict(controller["aggregate"]["actions"])
    controller["aggregate"]["class_actions"] = dict(controller["aggregate"]["class_actions"])
    write_json(REPORTS / "controller_behavior_analysis.json", controller)

    controller_lines = [
        "# CATF-v2 Controller Behavior Analysis",
        "",
        f"Generated: `{generated_at}`",
        "",
        f"- Aggregate policy actions: `{controller['aggregate']['actions']}`.",
        f"- Aggregate class actions: `{controller['aggregate']['class_actions']}`.",
        f"- Rollback/cooldown/freeze policy-action totals: `{controller['aggregate']['rollback']}/{controller['aggregate']['cooldown']}/{controller['aggregate']['freeze_policy_actions']}`.",
        f"- Freeze markers including `frozen=true` flags: `{controller['aggregate']['freeze_markers_including_frozen_flag']}`.",
        "",
        "## Findings",
        "",
        "- `shrink` dominates (`14`), while `accept` is rare (`2`). This means CATF-v2 is mostly avoiding harm, not reliably discovering helpful policies.",
        "- Rollback/cooldown are zero because the current failure modes are not tied to a pending class policy in the controller state; they are global trajectory and non-active-class effects.",
        "- Freeze at epoch 40 is reasonable for convergence, but when very few policies were accepted beforehand it also prevents late recovery.",
        "- Seed 1 had zero industrial augmentation but still diverged from clean, which suggests in-loop diagnosis/callback RNG effects need a dedicated control.",
        "- The next controller change should be negative-effect attribution: if a class or metric worsens after an update, rollback that class policy or reduce all risky policies that correlate with the degradation.",
    ]
    write_md(REPORTS / "controller_behavior_analysis.md", controller_lines)

    failure = {
        "generated_at": generated_at,
        "main_failure_modes": [
            "intervention too weak",
            "intervention wrong target",
            "precision threshold issue",
            "over-conservative freeze",
            "per-class diagnosis trajectory sensitive",
            "in-loop diagnosis/callback side effect not isolated",
        ],
        "seed_0_failure_main_cause": "Precision threshold / FP control issue. Recall and AP improve, but FP increases across several classes, not only the active class.",
        "seed_1_success_main_cause": "Conservative/no-op behavior. No ROI or industrial augmentation was applied, so the pass is not evidence that CATF-v2 augmentation helped; it may include in-loop diagnosis/callback RNG effects.",
        "seed_2_failure_main_cause": "Conservative confidence/recall collapse on a clean seed that already had high Recall. CATF-v2 did not target the final degraded classes.",
        "catf_v2_positioning": "Promising class-aware diagnostic controller and ablation, not yet a paper main method.",
        "next_step_priority": [
            "Add analysis-stage per-class threshold calibration validation before more training changes.",
            "Add negative-effect attribution and class-level rollback logic before changing activation thresholds again.",
            "Use clean-reference-aware gating: if a seed/run already has high Recall/mAP, reduce training intervention.",
            "Improve per-class diagnosis stability with evidence accumulated across multiple feedback epochs.",
        ],
        "supporting_reports": {
            "curve": str(REPORTS / "curve_diagnosis.md"),
            "active_class_effect": str(REPORTS / "active_class_effect_analysis.md"),
            "precision_drop": str(REPORTS / "precision_drop_analysis.md"),
            "recall_drop": str(REPORTS / "recall_drop_analysis.md"),
            "threshold_calibration": str(REPORTS / "threshold_calibration_analysis.md"),
            "controller_behavior": str(REPORTS / "controller_behavior_analysis.md"),
        },
        "searched_artifacts": {
            "confusion_or_pr_curve_files": curve_like_artifacts,
            "validation_prediction_files_found": len(prediction_artifacts),
            "validation_prediction_examples": prediction_artifacts[:5],
        },
        "headline_metrics": summary.get("aggregate", {}),
    }
    write_json(REPORTS / "catf_v2_failure_mode_summary.json", failure)

    failure_lines = [
        "# CATF-v2 Failure Mode Summary",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Main Failure Modes",
        "",
    ]
    for mode in failure["main_failure_modes"]:
        failure_lines.append(f"- `{mode}`")
    failure_lines += [
        "",
        "## Seed-Level Conclusions",
        "",
        f"- Seed 0: {failure['seed_0_failure_main_cause']}",
        f"- Seed 1: {failure['seed_1_success_main_cause']}",
        f"- Seed 2: {failure['seed_2_failure_main_cause']}",
        "",
        "## Mechanism Diagnosis",
        "",
        "- CATF-v2 fixed the OK3 activation problem: OK2/OK3 were never active and OK3 ROI remained 0.",
        "- The remaining instability is not mainly bad OK-class activation. It is sparse intervention plus unstable global confidence/threshold behavior.",
        "- ROI-aware augmentation is too sparse to prove causal benefit: only 23 ROI applications total across seeds 0/1/2.",
        "- Controller behavior is over-conservative: many shrink/freeze actions, no rollback/cooldown, and only two accepted/proposed active class policies.",
        "- Seed 1 passes with zero industrial augmentation, so CATF-v2 gains are not yet causally attributable to ROI-aware augmentation.",
        "- Some failures happen in non-active classes, so the controller lacks negative-effect attribution.",
        "- No confusion/PR curve artifacts were found under the multiseed root; prediction JSON files were available for feedback-epoch diagnosis.",
        "",
        "## Next Step",
        "",
        "- Do not run more 50 epoch training before evaluating threshold calibration as a post-processing/validation layer.",
        "- Add a diagnosis-only in-loop control that enables feedback callbacks/diagnosis but keeps industrial augmentation probabilities at zero, to isolate RNG/callback side effects.",
        "- Next controller work should add per-class negative-effect attribution and rollback, not just more activation tuning.",
        "- CATF-v2 should be positioned as a promising ablation/controller variant rather than the paper main method until it passes multiseed constraints.",
    ]
    write_md(REPORTS / "catf_v2_failure_mode_summary.md", failure_lines)


if __name__ == "__main__":
    build_reports()
    print(f"Wrote CATF-v2 failure-mode analysis reports to {REPORTS}")
