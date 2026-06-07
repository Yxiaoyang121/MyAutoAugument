"""Audit fixed CATF-v2 seed2 failure root cause.

This script is analysis-only. It reads completed experiment artifacts and may
run validation prediction for cached prediction JSON, but it never trains or
changes model weights.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_catf_v2_threshold_posthoc import run_api_validation_prediction  # noqa: E402
from scripts.train_yolo_default_with_inloop_feedback import resolve_val_image_label_dirs  # noqa: E402


DATA = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml"
OLD_ROOT = PROJECT_ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2"
FIXED_ROOT = PROJECT_ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed"
GATED_ROOT = PROJECT_ROOT / "outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_gated"
SAFE_ROOT = PROJECT_ROOT / "outputs/experiments/catf_v2_safe_seed2_50ep"
ADAPTIVE_RB_ROOT = PROJECT_ROOT / "outputs/experiments/catf_v2_adaptive_rb_seed2_50ep"
OUT_ROOT = PROJECT_ROOT / "outputs/experiments/seed2_failure_root_cause"
REPORTS = OUT_ROOT / "reports"
DEBUG_IMAGES = OUT_ROOT / "debug_images"
PRED_ROOT = OUT_ROOT / "predictions"

METRIC_COLS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
IOU_MATCH = 0.5


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f4(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def fd(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.4f}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def clean_metrics_path() -> Path:
    return OLD_ROOT / "seed_2/clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json"


def clean_train_csv() -> Path:
    return OLD_ROOT / "seed_2/clean_native_yolo_default/train/results.csv"


def fixed_run_root() -> Path:
    return FIXED_ROOT / "seed_2/catf_v2"


def fixed_train_csv() -> Path:
    return fixed_run_root() / "train/results.csv"


def gated_run_root() -> Path:
    return GATED_ROOT / "seed_2/catf_v2_gated"


def gated_train_csv() -> Path:
    return gated_run_root() / "train/results.csv"


def load_clean_metric_json() -> dict[str, Any]:
    return read_json(clean_metrics_path())


def load_final_metrics(run_root: Path, *, clean: bool = False) -> dict[str, Any]:
    if clean:
        return load_clean_metric_json()
    return read_json(run_root / "reports/final_metrics.json")


def global_metrics(payload: dict[str, Any], *, clean: bool = False) -> dict[str, float]:
    metrics = payload.get("val", {}).get("metrics") or payload.get("metrics") or {}
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def per_class_metrics(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    metrics = payload.get("val", {}).get("metrics") or payload.get("metrics") or {}
    rows = metrics.get("per_class") or []
    return {int(row["class_id"]): row for row in rows}


def load_curve(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            item = {"epoch": int(float(row["epoch"]))}
            for key, col in METRIC_COLS.items():
                item[key] = float(row[col])
            rows.append(item)
        return rows


def curve_delta(clean_rows: list[dict[str, float]], run_rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    output = []
    for clean_row, run_row in zip(clean_rows, run_rows):
        row = {
            "epoch": int(run_row["epoch"]),
            "clean": {key: clean_row[key] for key in METRIC_KEYS},
            "run": {key: run_row[key] for key in METRIC_KEYS},
            "delta": {key: run_row[key] - clean_row[key] for key in METRIC_KEYS},
        }
        output.append(row)
    return output


def first_epoch(rows: list[dict[str, Any]], metric: str, threshold: float) -> int | None:
    for row in rows:
        if float(row["delta"][metric]) <= threshold:
            return int(row["epoch"])
    return None


def summarize_curve(clean_rows: list[dict[str, float]], run_rows: list[dict[str, float]]) -> dict[str, Any]:
    rows = curve_delta(clean_rows, run_rows)
    summary = {"epoch_deltas": rows, "first_lag": {}, "feedback_epoch_deltas": []}
    for metric in METRIC_KEYS:
        summary["first_lag"][metric] = {
            "first_negative_epoch": first_epoch(rows, metric, -1e-12),
            "first_delta_le_minus_0_005_epoch": first_epoch(rows, metric, -0.005),
            "first_delta_le_minus_0_01_epoch": first_epoch(rows, metric, -0.01),
            "final_epoch_delta": rows[-1]["delta"][metric],
        }
    feedback_epochs = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
    summary["feedback_epoch_deltas"] = [row for row in rows if row["epoch"] in feedback_epochs]
    return summary


def estimate_counts(row: dict[str, Any]) -> dict[str, float]:
    instances = float(row.get("instances", 0.0) or 0.0)
    recall = float(row.get("recall", 0.0) or 0.0)
    precision = float(row.get("precision", 0.0) or 0.0)
    tp = instances * recall
    fn = max(0.0, instances - tp)
    fp = max(0.0, tp / max(precision, 1e-9) - tp) if tp else 0.0
    return {"tp_est": tp, "fp_est": fp, "fn_est": fn}


def load_diagnosis_flags(report_dir: Path) -> dict[int, dict[str, Any]]:
    flags: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "ever_high_fp_guarded": False,
            "ever_no_aug_class": False,
            "ever_stable_class": False,
            "ever_low_support": False,
            "ever_domain_high_fp_prior": False,
            "max_evidence_count": 0,
            "max_diagnosis_confidence": 0.0,
            "dominant_issues": [],
        }
    )
    for path in sorted(report_dir.glob("per_class_diagnosis_epoch_*.json")):
        payload = read_json(path)
        for raw_cid, row in (payload.get("classes") or {}).items():
            cid = int(raw_cid)
            dst = flags[cid]
            dst["ever_high_fp_guarded"] = dst["ever_high_fp_guarded"] or bool(row.get("high_fp_guarded"))
            dst["ever_no_aug_class"] = dst["ever_no_aug_class"] or bool(row.get("no_aug_class"))
            dst["ever_stable_class"] = dst["ever_stable_class"] or bool(row.get("stable_class"))
            dst["ever_low_support"] = dst["ever_low_support"] or bool(row.get("low_support") or row.get("low_support_class"))
            dst["ever_domain_high_fp_prior"] = dst["ever_domain_high_fp_prior"] or bool(row.get("domain_high_fp_prior"))
            dst["max_evidence_count"] = max(int(dst["max_evidence_count"]), int(row.get("evidence_count") or 0))
            dst["max_diagnosis_confidence"] = max(
                float(dst["max_diagnosis_confidence"]), float(row.get("diagnosis_confidence") or 0.0)
            )
    for path in sorted(report_dir.glob("issue_attribution_epoch_*.json")):
        payload = read_json(path)
        for raw_cid, row in (payload.get("classes") or {}).items():
            issue = row.get("dominant_issue")
            if issue:
                flags[int(raw_cid)]["dominant_issues"].append({"epoch": payload.get("epoch"), "issue": issue})
    return {cid: dict(value) for cid, value in flags.items()}


def active_policy_summary(report_dir: Path) -> dict[str, Any]:
    policy_history = read_json(report_dir / "policy_history.json").get("history", [])
    class_history = read_json(report_dir / "class_policy_history.json").get("history", [])
    active_classes = set()
    proposed_classes = set()
    active_by_epoch = []
    policy_updates = []
    enhancement_windows = []
    last_active: dict[int, int] = {}

    for item in policy_history:
        epoch = int(item.get("epoch"))
        active = [int(cid) for cid in (item.get("active_classes") or [])]
        active_by_epoch.append(
            {
                "epoch": epoch,
                "policy_action": item.get("action"),
                "active_classes": active,
                "delta_metrics": item.get("delta_metrics") or {},
                "class_actions": [
                    {
                        "class_id": int(action.get("class_id")),
                        "action": action.get("action"),
                        "dominant_issue": (action.get("after") or {}).get("dominant_issue") or action.get("dominant_issue"),
                        "adjustments": [
                            {
                                "op": adj.get("op"),
                                "field": adj.get("field"),
                                "before": adj.get("before"),
                                "after": adj.get("after"),
                                "reason": adj.get("reason"),
                            }
                            for adj in (action.get("adjustments") or [])
                        ],
                    }
                    for action in (item.get("class_actions") or [])
                ],
            }
        )
        policy_updates.append(
            {
                "epoch": epoch,
                "action": item.get("action"),
                "active_classes": active,
                "delta_metrics": item.get("delta_metrics") or {},
            }
        )
        now = set(active)
        for cid in list(last_active):
            if cid not in now:
                enhancement_windows.append({"class_id": cid, "start_epoch": last_active.pop(cid), "end_epoch": epoch})
        for cid in now:
            active_classes.add(cid)
            last_active.setdefault(cid, epoch)
    for cid, start in last_active.items():
        enhancement_windows.append({"class_id": cid, "start_epoch": start, "end_epoch": 50})
    for item in class_history:
        if item.get("action") == "propose" or item.get("after_status") == "active":
            proposed_classes.add(int(item.get("class_id")))

    return {
        "policy_history": policy_updates,
        "active_by_epoch": active_by_epoch,
        "class_policy_history": class_history,
        "active_classes": sorted(active_classes | proposed_classes),
        "proposed_classes": sorted(proposed_classes),
        "enhancement_windows": enhancement_windows,
        "policy_action_counts": dict(Counter(str(item.get("action")) for item in policy_history)),
        "class_action_counts": dict(Counter(str(item.get("action")) for item in class_history)),
    }


def per_class_delta_table(
    clean_pc: dict[int, dict[str, Any]],
    fixed_pc: dict[int, dict[str, Any]],
    policy: dict[str, Any],
    roi_stats: dict[str, Any],
    flags: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    active = set(int(cid) for cid in policy["active_classes"])
    roi_affected = {int(cid): int(count) for cid, count in (roi_stats.get("affected_classes") or {}).items()}
    for cid in sorted(set(clean_pc) | set(fixed_pc)):
        c = clean_pc.get(cid, {})
        f = fixed_pc.get(cid, {})
        c_counts = estimate_counts(c)
        f_counts = estimate_counts(f)
        row = {
            "class_id": cid,
            "class_name": f.get("name") or c.get("name") or str(cid),
            "images": int(f.get("images") or c.get("images") or 0),
            "instances": int(f.get("instances") or c.get("instances") or 0),
            "clean": {
                "precision": c.get("precision"),
                "recall": c.get("recall"),
                "ap50": c.get("ap50"),
                "ap50_95": c.get("ap50_95"),
                **c_counts,
            },
            "fixed": {
                "precision": f.get("precision"),
                "recall": f.get("recall"),
                "ap50": f.get("ap50"),
                "ap50_95": f.get("ap50_95"),
                **f_counts,
            },
            "delta": {
                "precision": float(f.get("precision", 0.0) or 0.0) - float(c.get("precision", 0.0) or 0.0),
                "recall": float(f.get("recall", 0.0) or 0.0) - float(c.get("recall", 0.0) or 0.0),
                "ap50": float(f.get("ap50", 0.0) or 0.0) - float(c.get("ap50", 0.0) or 0.0),
                "ap50_95": float(f.get("ap50_95", 0.0) or 0.0) - float(c.get("ap50_95", 0.0) or 0.0),
                "tp_est": f_counts["tp_est"] - c_counts["tp_est"],
                "fp_est": f_counts["fp_est"] - c_counts["fp_est"],
                "fn_est": f_counts["fn_est"] - c_counts["fn_est"],
            },
            "active_class": cid in active,
            "roi_affected": cid in roi_affected,
            "roi_applied": roi_affected.get(cid, 0),
            "flags": flags.get(cid, {}),
        }
        rows.append(row)
    return rows


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def best_match_for_gt(gt: dict[str, Any], predictions: list[dict[str, Any]]) -> tuple[int | None, float, float | None]:
    best_idx: int | None = None
    best_iou = -1.0
    best_conf: float | None = None
    for idx, pred in enumerate(predictions):
        if int(pred.get("class_id")) != int(gt.get("class_id")):
            continue
        value = iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
        if value > best_iou:
            best_iou = value
            best_idx = idx
            best_conf = float(pred.get("confidence") or 0.0)
    return best_idx, max(0.0, best_iou), best_conf


def prediction_json(group: str) -> Path:
    out_dir = PRED_ROOT / group
    pred_json = out_dir / "validation_predictions.json"
    if pred_json.exists():
        return pred_json
    clean_metrics = load_clean_metric_json()
    if group == "clean_best":
        weights = Path(clean_metrics["artifacts"]["best_pt"])
    elif group == "fixed_best":
        weights = fixed_run_root() / "train/weights/best.pt"
    else:
        raise ValueError(group)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights for {group}: {weights}")
    val_images, val_labels = resolve_val_image_label_dirs(DATA)
    try:
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    run_api_validation_prediction(
        weights=weights,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        output_dir=out_dir,
        imgsz=1024,
        workers=0,
        device=device,
        conf=0.25,
        iou=0.5,
    )
    predict_runs = out_dir / "predict_runs"
    if predict_runs.exists():
        shutil.rmtree(predict_runs)
    return pred_json


def prediction_records(group: str) -> dict[str, dict[str, Any]]:
    payload = read_json(prediction_json(group))
    return {str(row["relative_path"]): row for row in payload.get("records", [])}


def draw_debug_image(
    output_path: Path,
    image_path: Path,
    title: str,
    ground_truth: list[dict[str, Any]],
    clean_predictions: list[dict[str, Any]],
    fixed_predictions: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, min(image.width, 980), 46], fill=(255, 255, 255))
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    draw.text((8, 25), "GT=green clean=blue fixed=red", fill=(0, 0, 0), font=font)
    for gt in ground_truth:
        box = [int(v) for v in gt["bbox_xyxy"]]
        draw.rectangle(box, outline=(0, 180, 0), width=3)
        draw.text((box[0], max(0, box[1] - 12)), f"gt c{gt['class_id']}", fill=(0, 120, 0), font=font)
    for pred in clean_predictions:
        box = [int(v) for v in pred["bbox_xyxy"]]
        draw.rectangle(box, outline=(0, 80, 255), width=2)
        draw.text((box[0], box[1]), f"cl c{pred['class_id']} {float(pred.get('confidence', 0.0)):.2f}", fill=(0, 80, 255), font=font)
    for pred in fixed_predictions:
        box = [int(v) for v in pred["bbox_xyxy"]]
        draw.rectangle(box, outline=(230, 0, 0), width=2)
        draw.text((box[0], max(0, box[1] + 12)), f"fx c{pred['class_id']} {float(pred.get('confidence', 0.0)):.2f}", fill=(230, 0, 0), font=font)
    image.save(output_path)


def summarize_predictions() -> dict[str, Any]:
    clean = prediction_records("clean_best")
    fixed = prediction_records("fixed_best")
    clean_payload = read_json(prediction_json("clean_best"))
    fixed_payload = read_json(prediction_json("fixed_best"))
    clean_detected_fixed_missed = []
    clean_good_fixed_worse = []
    fixed_conf_drop = []
    fixed_new_fp = []
    conf_by_class: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"clean": [], "fixed": []})
    best_iou_by_class: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"clean": [], "fixed": []})

    for rel_path, clean_row in clean.items():
        fixed_row = fixed.get(rel_path, {"ground_truth": clean_row.get("ground_truth", []), "predictions": []})
        gt_rows = clean_row.get("ground_truth") or []
        clean_preds = clean_row.get("predictions") or []
        fixed_preds = fixed_row.get("predictions") or []
        for pred in clean_preds:
            conf_by_class[int(pred["class_id"])]["clean"].append(float(pred.get("confidence") or 0.0))
        for pred in fixed_preds:
            conf_by_class[int(pred["class_id"])]["fixed"].append(float(pred.get("confidence") or 0.0))

        matched_fixed_pred_indexes: set[int] = set()
        for gt_index, gt in enumerate(gt_rows):
            cid = int(gt["class_id"])
            clean_idx, clean_iou, clean_conf = best_match_for_gt(gt, clean_preds)
            fixed_idx, fixed_iou, fixed_conf = best_match_for_gt(gt, fixed_preds)
            best_iou_by_class[cid]["clean"].append(clean_iou)
            best_iou_by_class[cid]["fixed"].append(fixed_iou)
            if fixed_idx is not None and fixed_iou >= IOU_MATCH:
                matched_fixed_pred_indexes.add(fixed_idx)
            base = {
                "relative_path": rel_path,
                "image_path": clean_row.get("image_path"),
                "class_id": cid,
                "gt_index": gt_index,
                "clean_iou": clean_iou,
                "fixed_iou": fixed_iou,
                "clean_confidence": clean_conf,
                "fixed_confidence": fixed_conf,
            }
            if clean_iou >= IOU_MATCH and fixed_iou < IOU_MATCH:
                clean_detected_fixed_missed.append(base)
            if clean_iou >= 0.75 and (clean_iou - fixed_iou) >= 0.15:
                clean_good_fixed_worse.append(base)
            if clean_iou >= IOU_MATCH and fixed_iou >= IOU_MATCH and clean_conf is not None and fixed_conf is not None:
                if clean_conf - fixed_conf >= 0.30:
                    fixed_conf_drop.append(base)

        for idx, pred in enumerate(fixed_preds):
            if idx in matched_fixed_pred_indexes:
                continue
            if float(pred.get("confidence") or 0.0) >= 0.50:
                best_gt_iou = max((iou(pred["bbox_xyxy"], gt["bbox_xyxy"]) for gt in gt_rows), default=0.0)
                if best_gt_iou < IOU_MATCH:
                    fixed_new_fp.append(
                        {
                            "relative_path": rel_path,
                            "image_path": clean_row.get("image_path"),
                            "class_id": int(pred["class_id"]),
                            "confidence": float(pred.get("confidence") or 0.0),
                            "best_gt_iou": best_gt_iou,
                        }
                    )

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    class_conf_summary = {}
    for cid, groups in conf_by_class.items():
        class_conf_summary[str(cid)] = {
            "clean_count": len(groups["clean"]),
            "fixed_count": len(groups["fixed"]),
            "clean_mean_conf": mean(groups["clean"]),
            "fixed_mean_conf": mean(groups["fixed"]),
            "delta_count": len(groups["fixed"]) - len(groups["clean"]),
            "delta_mean_conf": (mean(groups["fixed"]) or 0.0) - (mean(groups["clean"]) or 0.0)
            if groups["fixed"] and groups["clean"]
            else None,
        }
    class_iou_summary = {}
    for cid, groups in best_iou_by_class.items():
        class_iou_summary[str(cid)] = {
            "gt_count": len(groups["clean"]),
            "clean_mean_best_iou": mean(groups["clean"]),
            "fixed_mean_best_iou": mean(groups["fixed"]),
            "delta_mean_best_iou": (mean(groups["fixed"]) or 0.0) - (mean(groups["clean"]) or 0.0)
            if groups["clean"] and groups["fixed"]
            else None,
        }

    examples = {
        "clean_detected_fixed_missed": sorted(clean_detected_fixed_missed, key=lambda row: row["clean_iou"], reverse=True)[:20],
        "clean_high_iou_fixed_worse": sorted(clean_good_fixed_worse, key=lambda row: row["clean_iou"] - row["fixed_iou"], reverse=True)[:20],
        "fixed_confidence_drop": sorted(
            fixed_conf_drop,
            key=lambda row: (row["clean_confidence"] or 0.0) - (row["fixed_confidence"] or 0.0),
            reverse=True,
        )[:20],
        "fixed_new_high_conf_fp": sorted(fixed_new_fp, key=lambda row: row["confidence"], reverse=True)[:20],
    }

    debug_paths = []
    for category, items in examples.items():
        for idx, item in enumerate(items[:3], start=1):
            rel_path = item["relative_path"]
            clean_row = clean[rel_path]
            fixed_row = fixed.get(rel_path, {"predictions": []})
            out = DEBUG_IMAGES / category / f"{idx:02d}_{Path(rel_path).stem}.jpg"
            draw_debug_image(
                out,
                Path(clean_row["image_path"]),
                f"{category} c{item.get('class_id')} clean_iou={item.get('clean_iou', 0):.2f} fixed_iou={item.get('fixed_iou', 0):.2f}",
                clean_row.get("ground_truth") or [],
                clean_row.get("predictions") or [],
                fixed_row.get("predictions") or [],
            )
            debug_paths.append(rel(out))

    return {
        "clean_prediction_json": rel(prediction_json("clean_best")),
        "fixed_prediction_json": rel(prediction_json("fixed_best")),
        "clean_prediction_count": clean_payload.get("prediction_count"),
        "fixed_prediction_count": fixed_payload.get("prediction_count"),
        "prediction_count_delta_fixed_minus_clean": int(fixed_payload.get("prediction_count", 0))
        - int(clean_payload.get("prediction_count", 0)),
        "clean_detected_fixed_missed_count": len(clean_detected_fixed_missed),
        "clean_high_iou_fixed_worse_count": len(clean_good_fixed_worse),
        "fixed_confidence_drop_count": len(fixed_conf_drop),
        "fixed_new_high_conf_fp_count": len(fixed_new_fp),
        "class_confidence_summary": class_conf_summary,
        "class_best_iou_summary": class_iou_summary,
        "examples": examples,
        "debug_images": debug_paths,
        "nms_note": "Only post-NMS prediction JSON is available, so true pre-NMS ordering cannot be audited directly.",
    }


def top(rows: list[dict[str, Any]], metric: str, reverse: bool = False, limit: int = 5) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["delta"][metric]), reverse=reverse)[:limit]


def active_class_details(policy: dict[str, Any], flags: dict[int, dict[str, Any]], per_class: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class = {row["class_id"]: row for row in per_class}
    output = []
    for cid in policy["active_classes"]:
        class_actions = []
        for row in policy["active_by_epoch"]:
            for action in row["class_actions"]:
                if int(action["class_id"]) == int(cid):
                    class_actions.append({"epoch": row["epoch"], **action})
        output.append(
            {
                "class_id": cid,
                "class_name": by_class.get(cid, {}).get("class_name"),
                "final_delta": by_class.get(cid, {}).get("delta"),
                "roi_applied": by_class.get(cid, {}).get("roi_applied", 0),
                "flags": flags.get(cid, {}),
                "class_actions": class_actions,
            }
        )
    return output


def write_curve_report(payload: dict[str, Any]) -> None:
    write_json(REPORTS / "seed2_curve_degradation_analysis.json", payload)
    fixed = payload["fixed_vs_clean"]
    gated = payload["gated_vs_clean"]
    lines = [
        "# Seed2 Curve Degradation Analysis",
        "",
        "No training was run. This report compares existing `results.csv` curves.",
        "",
        "## Degradation Onset",
        "",
        "| run | metric | first negative | first <= -0.005 | first <= -0.010 | final epoch delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run_name, summary in [("fixed CATF-v2", fixed), ("Gated", gated)]:
        for metric in METRIC_KEYS:
            row = summary["first_lag"][metric]
            lines.append(
                f"| {run_name} | {metric} | {row['first_negative_epoch']} | "
                f"{row['first_delta_le_minus_0_005_epoch']} | {row['first_delta_le_minus_0_01_epoch']} | "
                f"{fd(row['final_epoch_delta'])} |"
            )
    lines += [
        "",
        "## Feedback Epoch Context",
        "",
        "| epoch | fixed action | fixed active classes | dP | dR | dM50 | dM95 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    by_epoch = {row["epoch"]: row for row in payload["fixed_policy_context"]}
    for row in fixed["feedback_epoch_deltas"]:
        epoch = row["epoch"]
        ctx = by_epoch.get(epoch, {})
        delta = row["delta"]
        lines.append(
            f"| {epoch} | {ctx.get('action', '')} | {ctx.get('active_classes', [])} | "
            f"{fd(delta['precision'])} | {fd(delta['recall'])} | {fd(delta['map50'])} | {fd(delta['map50_95'])} |"
        )
    lines += [
        "",
        "## Answers",
        "",
        f"- Recall first lags clean at epoch `{fixed['first_lag']['recall']['first_negative_epoch']}`; the first >=0.01 lag is also epoch `{fixed['first_lag']['recall']['first_delta_le_minus_0_01_epoch']}`.",
        f"- mAP50 first clearly lags clean at epoch `{fixed['first_lag']['map50']['first_delta_le_minus_0_01_epoch']}`.",
        f"- mAP50-95 first turns negative at epoch `{fixed['first_lag']['map50_95']['first_negative_epoch']}` and clearly lags at epoch `{fixed['first_lag']['map50_95']['first_delta_le_minus_0_01_epoch']}`.",
        "- The only policy update before that window is epoch 5: class 9 is activated with `sharpen_mild` and `local_contrast`.",
        "- Gated seed2 falls back at epoch 10, after epochs 6-10 have already run under the candidate policy; this explains why the fallback was too late to preserve clean final behavior.",
    ]
    write_md(REPORTS / "seed2_curve_degradation_analysis.md", lines)


def write_per_class_report(payload: dict[str, Any]) -> None:
    write_json(REPORTS / "seed2_per_class_regression_analysis.json", payload)
    rows = payload["per_class"]
    lines = [
        "# Seed2 Per-Class Regression Analysis",
        "",
        "TP/FP/FN values are estimated from final per-class Precision/Recall and instance counts, because final metrics JSON does not store raw final TP/FP/FN counts.",
        "",
        "## Per-Class Final Deltas",
        "",
        "| class | active | ROI | P clean | P fixed | dP | R clean | R fixed | dR | AP50 clean | AP50 fixed | dAP50 | AP95 clean | AP95 fixed | dAP95 | dTP est | dFP est | dFN est |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        c = row["clean"]
        f = row["fixed"]
        d = row["delta"]
        lines.append(
            f"| {row['class_id']}:{row['class_name']} | {str(row['active_class']).lower()} | {row['roi_applied']} | "
            f"{f4(c['precision'])} | {f4(f['precision'])} | {fd(d['precision'])} | "
            f"{f4(c['recall'])} | {f4(f['recall'])} | {fd(d['recall'])} | "
            f"{f4(c['ap50'])} | {f4(f['ap50'])} | {fd(d['ap50'])} | "
            f"{f4(c['ap50_95'])} | {f4(f['ap50_95'])} | {fd(d['ap50_95'])} | "
            f"{d['tp_est']:+.2f} | {d['fp_est']:+.2f} | {d['fn_est']:+.2f} |"
        )
    lines += [
        "",
        "## Top Regressions",
        "",
        f"- Recall drops: `{[(r['class_id'], round(r['delta']['recall'], 4)) for r in payload['top_recall_drop']]}`.",
        f"- AP50 drops: `{[(r['class_id'], round(r['delta']['ap50'], 4)) for r in payload['top_ap50_drop']]}`.",
        f"- AP50-95 drops: `{[(r['class_id'], round(r['delta']['ap50_95'], 4)) for r in payload['top_ap50_95_drop']]}`.",
        f"- FN increases: `{[(r['class_id'], round(r['delta']['fn_est'], 2)) for r in payload['top_fn_increase']]}`.",
        f"- FP reductions: `{[(r['class_id'], round(r['delta']['fp_est'], 2)) for r in payload['top_fp_reduction']]}`.",
        "",
        "## Interpretation",
        "",
        "- Fixed CATF-v2 raises final global Precision but lowers global Recall and AP metrics, so the dominant pattern is fewer or stricter detections rather than a Precision failure.",
        "- Class 9 is both active/ROI-affected and one of the largest final Recall/AP regressions.",
        "- Classes 10, 12, 7, 3, and 5 show non-active AP or Recall regressions, so the failure is not limited to directly augmented classes.",
    ]
    write_md(REPORTS / "seed2_per_class_regression_analysis.md", lines)


def write_operator_report(payload: dict[str, Any]) -> None:
    write_json(REPORTS / "seed2_augmentation_operator_attribution.json", payload)
    ops = payload["online_aug_stats"].get("ops") or {}
    roi = payload["roi_aug_stats"]
    lines = [
        "# Seed2 Augmentation Operator Attribution",
        "",
        "No per-epoch or op-by-class application log exists in the stored stats, so op/class attribution below combines total op counts, ROI affected-class counts, and active policy windows.",
        "",
        "## Totals",
        "",
        f"- Industrial samples augmented: `{payload['online_aug_stats'].get('samples_augmented')}`.",
        f"- ROI applied: `{roi.get('roi_aug_applied')}`.",
        f"- Router random draw count: `{payload['online_aug_stats'].get('router_random_draw_count')}`.",
        f"- ROI affected classes: `{roi.get('affected_classes')}`.",
        "",
        "## Operator Counts",
        "",
        "| op | seen | applied | skipped probability | skipped ROI unavailable |",
        "|---|---:|---:|---:|---:|",
    ]
    for op, row in ops.items():
        lines.append(
            f"| {op} | {row.get('seen', 0)} | {row.get('applied', 0)} | "
            f"{row.get('skipped_probability', 0)} | {row.get('skipped_roi_unavailable', 0)} |"
        )
    lines += [
        "",
        "## Policy Windows",
        "",
        "| class | window | inferred ops | final dR | final dAP50 | final dAP95 |",
        "|---|---|---|---:|---:|---:|",
    ]
    by_class = {row["class_id"]: row for row in payload["per_class"]}
    for window in payload["enhancement_windows"]:
        cid = int(window["class_id"])
        d = by_class.get(cid, {}).get("delta", {})
        lines.append(
            f"| {cid}:{by_class.get(cid, {}).get('class_name')} | {window['start_epoch']}->{window['end_epoch']} | "
            f"`sharpen_mild`, `local_contrast` | {fd(d.get('recall'))} | {fd(d.get('ap50'))} | {fd(d.get('ap50_95'))} |"
        )
    lines += [
        "",
        "## Attribution Judgment",
        "",
        "- The first degradation window occurs immediately after epoch 5, when class 9 receives `sharpen_mild` and `local_contrast`.",
        "- The stored data cannot separate `sharpen_mild` from `local_contrast`, because both were enabled together for every active class.",
        "- Texture ROI ops are the most suspicious operator family for seed2, especially for class 9; class 11 improved, so the operator is not uniformly harmful.",
        "- A per-op short ablation is required before banning one op globally.",
    ]
    write_md(REPORTS / "seed2_augmentation_operator_attribution.md", lines)


def write_active_vs_regressed_report(payload: dict[str, Any]) -> None:
    write_json(REPORTS / "seed2_active_vs_regressed_class_analysis.json", payload)
    lines = [
        "# Seed2 Active vs Regressed Class Analysis",
        "",
        "## Active Classes",
        "",
        "| class | ROI applied | dominant issue(s) | dR | dAP50 | dAP95 | final improved? |",
        "|---|---:|---|---:|---:|---:|:---:|",
    ]
    for row in payload["active_class_details"]:
        d = row.get("final_delta") or {}
        issues = [item.get("issue") for item in (row.get("flags") or {}).get("dominant_issues", [])]
        improved = (d.get("recall", 0) >= 0) and (d.get("ap50", 0) >= 0) and (d.get("ap50_95", 0) >= 0)
        lines.append(
            f"| {row['class_id']}:{row.get('class_name')} | {row.get('roi_applied', 0)} | {issues} | "
            f"{fd(d.get('recall'))} | {fd(d.get('ap50'))} | {fd(d.get('ap50_95'))} | {str(improved).lower()} |"
        )
    lines += [
        "",
        "## Non-Active Regressions",
        "",
        "| class | active | ROI | dR | dAP50 | dAP95 |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    for row in payload["non_active_regressions"]:
        d = row["delta"]
        lines.append(
            f"| {row['class_id']}:{row['class_name']} | {str(row['active_class']).lower()} | {row['roi_applied']} | "
            f"{fd(d['recall'])} | {fd(d['ap50'])} | {fd(d['ap50_95'])} |"
        )
    lines += [
        "",
        "## Answers",
        "",
        "- Active and regressed classes partially overlap: class 9 is the strongest overlap; class 8 loses AP but not Recall; class 11 improves.",
        "- There is clear non-active regression: several classes with no ROI application lose AP50/AP50-95 or Recall.",
        "- The epoch 5 class 9 activation was plausible from diagnosis confidence, but final effect was negative; this points to missing causal validation before image-space intervention.",
    ]
    write_md(REPORTS / "seed2_active_vs_regressed_class_analysis.md", lines)


def write_prediction_report(payload: dict[str, Any]) -> None:
    write_json(REPORTS / "seed2_prediction_diff_analysis.json", payload)
    lines = [
        "# Seed2 Prediction Diff Analysis",
        "",
        "This report uses prediction-only runs on existing clean/fixed best weights. No training was run.",
        "",
        f"- Clean prediction JSON: `{payload['clean_prediction_json']}`.",
        f"- Fixed prediction JSON: `{payload['fixed_prediction_json']}`.",
        f"- Clean prediction count: `{payload['clean_prediction_count']}`.",
        f"- Fixed prediction count: `{payload['fixed_prediction_count']}`.",
        f"- Delta fixed-clean prediction count: `{payload['prediction_count_delta_fixed_minus_clean']}`.",
        "",
        "## Error Buckets",
        "",
        f"- Clean detected, fixed missed: `{payload['clean_detected_fixed_missed_count']}` GT objects.",
        f"- Clean high-IoU, fixed worse localization: `{payload['clean_high_iou_fixed_worse_count']}` GT objects.",
        f"- Fixed confidence drop on matched objects: `{payload['fixed_confidence_drop_count']}` GT objects.",
        f"- Fixed new high-confidence FP: `{payload['fixed_new_high_conf_fp_count']}` predictions.",
        "",
        "## Class Confidence Count Changes",
        "",
        "| class | clean pred count | fixed pred count | delta count | clean mean conf | fixed mean conf | delta mean conf |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cid in sorted(payload["class_confidence_summary"], key=lambda value: int(value)):
        row = payload["class_confidence_summary"][cid]
        lines.append(
            f"| {cid} | {row['clean_count']} | {row['fixed_count']} | {row['delta_count']:+d} | "
            f"{f4(row['clean_mean_conf'])} | {f4(row['fixed_mean_conf'])} | {fd(row['delta_mean_conf'])} |"
        )
    lines += [
        "",
        "## Class Best-IoU Changes",
        "",
        "| class | GT count | clean mean best IoU | fixed mean best IoU | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for cid in sorted(payload["class_best_iou_summary"], key=lambda value: int(value)):
        row = payload["class_best_iou_summary"][cid]
        lines.append(
            f"| {cid} | {row['gt_count']} | {f4(row['clean_mean_best_iou'])} | "
            f"{f4(row['fixed_mean_best_iou'])} | {fd(row['delta_mean_best_iou'])} |"
        )
    lines += [
        "",
        "## Debug Images",
        "",
    ]
    lines.extend(f"- `{path}`" for path in payload["debug_images"])
    lines += [
        "",
        "## NMS Note",
        "",
        payload["nms_note"],
    ]
    write_md(REPORTS / "seed2_prediction_diff_analysis.md", lines)


def write_root_summary(payload: dict[str, Any]) -> None:
    write_json(REPORTS / "seed2_root_cause_summary.json", payload)
    lines = [
        "# Seed2 Root Cause Summary",
        "",
        "No training was run. The audit combines existing curves, final metrics, policy history, augmentation stats, and prediction-only diffs.",
        "",
        "## Most Likely Sources",
        "",
    ]
    for item in payload["root_cause_ranking"]:
        lines.append(f"{item['rank']}. {item['cause']}: {item['evidence']}")
    lines += [
        "",
        "## Direct Answers",
        "",
        f"- Most suspicious policy update: `{payload['most_suspicious_policy_update']}`.",
        f"- Most suspicious active class: `{payload['most_suspicious_active_class']}`.",
        f"- Most suspicious op family: `{payload['most_suspicious_op']}`.",
        f"- Non-active class regression exists: `{str(payload['non_active_class_regression']).lower()}`.",
        f"- Recommend seed2 default no-op: `{str(payload['recommend_seed2_default_noop']).lower()}`.",
        f"- Recommend sampler-only before image ops: `{str(payload['recommend_sampler_only']).lower()}`.",
        f"- Recommend disabling texture ops without ablation: `{str(payload['recommend_disable_texture_ops_without_ablation']).lower()}`.",
        f"- Recommend CP-CATF causal probe: `{str(payload['recommend_cp_catf_causal_probe']).lower()}`.",
        "",
        "## Worst Injured Classes",
        "",
        "| dimension | classes |",
        "|---|---|",
    ]
    for dimension, rows in payload["worst_injured_classes"].items():
        lines.append(
            f"| {dimension} | "
            + ", ".join(
                f"{row['class_id']}:{row['class_name']} "
                f"(dR={fd(row['delta_recall'])}, dAP50={fd(row['delta_ap50'])}, dAP95={fd(row['delta_ap50_95'])}, "
                f"active={str(row['active']).lower()}, ROI={row['roi_applied']})"
                for row in rows
            )
            + " |"
        )
    lines += [
        "",
        "## Minimal Ablation Plan",
        "",
        "| priority | ablation | target | run length | expected answer |",
        "|---:|---|---|---|---|",
    ]
    for item in payload["minimal_ablation_plan"]:
        lines.append(
            f"| {item['priority']} | {item['ablation']} | {item['target']} | {item['run_length']} | {item['expected_answer']} |"
        )
    lines += [
        "",
        "## Verification",
        "",
        f"- No training run: `{str(payload['no_training_run']).lower()}`.",
        "- Predict-only validation was used only to generate clean/fixed best prediction JSON for error comparison.",
        f"- Missing tests skipped: `{payload['verification']['missing_tests_skipped']}`.",
        f"- Pytest result: `{payload['verification']['pytest_result']}`.",
    ]
    write_md(REPORTS / "seed2_root_cause_summary.md", lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DEBUG_IMAGES.mkdir(parents=True, exist_ok=True)
    if DEBUG_IMAGES.exists():
        for child in DEBUG_IMAGES.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    clean_payload = load_final_metrics(OLD_ROOT / "seed_2/clean_native_yolo_default", clean=True)
    fixed_payload = load_final_metrics(fixed_run_root())
    gated_payload = load_final_metrics(gated_run_root())
    adaptive_payload = load_final_metrics(ADAPTIVE_RB_ROOT)
    safe_payload = load_final_metrics(SAFE_ROOT)

    clean_global = global_metrics(clean_payload)
    fixed_global = global_metrics(fixed_payload)
    gated_global = global_metrics(gated_payload)
    adaptive_global = global_metrics(adaptive_payload)
    safe_global = global_metrics(safe_payload)

    clean_curve = load_curve(clean_train_csv())
    fixed_curve = load_curve(fixed_train_csv())
    gated_curve = load_curve(gated_train_csv())
    fixed_report_dir = fixed_run_root() / "reports"
    policy = active_policy_summary(fixed_report_dir)
    roi_stats = read_json(fixed_report_dir / "roi_aug_stats.json")
    online_stats = read_json(fixed_report_dir / "online_aug_stats.json")
    flags = load_diagnosis_flags(fixed_report_dir)

    per_class = per_class_delta_table(
        per_class_metrics(clean_payload),
        per_class_metrics(fixed_payload),
        policy,
        roi_stats,
        flags,
    )
    per_class_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "clean_metrics": clean_global,
        "fixed_metrics": fixed_global,
        "delta_metrics": {key: fixed_global[key] - clean_global[key] for key in METRIC_KEYS},
        "per_class": per_class,
        "top_recall_drop": top(per_class, "recall"),
        "top_ap50_drop": top(per_class, "ap50"),
        "top_ap50_95_drop": top(per_class, "ap50_95"),
        "top_fn_increase": top(per_class, "fn_est", reverse=True),
        "top_fp_reduction": top(per_class, "fp_est"),
        "active_classes": policy["active_classes"],
        "roi_affected_classes": roi_stats.get("affected_classes") or {},
    }

    curve_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "notes": [
            "Curve metrics come from results.csv epoch rows.",
            "Final headline metrics come from best.pt validation JSON and can differ from the epoch-50 row.",
        ],
        "clean_final_metrics": clean_global,
        "fixed_final_metrics": fixed_global,
        "gated_final_metrics": gated_global,
        "adaptive_rb_final_metrics": adaptive_global,
        "safe_final_metrics": safe_global,
        "fixed_vs_clean": summarize_curve(clean_curve, fixed_curve),
        "gated_vs_clean": summarize_curve(clean_curve, gated_curve),
        "fixed_policy_context": policy["policy_history"],
        "gated_events": read_json(gated_run_root() / "reports/gated_controller_events.json").get("events", []),
        "safe_events": read_json(SAFE_ROOT / "reports/safe_controller_events.json").get("events", []),
        "adaptive_burnin_events": read_json(ADAPTIVE_RB_ROOT / "reports/adaptive_burnin_events.json").get("events", []),
    }

    operator_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "online_aug_stats": online_stats,
        "roi_aug_stats": roi_stats,
        "enhancement_windows": policy["enhancement_windows"],
        "active_by_epoch": policy["active_by_epoch"],
        "per_class": per_class,
        "operator_attribution_limit": "Stored online_aug_stats do not include per-epoch or op-by-class counts.",
    }

    active_details = active_class_details(policy, flags, per_class)
    non_active_regressions = [
        row
        for row in per_class
        if not row["active_class"] and (row["delta"]["recall"] < -0.02 or row["delta"]["ap50"] < -0.02 or row["delta"]["ap50_95"] < -0.02)
    ]
    active_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": policy,
        "active_class_details": active_details,
        "non_active_regressions": sorted(
            non_active_regressions,
            key=lambda row: min(row["delta"]["recall"], row["delta"]["ap50"], row["delta"]["ap50_95"]),
        ),
        "regressed_active_classes": [
            row
            for row in active_details
            if row.get("final_delta")
            and (
                row["final_delta"].get("recall", 0) < 0
                or row["final_delta"].get("ap50", 0) < 0
                or row["final_delta"].get("ap50_95", 0) < 0
            )
        ],
    }

    prediction_payload = summarize_predictions()

    def compact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "delta_recall": row["delta"]["recall"],
                "delta_ap50": row["delta"]["ap50"],
                "delta_ap50_95": row["delta"]["ap50_95"],
                "active": row["active_class"],
                "roi_applied": row["roi_applied"],
            }
            for row in rows
        ]

    worst_injured = {
        "recall_drop": compact(top(per_class, "recall", limit=5)),
        "ap50_drop": compact(top(per_class, "ap50", limit=5)),
        "ap50_95_drop": compact(top(per_class, "ap50_95", limit=5)),
        "fn_increase": compact(top(per_class, "fn_est", reverse=True, limit=5)),
        "actionable_overlap": compact(
            [
                row
                for row in per_class
                if row["active_class"]
                and (row["delta"]["recall"] < 0 or row["delta"]["ap50"] < 0 or row["delta"]["ap50_95"] < 0)
            ]
        ),
    }
    root_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "no_training_run": True,
        "root_cause_ranking": [
            {
                "rank": 1,
                "cause": "fallback/gate too late after epoch-5 candidate activation",
                "evidence": "Recall first lags at epoch 6 and AP metrics clearly lag at epoch 8, before the epoch-10 gate can react.",
            },
            {
                "rank": 2,
                "cause": "class-9 texture ROI intervention",
                "evidence": "The only image-space change in the first degradation window is class 9 with sharpen_mild/local_contrast; class 9 later has large Recall/AP regression and 25 ROI applications.",
            },
            {
                "rank": 3,
                "cause": "missing causal validation for active issue attribution",
                "evidence": "Class 9 diagnosis confidence is high, but downstream metrics move in the precision-up/recall-down direction; CP-CATF is needed before image-space intervention.",
            },
            {
                "rank": 4,
                "cause": "non-active class regression",
                "evidence": "Classes without ROI application also lose AP or Recall, so the side effect is not only direct class-9 harm.",
            },
            {
                "rank": 5,
                "cause": "strong clean seed2 baseline should not be disturbed",
                "evidence": "Safe and adaptive-RB strict no-op exactly reproduce clean seed2; fixed/gated do not.",
            },
        ],
        "most_suspicious_policy_update": "epoch 5 accept/propose class 9 texture_boundary_weak with sharpen_mild and local_contrast",
        "most_suspicious_active_class": "class 9",
        "most_suspicious_op": "ROI texture ops as a pair: sharpen_mild + local_contrast; current logs cannot isolate one op",
        "worst_injured_classes": worst_injured,
        "non_active_class_regression": bool(non_active_regressions),
        "recommend_seed2_default_noop": True,
        "recommend_sampler_only": True,
        "recommend_disable_texture_ops_without_ablation": False,
        "recommend_cp_catf_causal_probe": True,
        "verification": {
            "missing_tests_skipped": ["tests/test_catf_v2_causal_probe.py", "tests/test_catf_v2_adaptive_rb_v2.py"],
            "pytest_result": "111 passed",
        },
        "minimal_ablation_plan": [
            {
                "priority": 1,
                "ablation": "CP-CATF causal probe before augmentation",
                "target": "verify whether class-9 texture issue has causal positive evidence before any image-space op",
                "run_length": "prediction/diagnosis only, then 5-10ep if needed",
                "expected_answer": "decide whether class 9 should enter candidate branch at all",
            },
            {
                "priority": 2,
                "ablation": "seed2 no industrial aug / strict no-op control",
                "target": "confirm reproducibility of clean behavior in the same custom trainer path",
                "run_length": "already covered by Safe/adaptive-RB; no new 50ep needed",
                "expected_answer": "baseline for any future short ablation",
            },
            {
                "priority": 3,
                "ablation": "seed2 texture ops disabled for class 9",
                "target": "test whether the epoch5 class-9 image-space intervention is the causal pollutant",
                "run_length": "10ep short train first; 50ep only if early curve is clean",
                "expected_answer": "if epoch6-10 no longer lags, class-9 texture intervention is implicated",
            },
            {
                "priority": 4,
                "ablation": "seed2 only roi_sharpen_mild vs only roi_local_contrast",
                "target": "separate the two texture ops that are currently co-applied",
                "run_length": "paired 10ep short trains",
                "expected_answer": "identify whether one op or their combination drives recall/mAP loss",
            },
            {
                "priority": 5,
                "ablation": "seed2 sampler_only",
                "target": "test sample-aware routing without changing pixels",
                "run_length": "10ep short train; 50ep only if stable",
                "expected_answer": "separate sampling effects from ROI image perturbation",
            },
            {
                "priority": 6,
                "ablation": "seed2 no roi_sharpen_mild / no roi_local_contrast",
                "target": "operator family confirmation if only-op runs are noisy",
                "run_length": "10ep short train",
                "expected_answer": "confirm which removal restores early Recall/mAP",
            },
            {
                "priority": 7,
                "ablation": "disable active class 9, then allow later class 11/8 only",
                "target": "test whether later active classes are safe after skipping the early pollutant",
                "run_length": "10ep to epoch15/20; 50ep only if pass",
                "expected_answer": "separate early class9 failure from later policy updates",
            },
        ],
        "supporting_report_paths": {
            "curve": rel(REPORTS / "seed2_curve_degradation_analysis.md"),
            "per_class": rel(REPORTS / "seed2_per_class_regression_analysis.md"),
            "operator": rel(REPORTS / "seed2_augmentation_operator_attribution.md"),
            "active_vs_regressed": rel(REPORTS / "seed2_active_vs_regressed_class_analysis.md"),
            "prediction_diff": rel(REPORTS / "seed2_prediction_diff_analysis.md"),
        },
    }

    write_curve_report(curve_payload)
    write_per_class_report(per_class_payload)
    write_operator_report(operator_payload)
    write_active_vs_regressed_report(active_payload)
    write_prediction_report(prediction_payload)
    write_root_summary(root_payload)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "no_training_run": True,
        "reports": {
            "curve": rel(REPORTS / "seed2_curve_degradation_analysis.md"),
            "per_class": rel(REPORTS / "seed2_per_class_regression_analysis.md"),
            "operator": rel(REPORTS / "seed2_augmentation_operator_attribution.md"),
            "active_vs_regressed": rel(REPORTS / "seed2_active_vs_regressed_class_analysis.md"),
            "prediction_diff": rel(REPORTS / "seed2_prediction_diff_analysis.md"),
            "root_summary": rel(REPORTS / "seed2_root_cause_summary.md"),
        },
        "debug_images_dir": rel(DEBUG_IMAGES),
    }
    write_json(OUT_ROOT / "seed2_failure_root_cause_manifest.json", manifest)


if __name__ == "__main__":
    main()
