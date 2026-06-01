"""Post-hoc per-class threshold calibration for CATF-v2 multiseed runs.

This script runs prediction only when cached prediction JSON is missing. It does
not train or modify model weights.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostic_pipeline.prediction import collect_validation_prediction_records  # noqa: E402
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402
from scripts.train_yolo_default_with_inloop_feedback import resolve_val_image_label_dirs  # noqa: E402


ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
SOURCE_CLEAN_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_inloop_feedback")
REPORTS = ROOT / "reports"
PRED_ROOT = REPORTS / "threshold_posthoc_predictions"
DATA = Path("outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml")
SEEDS = [0, 1, 2]
THRESHOLDS = [round(0.10 + 0.05 * idx, 2) for idx in range(13)]
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
IOU_THRESHOLDS = [round(0.50 + 0.05 * idx, 2) for idx in range(10)]
CANONICAL_CLASS_NAMES = {
    0: "OK2",
    1: "OK3",
    2: "加强筋打伤",
    3: "开裂",
    4: "油污",
    5: "浅划伤",
    6: "漏背锡",
    7: "碰伤",
    8: "脏污",
    9: "轮廓划伤",
    10: "锡丝残留",
    11: "锡尖",
    12: "锡膏",
}


@dataclass
class EvalResult:
    metrics: dict[str, float]
    per_class: dict[int, dict[str, float]]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f4(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def fd(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def ap_from_pr(recall: list[float], precision: list[float]) -> float:
    if not recall:
        return 0.0
    mrec = [0.0] + recall + [1.0]
    mpre = [0.0] + precision + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return float(ap)


def match_counts(records: list[dict[str, Any]], thresholds: dict[int, float], iou_threshold: float = 0.5) -> dict[int, dict[str, float]]:
    class_ids = sorted(
        {
            int(item["class_id"])
            for record in records
            for item in record.get("ground_truth", []) + record.get("predictions", [])
        }
    )
    counts = {cid: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "gt": 0.0, "pred": 0.0} for cid in class_ids}
    for record in records:
        by_class_gt: dict[int, list[list[float]]] = defaultdict(list)
        by_class_pred: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for gt in record.get("ground_truth", []):
            by_class_gt[int(gt["class_id"])].append(gt["bbox_xyxy"])
        for pred in record.get("predictions", []):
            cid = int(pred["class_id"])
            if float(pred.get("confidence", 0.0)) >= thresholds.get(cid, 0.25):
                by_class_pred[cid].append(pred)
        for cid in class_ids:
            gt_boxes = by_class_gt.get(cid, [])
            preds = sorted(by_class_pred.get(cid, []), key=lambda item: float(item["confidence"]), reverse=True)
            matched: set[int] = set()
            counts[cid]["gt"] += len(gt_boxes)
            counts[cid]["pred"] += len(preds)
            for pred in preds:
                best_iou = 0.0
                best_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if idx in matched:
                        continue
                    score = iou_xyxy(pred["bbox_xyxy"], gt_box)
                    if score > best_iou:
                        best_iou = score
                        best_idx = idx
                if best_iou >= iou_threshold and best_idx >= 0:
                    counts[cid]["tp"] += 1.0
                    matched.add(best_idx)
                else:
                    counts[cid]["fp"] += 1.0
            counts[cid]["fn"] += max(0, len(gt_boxes) - len(matched))
    return counts


def ap_per_class(records: list[dict[str, Any]], thresholds: dict[int, float], iou_threshold: float) -> dict[int, float]:
    class_ids = sorted(
        {
            int(item["class_id"])
            for record in records
            for item in record.get("ground_truth", []) + record.get("predictions", [])
        }
    )
    gt_by_class_image: dict[int, dict[int, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    pred_by_class: dict[int, list[tuple[int, float, list[float]]]] = defaultdict(list)
    for image_idx, record in enumerate(records):
        for gt in record.get("ground_truth", []):
            gt_by_class_image[int(gt["class_id"])][image_idx].append(gt["bbox_xyxy"])
        for pred in record.get("predictions", []):
            cid = int(pred["class_id"])
            conf = float(pred.get("confidence", 0.0))
            if conf >= thresholds.get(cid, 0.25):
                pred_by_class[cid].append((image_idx, conf, pred["bbox_xyxy"]))
    ap: dict[int, float] = {}
    for cid in class_ids:
        gt_total = sum(len(items) for items in gt_by_class_image[cid].values())
        if gt_total == 0:
            continue
        preds = sorted(pred_by_class.get(cid, []), key=lambda item: item[1], reverse=True)
        matched = {image_idx: set() for image_idx in gt_by_class_image[cid]}
        tp: list[float] = []
        fp: list[float] = []
        for image_idx, _conf, box in preds:
            gt_boxes = gt_by_class_image[cid].get(image_idx, [])
            best_iou = 0.0
            best_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched.setdefault(image_idx, set()):
                    continue
                score = iou_xyxy(box, gt_box)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matched[image_idx].add(best_idx)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        if not preds:
            ap[cid] = 0.0
            continue
        cum_tp = []
        cum_fp = []
        running_tp = 0.0
        running_fp = 0.0
        for t, f in zip(tp, fp):
            running_tp += t
            running_fp += f
            cum_tp.append(running_tp)
            cum_fp.append(running_fp)
        recall = [value / gt_total for value in cum_tp]
        precision = [cum_tp[idx] / max(cum_tp[idx] + cum_fp[idx], 1e-9) for idx in range(len(cum_tp))]
        ap[cid] = ap_from_pr(recall, precision)
    return ap


def evaluate(records: list[dict[str, Any]], thresholds: dict[int, float]) -> EvalResult:
    counts = match_counts(records, thresholds, 0.5)
    ap50 = ap_per_class(records, thresholds, 0.5)
    ap_all = {iou: ap_per_class(records, thresholds, iou) for iou in IOU_THRESHOLDS}
    per_class: dict[int, dict[str, float]] = {}
    for cid, row in counts.items():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        ap95_values = [ap_all[iou].get(cid, 0.0) for iou in IOU_THRESHOLDS]
        per_class[cid] = {
            "precision": precision,
            "recall": recall,
            "ap50": ap50.get(cid, 0.0),
            "ap50_95": mean(ap95_values) if ap95_values else 0.0,
            **row,
        }
    valid = [row for row in per_class.values() if row["gt"] > 0]
    metrics = {
        "precision": mean([row["precision"] for row in valid]) if valid else 0.0,
        "recall": mean([row["recall"] for row in valid]) if valid else 0.0,
        "map50": mean([row["ap50"] for row in valid]) if valid else 0.0,
        "map50_95": mean([row["ap50_95"] for row in valid]) if valid else 0.0,
    }
    return EvalResult(metrics=metrics, per_class=per_class)


def score_metrics(metrics: dict[str, float], objective: str, reference: dict[str, float]) -> float:
    if objective == "balanced_score":
        return 0.25 * metrics["precision"] + 0.25 * metrics["recall"] + 0.25 * metrics["map50"] + 0.25 * metrics["map50_95"]
    if objective == "industrial_score":
        return 0.25 * metrics["precision"] + 0.30 * metrics["recall"] + 0.20 * metrics["map50"] + 0.25 * metrics["map50_95"]
    failures = constraint_failures(metrics, reference)
    if failures:
        deficit = sum(max(0.0, reference[key] - 0.01 - metrics[key]) for key in ("precision", "map50", "map50_95"))
        return -1000.0 - deficit
    return metrics["recall"] + 0.01 * metrics["map50_95"]


def constraint_failures(metrics: dict[str, float], reference: dict[str, float]) -> list[str]:
    failures = []
    if metrics["precision"] < reference["precision"] - 0.01:
        failures.append("precision_drop_gt_0.01")
    if metrics["map50"] < reference["map50"] - 0.01:
        failures.append("map50_drop_gt_0.01")
    if metrics["map50_95"] < reference["map50_95"] - 0.01:
        failures.append("map50_95_drop_gt_0.01")
    return failures


def greedy_search(records: list[dict[str, Any]], class_ids: list[int], reference: dict[str, float], objective: str) -> dict[str, Any]:
    thresholds = {cid: 0.25 for cid in class_ids}
    current = evaluate(records, thresholds)
    best_score = score_metrics(current.metrics, objective, reference)
    steps = []
    for pass_idx in range(2):
        improved = False
        for cid in class_ids:
            best_for_class = (best_score, thresholds[cid], current)
            for candidate in THRESHOLDS:
                trial_thresholds = dict(thresholds)
                trial_thresholds[cid] = candidate
                result = evaluate(records, trial_thresholds)
                score = score_metrics(result.metrics, objective, reference)
                if score > best_for_class[0] + 1e-9:
                    best_for_class = (score, candidate, result)
            if best_for_class[1] != thresholds[cid]:
                before = thresholds[cid]
                thresholds[cid] = best_for_class[1]
                best_score = best_for_class[0]
                current = best_for_class[2]
                improved = True
                steps.append({"pass": pass_idx + 1, "class_id": cid, "before": before, "after": thresholds[cid], "score": best_score})
        if not improved:
            break
    return {
        "objective": objective,
        "thresholds": {str(cid): thresholds[cid] for cid in class_ids},
        "metrics": current.metrics,
        "per_class": {str(cid): row for cid, row in current.per_class.items()},
        "score": best_score,
        "steps": steps,
        "constraint_failed": bool(constraint_failures(current.metrics, reference)),
        "failure_reasons": constraint_failures(current.metrics, reference),
    }


def resolve_weights(seed: int, group: str) -> Path:
    if group == "clean":
        metrics = load_json(ROOT / f"seed_{seed}/clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json")
        return Path(metrics["artifacts"]["best_pt"])
    return ROOT / f"seed_{seed}/catf_v2/train/weights/best.pt"


def run_api_validation_prediction(
    *,
    weights: Path,
    val_images_dir: Path,
    val_labels_dir: Path,
    output_dir: Path,
    imgsz: int,
    workers: int,
    device: str,
    conf: float,
    iou: float,
) -> None:
    """Generate validation prediction JSON using the current Python environment."""

    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    predict_root = output_dir / "predict_runs"
    if predict_root.exists():
        shutil.rmtree(predict_root)
    command = (
        "YOLO(...).predict("
        f"model={weights}, source={val_images_dir}, imgsz={imgsz}, conf={conf}, iou={iou}, "
        f"save_txt=True, save_conf=True, workers={workers}, device={device}, "
        f"project={predict_root}, name=pred, exist_ok=True)"
    )
    (output_dir / "predict_command.txt").write_text(command + "\n", encoding="utf-8")
    model = YOLO(str(weights))
    model.predict(
        source=str(val_images_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save_txt=True,
        save_conf=True,
        workers=workers,
        device=device,
        project=str(predict_root),
        name="pred",
        exist_ok=True,
        verbose=False,
    )
    predictions_dir = predict_root / "pred" / "labels"
    records = collect_validation_prediction_records(
        val_images_dir=val_images_dir,
        val_labels_dir=val_labels_dir,
        predictions_dir=predictions_dir,
    )
    record = {
        "stage": "validation_prediction",
        "status": "completed",
        "weights": str(weights.resolve()),
        "val_images_dir": str(val_images_dir.resolve()),
        "val_labels_dir": str(val_labels_dir.resolve()),
        "predictions_dir": str(predictions_dir.resolve()),
        "imgsz": int(imgsz),
        "workers": int(workers),
        "conf": float(conf),
        "iou": float(iou),
        "prediction_count": sum(len(item["predictions"]) for item in records),
        "image_count": len(records),
        "records": records,
    }
    write_json(output_dir / "validation_predictions.json", record)


def prediction_json(seed: int, group: str, *, force: bool = False) -> Path:
    out_dir = PRED_ROOT / f"seed_{seed}" / group
    pred_json = out_dir / "validation_predictions.json"
    if pred_json.exists() and not force:
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


def metric_delta(metrics: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: metrics[key] - reference[key] for key in METRIC_KEYS}


def threshold_changes(thresholds: dict[str, float], class_names: dict[int, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raised = []
    lowered = []
    for cid_s, value in sorted(thresholds.items(), key=lambda item: int(item[0])):
        cid = int(cid_s)
        row = {"class_id": cid, "class_name": class_names.get(cid, str(cid)), "threshold": value}
        if value > 0.25:
            raised.append(row)
        elif value < 0.25:
            lowered.append(row)
    return raised, lowered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-predict", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names_from_data_yaml(DATA)
    if set(CANONICAL_CLASS_NAMES).issubset(set(class_names)):
        class_names = {**class_names, **CANONICAL_CLASS_NAMES}
    output: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_conf": 0.10,
        "nms_iou": 0.5,
        "thresholds": THRESHOLDS,
        "note": "Post-hoc evaluator uses cached/generated prediction JSON and may not exactly match Ultralytics val mAP.",
        "seeds": {},
    }
    md = [
        "# CATF-v2 Post-hoc Per-Class Threshold Calibration",
        "",
        f"Generated: `{output['generated_at']}`",
        "",
        "This is prediction-only analysis. No training was run.",
        "",
        "The evaluator uses prediction JSON at `conf=0.10` and searches class thresholds from `0.10` to `0.70` in `0.05` steps.",
        "",
    ]

    pass_count = 0
    catf_constrained_rows = []
    for seed in SEEDS:
        output["seeds"][str(seed)] = {}
        records_by_group = {}
        default_by_group = {}
        class_ids = sorted(class_names)
        for group in ("clean", "catf_v2"):
            pred_path = prediction_json(seed, group, force=args.force_predict)
            records = load_json(pred_path)["records"]
            records_by_group[group] = records
            default_thresholds = {cid: 0.25 for cid in class_ids}
            default_eval = evaluate(records, default_thresholds)
            default_by_group[group] = default_eval
            output["seeds"][str(seed)][group] = {
                "prediction_json": str(pred_path),
                "default_threshold_metrics": default_eval.metrics,
            }
        clean_reference = default_by_group["clean"].metrics
        output["seeds"][str(seed)]["reference_for_constraints"] = clean_reference
        output["seeds"][str(seed)]["catf_v2_default_delta_vs_clean_default"] = metric_delta(default_by_group["catf_v2"].metrics, clean_reference)

        for group in ("clean", "catf_v2"):
            searches = {}
            for objective in ("constrained_score", "balanced_score", "industrial_score"):
                reference = clean_reference if group == "catf_v2" else default_by_group["clean"].metrics
                searches[objective] = greedy_search(records_by_group[group], class_ids, reference, objective)
            output["seeds"][str(seed)][group]["searches"] = searches

        constrained = output["seeds"][str(seed)]["catf_v2"]["searches"]["constrained_score"]
        delta = metric_delta(constrained["metrics"], clean_reference)
        raised, lowered = threshold_changes(constrained["thresholds"], class_names)
        passed = not constrained["constraint_failed"]
        pass_count += int(passed)
        row = {
            "seed": seed,
            "default_metrics": default_by_group["catf_v2"].metrics,
            "calibrated_metrics": constrained["metrics"],
            "delta_vs_clean_default": delta,
            "constraint_failed": constrained["constraint_failed"],
            "failure_reasons": constrained["failure_reasons"],
            "raise_threshold": raised,
            "lower_threshold": lowered,
        }
        catf_constrained_rows.append(row)
        output["seeds"][str(seed)]["catf_v2"]["constrained_summary"] = row

    output["summary"] = {
        "catf_v2_constraint_pass_count_after_constrained_calibration": pass_count,
        "seed0_precision_repair": "yes" if not output["seeds"]["0"]["catf_v2"]["constrained_summary"]["constraint_failed"] else "partial_or_no",
        "seed2_recall_repair": "yes" if output["seeds"]["2"]["catf_v2"]["constrained_summary"]["delta_vs_clean_default"]["recall"] >= 0 else "no",
    }
    write_json(REPORTS / "threshold_calibration_posthoc.json", output)

    md += [
        "## CATF-v2 Constrained Calibration Summary",
        "",
        "| seed | P | R | mAP50 | mAP50-95 | ΔP vs clean | ΔR vs clean | ΔmAP50 vs clean | ΔmAP50-95 vs clean | constraint_failed | raise threshold | lower threshold |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|---|",
    ]
    for row in catf_constrained_rows:
        metrics = row["calibrated_metrics"]
        delta = row["delta_vs_clean_default"]
        raised = [f"{item['class_id']}:{item['class_name']}->{item['threshold']:.2f}" for item in row["raise_threshold"]]
        lowered = [f"{item['class_id']}:{item['class_name']}->{item['threshold']:.2f}" for item in row["lower_threshold"]]
        md.append(
            f"| {row['seed']} | {f4(metrics['precision'])} | {f4(metrics['recall'])} | {f4(metrics['map50'])} | {f4(metrics['map50_95'])} | "
            f"{fd(delta['precision'])} | {fd(delta['recall'])} | {fd(delta['map50'])} | {fd(delta['map50_95'])} | "
            f"{str(row['constraint_failed']).lower()} | {raised} | {lowered} |"
        )
    md += [
        "",
        "## Objective-Specific Threshold Recommendations",
        "",
        "- `constrained_score` maximizes Recall while keeping P/mAP within the clean-reference tolerance.",
        "- `balanced_score` and `industrial_score` are more useful when the operator wants an explicit Precision/Recall balance.",
        "",
    ]
    for seed in SEEDS:
        md.append(f"### Seed {seed}")
        for objective in ("constrained_score", "balanced_score", "industrial_score"):
            result = output["seeds"][str(seed)]["catf_v2"]["searches"][objective]
            raised, lowered = threshold_changes(result["thresholds"], class_names)
            raised_s = [f"{item['class_id']}:{item['class_name']}->{item['threshold']:.2f}" for item in raised]
            lowered_s = [f"{item['class_id']}:{item['class_name']}->{item['threshold']:.2f}" for item in lowered]
            metrics = result["metrics"]
            md.append(
                f"- `{objective}`: P={f4(metrics['precision'])}, R={f4(metrics['recall'])}, "
                f"mAP50={f4(metrics['map50'])}, mAP50-95={f4(metrics['map50_95'])}, "
                f"constraint_failed={str(result['constraint_failed']).lower()}, "
                f"raise={raised_s}, lower={lowered_s}"
            )
        md.append("")
    md += [
        "",
        "## Answers",
        "",
        f"- Seed 0 Precision repair: `{output['summary']['seed0_precision_repair']}`.",
        f"- Seed 2 Recall repair: `{output['summary']['seed2_recall_repair']}`.",
        f"- CATF-v2 pass count after constrained calibration: `{pass_count}/3`.",
        "- Threshold calibration is best treated as a post-processing/deployment layer, not evidence that training augmentation itself is stable.",
        "- If seed 2 still fails Recall after calibration, its failure is training/trajectory degradation rather than a pure threshold issue.",
    ]
    write_md(REPORTS / "threshold_calibration_posthoc.md", md)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
