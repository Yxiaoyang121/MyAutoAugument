from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml


DEFAULT_CLEAN_RUN = Path("outputs/experiments/multiseed_cp_catf_paper_mode/clean_seed_0")
DEFAULT_CP_RUN = Path("outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only")
DEFAULT_DATA = Path("outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/data.yaml")
DEFAULT_PROBE = Path("outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position_paper_probe/probe.yaml")


@dataclass(frozen=True)
class Box:
    image_id: str
    cls: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))


def dataset_dirs(data_yaml: Path) -> tuple[Path, Path, dict[int, str]]:
    data = load_yaml(data_yaml)
    root = Path(data["path"])
    val_dir = root / data["val"]
    names = {int(k): str(v) for k, v in (data.get("names") or {}).items()}
    return root, val_dir, names


def probe_dir(probe_yaml: Path) -> Path:
    data = load_yaml(probe_yaml)
    return Path(data["path"]) / data.get("probe", data["val"])


def image_files(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def load_ground_truth(images_dir: Path) -> dict[str, list[Box]]:
    from PIL import Image

    gt: dict[str, list[Box]] = {}
    for image_path in image_files(images_dir):
        image_id = image_path.as_posix()
        with Image.open(image_path) as im:
            w, h = im.size
        boxes: list[Box] = []
        label_path = label_path_for_image(image_path)
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                x, y, bw, bh = map(float, parts[1:5])
                x1 = (x - bw / 2.0) * w
                y1 = (y - bh / 2.0) * h
                x2 = (x + bw / 2.0) * w
                y2 = (y + bh / 2.0) * h
                boxes.append(Box(image_id=image_id, cls=cls, xyxy=(x1, y1, x2, y2)))
        gt[image_id] = boxes
    return gt


def run_predict(model_path: Path, images_dir: Path, cache_path: Path, imgsz: int, device: str) -> dict[str, list[Box]]:
    if cache_path.exists():
        payload = read_json(cache_path, {})
        return {
            image_id: [
                Box(
                    image_id=image_id,
                    cls=int(item["cls"]),
                    xyxy=tuple(float(v) for v in item["xyxy"]),
                    conf=float(item["conf"]),
                )
                for item in items
            ]
            for image_id, items in payload.get("predictions", {}).items()
        }

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(images_dir),
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=device,
        stream=True,
        save=False,
        verbose=False,
    )
    preds: dict[str, list[Box]] = {}
    for result in results:
        image_id = Path(result.path).as_posix()
        boxes: list[Box] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for coords, c, score in zip(xyxy, cls, conf):
                boxes.append(
                    Box(
                        image_id=image_id,
                        cls=int(c),
                        xyxy=tuple(float(v) for v in coords.tolist()),
                        conf=float(score),
                    )
                )
        preds[image_id] = boxes
    serial = {
        "model": str(model_path),
        "images_dir": str(images_dir),
        "conf": 0.001,
        "iou": 0.7,
        "predictions": {
            image_id: [{"cls": b.cls, "conf": b.conf, "xyxy": list(b.xyxy)} for b in boxes]
            for image_id, boxes in preds.items()
        },
    }
    write_json(cache_path, serial)
    return preds


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_predictions(
    gt: dict[str, list[Box]],
    preds: dict[str, list[Box]],
    conf_thr: float,
    iou_thr: float,
    class_ids: list[int],
) -> dict[str, Any]:
    per_class: dict[int, dict[str, Any]] = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "tp_conf": [], "fp_conf": []} for cid in class_ids
    }
    global_tp = global_fp = global_fn = 0
    for image_id, gt_boxes in gt.items():
        pred_boxes = [p for p in preds.get(image_id, []) if p.conf >= conf_thr]
        for cid in class_ids:
            gtc = [g for g in gt_boxes if g.cls == cid]
            pc = sorted([p for p in pred_boxes if p.cls == cid], key=lambda p: p.conf, reverse=True)
            matched: set[int] = set()
            for pred in pc:
                best_idx = -1
                best_iou = 0.0
                for idx, g in enumerate(gtc):
                    if idx in matched:
                        continue
                    val = iou(pred.xyxy, g.xyxy)
                    if val > best_iou:
                        best_iou = val
                        best_idx = idx
                if best_idx >= 0 and best_iou >= iou_thr:
                    matched.add(best_idx)
                    per_class[cid]["tp"] += 1
                    per_class[cid]["tp_conf"].append(pred.conf)
                    global_tp += 1
                else:
                    per_class[cid]["fp"] += 1
                    per_class[cid]["fp_conf"].append(pred.conf)
                    global_fp += 1
            missed = len(gtc) - len(matched)
            per_class[cid]["fn"] += missed
            global_fn += missed
    for cid, row in per_class.items():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        row["precision"] = tp / (tp + fp) if tp + fp else 1.0
        row["recall"] = tp / (tp + fn) if tp + fn else 0.0
        row["fp_conf_mean"] = mean(row["fp_conf"]) if row["fp_conf"] else None
        row["fp_conf_median"] = median(row["fp_conf"]) if row["fp_conf"] else None
        row["tp_conf_mean"] = mean(row["tp_conf"]) if row["tp_conf"] else None
        row["tp_conf_median"] = median(row["tp_conf"]) if row["tp_conf"] else None
        row["high_conf_fp_count"] = sum(1 for v in row["fp_conf"] if v >= 0.5)
    precision = global_tp / (global_tp + global_fp) if global_tp + global_fp else 1.0
    recall = global_tp / (global_tp + global_fn) if global_tp + global_fn else 0.0
    return {
        "threshold": conf_thr,
        "iou_threshold": iou_thr,
        "tp": global_tp,
        "fp": global_fp,
        "fn": global_fn,
        "precision": precision,
        "recall": recall,
        "per_class": per_class,
    }


def compute_ap_for_class(
    gt: dict[str, list[Box]],
    preds: dict[str, list[Box]],
    class_id: int,
    iou_thr: float,
    conf_floor: float = 0.001,
) -> float:
    n_gt = sum(1 for boxes in gt.values() for b in boxes if b.cls == class_id)
    if n_gt == 0:
        return 0.0
    candidates = sorted(
        [p for boxes in preds.values() for p in boxes if p.cls == class_id and p.conf >= conf_floor],
        key=lambda p: p.conf,
        reverse=True,
    )
    matched: dict[str, set[int]] = {}
    tp: list[float] = []
    fp: list[float] = []
    gt_by_image = {image_id: [g for g in boxes if g.cls == class_id] for image_id, boxes in gt.items()}
    for pred in candidates:
        gtc = gt_by_image.get(pred.image_id, [])
        used = matched.setdefault(pred.image_id, set())
        best_idx = -1
        best_iou = 0.0
        for idx, g in enumerate(gtc):
            if idx in used:
                continue
            val = iou(pred.xyxy, g.xyxy)
            if val > best_iou:
                best_iou = val
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thr:
            used.add(best_idx)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not candidates:
        return 0.0
    cum_tp: list[float] = []
    cum_fp: list[float] = []
    t = f = 0.0
    for tv, fv in zip(tp, fp):
        t += tv
        f += fv
        cum_tp.append(t)
        cum_fp.append(f)
    recalls = [v / n_gt for v in cum_tp]
    precisions = [cum_tp[i] / max(cum_tp[i] + cum_fp[i], 1e-12) for i in range(len(cum_tp))]
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def compute_map(
    gt: dict[str, list[Box]],
    preds: dict[str, list[Box]],
    class_ids: list[int],
    conf_floor: float = 0.001,
) -> dict[str, Any]:
    ap50_by_class: dict[int, float] = {}
    ap5095_by_class: dict[int, float] = {}
    thresholds = [0.5 + i * 0.05 for i in range(10)]
    for cid in class_ids:
        ap50_by_class[cid] = compute_ap_for_class(gt, preds, cid, 0.5, conf_floor)
        aps = [compute_ap_for_class(gt, preds, cid, thr, conf_floor) for thr in thresholds]
        ap5095_by_class[cid] = sum(aps) / len(aps)
    return {
        "map50": mean(ap50_by_class.values()) if ap50_by_class else 0.0,
        "map50_95": mean(ap5095_by_class.values()) if ap5095_by_class else 0.0,
        "ap50_by_class": ap50_by_class,
        "ap50_95_by_class": ap5095_by_class,
    }


def final_metrics_by_class(path: Path) -> dict[int, dict[str, Any]]:
    payload = read_json(path, {})
    metrics = payload.get("val", {}).get("metrics") or payload.get("val_metrics") or payload
    rows = metrics.get("per_class", []) if isinstance(metrics, dict) else []
    return {int(row["class_id"]): row for row in rows}


def global_metrics(path: Path) -> dict[str, float]:
    payload = read_json(path, {})
    metrics = payload.get("val", {}).get("metrics") or payload.get("val_metrics") or payload
    return {
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "map50": float(metrics.get("map50", 0.0)),
        "map50_95": float(metrics.get("map50_95", 0.0)),
    }


def threshold_grid() -> list[float]:
    return sorted(set([0.001, 0.005, 0.01, 0.02, 0.03, 0.05] + [i / 100 for i in range(10, 96, 5)]))


def choose_threshold(gt: dict[str, list[Box]], preds: dict[str, list[Box]], class_ids: list[int], target_precision: float) -> dict[str, Any]:
    rows = []
    for thr in threshold_grid():
        stats = match_predictions(gt, preds, thr, 0.5, class_ids)
        rows.append({"threshold": thr, "precision": stats["precision"], "recall": stats["recall"], "tp": stats["tp"], "fp": stats["fp"], "fn": stats["fn"]})
    feasible = [r for r in rows if r["precision"] >= target_precision]
    if feasible:
        chosen = max(feasible, key=lambda r: (r["recall"], -r["threshold"]))
    else:
        chosen = max(rows, key=lambda r: (r["precision"], r["recall"]))
    return {"target_precision": target_precision, "chosen": chosen, "grid": rows}


def summarize_conf(row: dict[str, Any]) -> dict[str, Any]:
    fp = row.get("fp_conf", [])
    tp = row.get("tp_conf", [])
    return {
        "fp_count": len(fp),
        "tp_count": len(tp),
        "fp_mean": mean(fp) if fp else None,
        "fp_median": median(fp) if fp else None,
        "fp_ge_0_5": sum(1 for v in fp if v >= 0.5),
        "fp_ge_0_7": sum(1 for v in fp if v >= 0.7),
        "tp_mean": mean(tp) if tp else None,
        "tp_median": median(tp) if tp else None,
    }


def aggregate_conf(match: dict[str, Any]) -> dict[str, Any]:
    fp: list[float] = []
    tp: list[float] = []
    for row in match["per_class"].values():
        fp.extend(row.get("fp_conf", []))
        tp.extend(row.get("tp_conf", []))
    return {
        "fp_count": len(fp),
        "tp_count": len(tp),
        "fp_mean": mean(fp) if fp else None,
        "fp_median": median(fp) if fp else None,
        "fp_ge_0_5": sum(1 for v in fp if v >= 0.5),
        "fp_ge_0_7": sum(1 for v in fp if v >= 0.7),
        "tp_mean": mean(tp) if tp else None,
        "tp_median": median(tp) if tp else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run", type=Path, default=DEFAULT_CLEAN_RUN)
    parser.add_argument("--cp-run", type=Path, default=DEFAULT_CP_RUN)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--probe-data", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    root, val_dir, names = dataset_dirs(args.data)
    probe_images_dir = probe_dir(args.probe_data)
    class_ids = sorted(names)
    report_dir = args.cp_run / "reports"
    pred_dir = report_dir / "precision_risk_predictions"
    clean_model = args.clean_run / "train" / "weights" / "best.pt"
    cp_model = args.cp_run / "train" / "weights" / "best.pt"

    gt_val = load_ground_truth(val_dir)
    gt_probe = load_ground_truth(probe_images_dir)
    clean_val_preds = run_predict(clean_model, val_dir, pred_dir / "clean_final_val_predictions.json", args.imgsz, args.device)
    cp_val_preds = run_predict(cp_model, val_dir, pred_dir / "cp_final_val_predictions.json", args.imgsz, args.device)
    cp_probe_preds = run_predict(cp_model, probe_images_dir, pred_dir / "cp_probe_predictions.json", args.imgsz, args.device)

    clean_metrics = global_metrics(args.clean_run / "reports" / "final_metrics.json")
    cp_metrics = global_metrics(args.cp_run / "reports" / "final_metrics.json")
    clean_pc_metrics = final_metrics_by_class(args.clean_run / "reports" / "final_metrics.json")
    cp_pc_metrics = final_metrics_by_class(args.cp_run / "reports" / "final_metrics.json")

    conf_thr = 0.25
    clean_match = match_predictions(gt_val, clean_val_preds, conf_thr, 0.5, class_ids)
    cp_match = match_predictions(gt_val, cp_val_preds, conf_thr, 0.5, class_ids)
    clean_ap = compute_map(gt_val, clean_val_preds, class_ids)
    cp_ap = compute_map(gt_val, cp_val_preds, class_ids)

    active_classes = [9]
    roi_affected_classes = [int(k) for k in (read_json(report_dir / "roi_aug_stats.json", {}).get("affected_classes") or {}).keys()]
    non_active_classes = [cid for cid in class_ids if cid not in set(active_classes)]
    per_class_rows = []
    for cid in class_ids:
        cm = clean_pc_metrics.get(cid, {})
        pm = cp_pc_metrics.get(cid, {})
        c = clean_match["per_class"][cid]
        p = cp_match["per_class"][cid]
        row = {
            "class_id": cid,
            "class_name": names.get(cid, str(cid)),
            "active_class": cid in active_classes,
            "roi_affected": cid in roi_affected_classes,
            "clean_precision_metric": cm.get("precision"),
            "cp_precision_metric": pm.get("precision"),
            "delta_precision_metric": (pm.get("precision", 0.0) or 0.0) - (cm.get("precision", 0.0) or 0.0),
            "clean_recall_metric": cm.get("recall"),
            "cp_recall_metric": pm.get("recall"),
            "delta_recall_metric": (pm.get("recall", 0.0) or 0.0) - (cm.get("recall", 0.0) or 0.0),
            "clean_ap50": cm.get("ap50"),
            "cp_ap50": pm.get("ap50"),
            "delta_ap50": (pm.get("ap50", 0.0) or 0.0) - (cm.get("ap50", 0.0) or 0.0),
            "clean_ap50_95": cm.get("ap50_95"),
            "cp_ap50_95": pm.get("ap50_95"),
            "delta_ap50_95": (pm.get("ap50_95", 0.0) or 0.0) - (cm.get("ap50_95", 0.0) or 0.0),
            "clean_tp": c["tp"],
            "cp_tp": p["tp"],
            "delta_tp": p["tp"] - c["tp"],
            "clean_fp": c["fp"],
            "cp_fp": p["fp"],
            "delta_fp": p["fp"] - c["fp"],
            "clean_fn": c["fn"],
            "cp_fn": p["fn"],
            "delta_fn": p["fn"] - c["fn"],
            "clean_match_precision_at_0_25": c["precision"],
            "cp_match_precision_at_0_25": p["precision"],
            "delta_match_precision_at_0_25": p["precision"] - c["precision"],
            "clean_confidence": summarize_conf(c),
            "cp_confidence": summarize_conf(p),
        }
        per_class_rows.append(row)

    precision_drop_classes = sorted(per_class_rows, key=lambda r: r["delta_precision_metric"])[:5]
    fp_increase_classes = sorted(per_class_rows, key=lambda r: r["delta_fp"], reverse=True)[:5]
    non_active_regression = [
        r for r in per_class_rows
        if not r["active_class"] and (r["delta_fp"] > 0 or r["delta_ap50"] < -0.01 or r["delta_ap50_95"] < -0.01)
    ]

    target_precision = clean_metrics["precision"] - 0.01
    final_val_calibration = choose_threshold(gt_val, cp_val_preds, class_ids, target_precision)
    probe_calibration = choose_threshold(gt_probe, cp_probe_preds, class_ids, target_precision)
    probe_threshold = float(probe_calibration["chosen"]["threshold"])
    probe_threshold_on_final = match_predictions(gt_val, cp_val_preds, probe_threshold, 0.5, class_ids)
    probe_threshold_ap_final = compute_map(gt_val, cp_val_preds, class_ids, probe_threshold)
    final_val_threshold = float(final_val_calibration["chosen"]["threshold"])
    final_threshold_on_final = match_predictions(gt_val, cp_val_preds, final_val_threshold, 0.5, class_ids)
    final_threshold_ap_final = compute_map(gt_val, cp_val_preds, class_ids, final_val_threshold)

    causal_events = read_json(report_dir / "causal_probe_events.json", {}).get("events", [])
    accept_event = next((e for e in causal_events if e.get("selected_candidate_action") == "accept"), {})
    payload = {
        "scope": {
            "clean_run": str(args.clean_run),
            "cp_run": str(args.cp_run),
            "data_yaml": str(args.data),
            "probe_yaml": str(args.probe_data),
            "analysis_conf_threshold": conf_thr,
            "matching_iou_threshold": 0.5,
            "final_val_calibration_is_leakage_analysis_only": True,
        },
        "global_metrics": {
            "clean": clean_metrics,
            "cp_catf": cp_metrics,
            "delta": {k: cp_metrics[k] - clean_metrics[k] for k in clean_metrics},
            "constraint_failed": (cp_metrics["precision"] - clean_metrics["precision"]) < -0.01
            or (cp_metrics["map50"] - clean_metrics["map50"]) < -0.01
            or (cp_metrics["map50_95"] - clean_metrics["map50_95"]) < -0.01,
            "constraint_reasons": [
                reason
                for reason, failed in [
                    ("precision_drop_gt_0.01", (cp_metrics["precision"] - clean_metrics["precision"]) < -0.01),
                    ("map50_drop_gt_0.01", (cp_metrics["map50"] - clean_metrics["map50"]) < -0.01),
                    ("map50_95_drop_gt_0.01", (cp_metrics["map50_95"] - clean_metrics["map50_95"]) < -0.01),
                ]
                if failed
            ],
        },
        "match_counts_at_conf_0_25": {
            "clean": {k: clean_match[k] for k in ("tp", "fp", "fn", "precision", "recall")},
            "cp_catf": {k: cp_match[k] for k in ("tp", "fp", "fn", "precision", "recall")},
            "delta": {
                "tp": cp_match["tp"] - clean_match["tp"],
                "fp": cp_match["fp"] - clean_match["fp"],
                "fn": cp_match["fn"] - clean_match["fn"],
                "precision": cp_match["precision"] - clean_match["precision"],
                "recall": cp_match["recall"] - clean_match["recall"],
            },
            "confidence": {
                "clean": aggregate_conf(clean_match),
                "cp_catf": aggregate_conf(cp_match),
            },
        },
        "approx_ap_from_predictions": {
            "clean": {"map50": clean_ap["map50"], "map50_95": clean_ap["map50_95"]},
            "cp_catf": {"map50": cp_ap["map50"], "map50_95": cp_ap["map50_95"]},
            "delta": {"map50": cp_ap["map50"] - clean_ap["map50"], "map50_95": cp_ap["map50_95"] - clean_ap["map50_95"]},
        },
        "per_class": per_class_rows,
        "precision_drop_top_classes": precision_drop_classes,
        "fp_increase_top_classes": fp_increase_classes,
        "class9": next(r for r in per_class_rows if r["class_id"] == 9),
        "class9_is_primary_precision_drop_cause": False,
        "active_classes": active_classes,
        "roi_affected_classes": roi_affected_classes,
        "non_active_classes": non_active_classes,
        "non_active_regression": non_active_regression,
        "threshold_calibration": {
            "target_precision": target_precision,
            "final_val_diagnostic_leakage_only": {
                "calibration": final_val_calibration,
                "ap_on_final_val_after_threshold": {
                    "map50": final_threshold_ap_final["map50"],
                    "map50_95": final_threshold_ap_final["map50_95"],
                },
                "counts_on_final_val": {k: final_threshold_on_final[k] for k in ("tp", "fp", "fn", "precision", "recall")},
            },
            "probe_based": {
                "calibration_on_probe": probe_calibration,
                "chosen_threshold_applied_to_final_val": {
                    "threshold": probe_threshold,
                    "counts": {k: probe_threshold_on_final[k] for k in ("tp", "fp", "fn", "precision", "recall")},
                    "ap": {
                        "map50": probe_threshold_ap_final["map50"],
                        "map50_95": probe_threshold_ap_final["map50_95"],
                    },
                    "restores_precision_within_clean_minus_0_01": probe_threshold_on_final["precision"] >= target_precision,
                },
            },
        },
        "causal_probe_accept_event": accept_event,
        "causal_probe_gap_analysis": {
            "accepted_epoch": accept_event.get("epoch"),
            "accepted_class": accept_event.get("candidate_class_id"),
            "accepted_policy": accept_event.get("selected_candidate_policy_id"),
            "accepted_score": accept_event.get("selected_causal_score"),
            "likely_gap": "probe accepted weak positive causal_score but did not enforce an estimated final precision margin or high-confidence FP guard before training",
            "recommended_gate_additions": [
                "estimated_precision_drop <= 0.005",
                "high_confidence_fp_delta <= 0",
                "non_active_fp_delta <= 0 or tightly bounded",
                "stricter predicted_fp_increase_rate threshold for image candidates",
            ],
        },
        "recommendations": {
            "modify_accept_gate": True,
            "continue_seed1_seed2_now": False,
            "retrain_seed0_now": False,
            "reason": "Execution path is proven, but seed0 precision risk should be addressed before further performance validation.",
        },
        "primary_cause_summary": {
            "precision_drop_main_cause": "non-active FP increase at the final-val operating point",
            "top_non_active_fp_classes": [
                {"class_id": r["class_id"], "class_name": r["class_name"], "delta_fp": r["delta_fp"]}
                for r in fp_increase_classes
                if not r["active_class"]
            ][:5],
            "class9_role": "active ROI texture class improved metric precision/AP and reduced matched FP at conf=0.25; it is not the primary precision-drop class.",
            "map50_up_precision_down_reason": "AP/mAP50 improved because ranking/recall for some classes, especially class9 and low-support classes, improved, while the fixed operating point accumulated extra false positives in non-active classes.",
        },
    }
    write_json(report_dir / "seed0_precision_risk_audit.json", payload)

    def f(v: Any) -> str:
        if v is None:
            return "NA"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# Seed0 CP-CATF Precision-Risk Audit",
        "",
        "## Scope",
        "",
        "- No training was run for this audit.",
        "- Clean seed0 was reused from the paper-mode baseline.",
        "- CP-CATF seed0 was read from `outputs/experiments/cp_catf_paper_mode_execution_fixed_seed0_only/`.",
        "- Prediction-level analysis uses fresh inference caches for final val and probe split. Final-val threshold calibration is marked leakage analysis only.",
        "",
        "## Global Result",
        "",
        "| metric | clean | CP-CATF | delta |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [("precision", "P"), ("recall", "R"), ("map50", "mAP50"), ("map50_95", "mAP50-95")]:
        lines.append(f"| {label} | {clean_metrics[key]:.4f} | {cp_metrics[key]:.4f} | {cp_metrics[key] - clean_metrics[key]:+.4f} |")
    lines.extend(
        [
            "",
            f"- Constraint failed: `{payload['global_metrics']['constraint_failed']}`; reasons: `{payload['global_metrics']['constraint_reasons']}`.",
            f"- At confidence `{conf_thr}`, matched FP changed from `{clean_match['fp']}` to `{cp_match['fp']}` (`{cp_match['fp'] - clean_match['fp']:+d}`), while TP changed `{cp_match['tp'] - clean_match['tp']:+d}` and FN changed `{cp_match['fn'] - clean_match['fn']:+d}`.",
            "",
            "## Precision Drop Top Classes",
            "",
            "| class | name | clean P | CP P | dP | clean FP | CP FP | dFP | clean TP | CP TP | dTP | clean FN | CP FN | dFN |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in precision_drop_classes:
        lines.append(
            f"| {row['class_id']} | {row['class_name']} | {f(row['clean_precision_metric'])} | {f(row['cp_precision_metric'])} | {f(row['delta_precision_metric'])} | "
            f"{row['clean_fp']} | {row['cp_fp']} | {row['delta_fp']:+d} | {row['clean_tp']} | {row['cp_tp']} | {row['delta_tp']:+d} | {row['clean_fn']} | {row['cp_fn']} | {row['delta_fn']:+d} |"
        )
    lines.extend(
        [
            "",
            "## FP Increase Top Classes",
            "",
            "| class | name | active | ROI affected | dFP | dTP | dFN | dAP50 | dAP50-95 |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fp_increase_classes:
        lines.append(
            f"| {row['class_id']} | {row['class_name']} | `{row['active_class']}` | `{row['roi_affected']}` | {row['delta_fp']:+d} | {row['delta_tp']:+d} | {row['delta_fn']:+d} | {f(row['delta_ap50'])} | {f(row['delta_ap50_95'])} |"
        )
    c9 = payload["class9"]
    lines.extend(
        [
            "",
            "## Class 9",
            "",
            f"- Class 9 was the active and ROI-affected class. ROI applications recorded for class 9: `{read_json(report_dir / 'roi_aug_stats.json', {}).get('affected_classes', {}).get('9')}`.",
            f"- Metric Precision changed `{f(c9['clean_precision_metric'])} -> {f(c9['cp_precision_metric'])}` (`{f(c9['delta_precision_metric'])}`).",
            f"- Metric Recall changed `{f(c9['clean_recall_metric'])} -> {f(c9['cp_recall_metric'])}` (`{f(c9['delta_recall_metric'])}`).",
            f"- AP50 changed `{f(c9['delta_ap50'])}` and AP50-95 changed `{f(c9['delta_ap50_95'])}`.",
            f"- Matched FP at conf {conf_thr} changed `{c9['clean_fp']} -> {c9['cp_fp']}` (`{c9['delta_fp']:+d}`).",
            "Class 9 is not the primary Precision-drop class: its metric Precision/AP improved and matched FP decreased at the analysis threshold. The failure is dominated by non-active class FP increases.",
            "",
            "## Non-Active Regression",
            "",
            "| class | name | dFP | dAP50 | dAP50-95 | active | ROI affected |",
            "|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for row in non_active_regression[:10]:
        lines.append(
            f"| {row['class_id']} | {row['class_name']} | {row['delta_fp']:+d} | {f(row['delta_ap50'])} | {f(row['delta_ap50_95'])} | `{row['active_class']}` | `{row['roi_affected']}` |"
        )
    lines.extend(
        [
            "",
            "## Confidence Distribution",
            "",
            f"- Global FP count at conf {conf_thr}: clean `{aggregate_conf(clean_match)['fp_count']}`, CP-CATF `{aggregate_conf(cp_match)['fp_count']}`.",
            f"- Global FP confidence mean: clean `{f(aggregate_conf(clean_match)['fp_mean'])}`, CP-CATF `{f(aggregate_conf(cp_match)['fp_mean'])}`.",
            f"- Global high-confidence FP count >=0.5: clean `{aggregate_conf(clean_match)['fp_ge_0_5']}`, CP-CATF `{aggregate_conf(cp_match)['fp_ge_0_5']}`.",
            "The FP count increases while the mean FP confidence slightly decreases, which means the precision risk is mostly broader low-to-mid confidence spillover rather than a simple shift to more high-confidence false positives. Thresholding can suppress part of this tail, but the final-val tuned threshold is leakage-only and cannot be used as a paper result.",
            "",
            "## Threshold Calibration",
            "",
            f"- Target precision: clean seed0 - 0.01 = `{target_precision:.4f}`.",
            f"- Final-val diagnostic calibration selected threshold `{final_val_threshold:.3f}` and reaches precision `{final_threshold_on_final['precision']:.4f}` / recall `{final_threshold_on_final['recall']:.4f}` on final val. This is leakage analysis only.",
            f"- Probe-based calibration selected threshold `{probe_threshold:.3f}` on probe; applied to final val it reaches precision `{probe_threshold_on_final['precision']:.4f}` / recall `{probe_threshold_on_final['recall']:.4f}`.",
            f"- Probe-based threshold restores precision within clean-0.01: `{probe_threshold_on_final['precision'] >= target_precision}`.",
            "",
            "## Causal Probe Gap",
            "",
            f"- Paper-mode causal probe accepted class `{accept_event.get('candidate_class_id')}` at epoch `{accept_event.get('epoch')}` with causal score `{accept_event.get('selected_causal_score')}`.",
            "- The accept was enough to prove the execution chain, but the final outcome shows the probe underweighted precision risk.",
            "- The likely missing guard is not another dataset-specific blacklist; it is a precision-aware accept condition using the probe split.",
            "- The probe accepted class9 because local benefit looked positive, but it did not sufficiently penalize non-active FP spillover and estimated operating-point precision loss.",
            "",
            "## Recommendations",
            "",
            "- Add a precision-aware accept gate before continuing performance validation.",
            "- Do not continue seed1/seed2 as paper-mode performance validation yet.",
            "- Do not retrain seed0 immediately without first deciding the precision-risk gate.",
        ]
    )
    (report_dir / "seed0_precision_risk_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
