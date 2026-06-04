"""Validate CATF-v2-RC thresholds through the official YOLO val/predict path.

This is validation/inference-only. It does not train or mutate weights.
Ultralytics `YOLO.val` is used for fixed CATF-v2 official validation, and
Ultralytics `YOLO.predict` is used to create prediction labels for the
per-class threshold post-processing evaluator.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402

from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402
from AutoAugment.catf_v2.per_class_thresholds import constraint_failures, metric_delta  # noqa: E402
from scripts.evaluate_catf_v2_threshold_posthoc import CANONICAL_CLASS_NAMES, evaluate, f4, fd, run_api_validation_prediction  # noqa: E402
from scripts.train_yolo_default_with_inloop_feedback import resolve_val_image_label_dirs  # noqa: E402


DATA = Path("outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml")
FIXED_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2_fixed")
OLD_ROOT = Path("outputs/experiments/multiseed_clean_yolo_default_vs_catf_v2")
REPORTS = FIXED_ROOT / "reports"
OUT = REPORTS / "catf_v2_rc_official_path_validation"
THRESHOLD_CONFIG = REPORTS / "catf_v2_rc_per_class_thresholds.json"
SEEDS = [0, 1, 2]
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixed_run_root(seed: int) -> Path:
    if seed == 1:
        return Path("outputs/experiments/catf_v2_fixed_seed1_50ep")
    return FIXED_ROOT / f"seed_{seed}/catf_v2"


def fixed_weights(seed: int) -> Path:
    return fixed_run_root(seed) / "train/weights/best.pt"


def clean_metrics_path(seed: int) -> Path:
    return OLD_ROOT / f"seed_{seed}/clean_native_yolo_default/reports/clean_native_yolo_default_metrics.json"


def fixed_saved_metrics_path(seed: int) -> Path:
    return fixed_run_root(seed) / "reports/final_metrics.json"


def clean_prediction_json(seed: int) -> Path:
    return OLD_ROOT / f"reports/threshold_posthoc_predictions/seed_{seed}/clean/validation_predictions.json"


def clean_official_metrics(seed: int) -> dict[str, float]:
    row = read_json(clean_metrics_path(seed))["metrics"]
    return {key: float(row[key]) for key in METRIC_KEYS}


def fixed_saved_official_metrics(seed: int) -> dict[str, float]:
    row = read_json(fixed_saved_metrics_path(seed))["constraint_scoring"]["metrics"]
    return {key: float(row[key]) for key in METRIC_KEYS}


def metrics_from_results_dict(results_dict: dict[str, Any]) -> dict[str, float]:
    aliases = {
        "precision": ("metrics/precision(B)", "metrics/precision"),
        "recall": ("metrics/recall(B)", "metrics/recall"),
        "map50": ("metrics/mAP50(B)", "metrics/mAP50"),
        "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95"),
    }
    metrics: dict[str, float] = {}
    for key, names in aliases.items():
        for name in names:
            if name in results_dict:
                metrics[key] = float(results_dict[name])
                break
        else:
            raise KeyError(f"Missing {key} in Ultralytics results_dict keys={sorted(results_dict)}")
    return metrics


def run_fixed_official_val(seed: int, *, force: bool = False) -> dict[str, Any]:
    output_dir = OUT / f"seed_{seed}/fixed_val"
    metrics_json = output_dir / "official_val_metrics.json"
    if metrics_json.exists() and not force:
        return read_json(metrics_json)
    weights = fixed_weights(seed)
    if not weights.exists():
        raise FileNotFoundError(weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = (
        f"YOLO({weights}).val(data={DATA}, imgsz=1024, batch=2, workers=0, device=0, split=val, "
        f"project={output_dir / 'runs'}, name=val, exist_ok=True)"
    )
    (output_dir / "official_val_command.txt").write_text(command + "\n", encoding="utf-8")
    model = YOLO(str(weights))
    result = model.val(
        data=str(DATA),
        imgsz=1024,
        batch=2,
        workers=0,
        device="0",
        split="val",
        project=str(output_dir / "runs"),
        name="val",
        exist_ok=True,
        verbose=False,
    )
    payload = {
        "seed": seed,
        "weights": str(weights.resolve()),
        "data": str(DATA.resolve()),
        "status": "completed",
        "metrics": metrics_from_results_dict(dict(result.results_dict)),
        "results_dict": {str(k): float(v) if isinstance(v, (int, float)) else v for k, v in dict(result.results_dict).items()},
        "save_dir": str(Path(result.save_dir).resolve()) if getattr(result, "save_dir", None) else None,
    }
    write_json(metrics_json, payload)
    return payload


def run_fixed_official_predict(seed: int, *, force: bool = False) -> Path:
    output_dir = OUT / f"seed_{seed}/fixed_predict"
    pred_json = output_dir / "validation_predictions.json"
    if pred_json.exists() and not force:
        return pred_json
    weights = fixed_weights(seed)
    if not weights.exists():
        raise FileNotFoundError(weights)
    val_images, val_labels = resolve_val_image_label_dirs(DATA)
    run_api_validation_prediction(
        weights=weights,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        output_dir=output_dir,
        imgsz=1024,
        workers=0,
        device="0",
        conf=0.10,
        iou=0.5,
    )
    return pred_json


def load_rc_thresholds() -> dict[int, float]:
    payload = read_json(THRESHOLD_CONFIG)
    return {int(class_id): float(row["threshold"]) for class_id, row in payload["classes"].items()}


def load_default_thresholds(records: list[dict[str, Any]]) -> dict[int, float]:
    class_ids = sorted(
        {
            int(item["class_id"])
            for record in records
            for item in record.get("ground_truth", []) + record.get("predictions", [])
        }
    )
    return {class_id: 0.25 for class_id in class_ids}


def validate_seed(seed: int, thresholds: dict[int, float]) -> dict[str, Any]:
    fixed_val = run_fixed_official_val(seed)
    fixed_pred_path = run_fixed_official_predict(seed)
    clean_pred_path = clean_prediction_json(seed)
    if not clean_pred_path.exists():
        raise FileNotFoundError(f"Missing clean official-predict cache: {clean_pred_path}")

    clean_records = read_json(clean_pred_path)["records"]
    fixed_records = read_json(fixed_pred_path)["records"]
    clean_default = evaluate(clean_records, load_default_thresholds(clean_records))
    fixed_default = evaluate(fixed_records, load_default_thresholds(fixed_records))
    fixed_rc = evaluate(fixed_records, thresholds)
    fixed_default_failures = constraint_failures(fixed_default.metrics, clean_default.metrics)
    failures = constraint_failures(fixed_rc.metrics, clean_default.metrics)
    return {
        "seed": seed,
        "clean_official_val_metrics": clean_official_metrics(seed),
        "fixed_saved_official_val_metrics": fixed_saved_official_metrics(seed),
        "fixed_rerun_official_val_metrics": fixed_val["metrics"],
        "official_val_delta_fixed_vs_clean": metric_delta(fixed_val["metrics"], clean_official_metrics(seed)),
        "clean_prediction_json": str(clean_pred_path),
        "fixed_prediction_json": str(fixed_pred_path),
        "official_predict_posthoc": {
            "clean_default_0.25": clean_default.metrics,
            "fixed_default_0.25": fixed_default.metrics,
            "fixed_catf_v2_rc_thresholds": fixed_rc.metrics,
            "fixed_default_delta_vs_clean_default": metric_delta(fixed_default.metrics, clean_default.metrics),
            "fixed_default_constraint_failed": bool(fixed_default_failures),
            "fixed_default_failure_reasons": fixed_default_failures,
            "fixed_rc_delta_vs_clean_default": metric_delta(fixed_rc.metrics, clean_default.metrics),
            "fixed_rc_constraint_failed": bool(failures),
            "fixed_rc_failure_reasons": failures,
            "fixed_rc_per_class": {str(cid): row for cid, row in fixed_rc.per_class.items()},
        },
    }


def write_report(payload: dict[str, Any], class_names: dict[int, str]) -> None:
    write_json(REPORTS / "catf_v2_rc_official_path_validation.json", payload)
    thresholds = payload["thresholds"]
    lines = [
        "# CATF-v2-RC Official Path Validation",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "No training was run. Fixed CATF-v2 `best.pt` checkpoints were evaluated with Ultralytics `YOLO.val` and `YOLO.predict`.",
        "",
        "Important: per-class thresholds cannot be represented directly in Ultralytics `YOLO.val`, so CATF-v2-RC is evaluated as `YOLO.predict(conf=0.10) + per-class threshold post-processing + the same cached-prediction evaluator used for threshold selection`.",
        "",
        "## Per-Class Thresholds",
        "",
        "| class | threshold |",
        "|---|---:|",
    ]
    for class_id, threshold in sorted(thresholds.items(), key=lambda item: int(item[0])):
        cid = int(class_id)
        lines.append(f"| {cid}:{class_names.get(cid, str(cid))} | {float(threshold):.2f} |")
    lines += [
        "",
        "## Official Ultralytics Val: Clean vs Fixed CATF-v2",
        "",
        "| seed | clean P | clean R | clean mAP50 | clean mAP50-95 | fixed P | fixed R | fixed mAP50 | fixed mAP50-95 | dP | dR | dM50 | dM95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = payload["seeds"][str(seed)]
        clean = row["clean_official_val_metrics"]
        fixed = row["fixed_rerun_official_val_metrics"]
        delta = row["official_val_delta_fixed_vs_clean"]
        lines.append(
            f"| {seed} | {f4(clean['precision'])} | {f4(clean['recall'])} | {f4(clean['map50'])} | {f4(clean['map50_95'])} | "
            f"{f4(fixed['precision'])} | {f4(fixed['recall'])} | {f4(fixed['map50'])} | {f4(fixed['map50_95'])} | "
            f"{fd(delta['precision'])} | {fd(delta['recall'])} | {fd(delta['map50'])} | {fd(delta['map50_95'])} |"
        )
    lines += [
        "",
        "## Official Predict + Post-Processing: Clean vs Fixed vs CATF-v2-RC",
        "",
        "| seed | group | P | R | mAP50 | mAP50-95 | dP vs clean | dR vs clean | dM50 vs clean | dM95 vs clean | constraint_failed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in SEEDS:
        row = payload["seeds"][str(seed)]["official_predict_posthoc"]
        clean = row["clean_default_0.25"]
        fixed = row["fixed_default_0.25"]
        fixed_delta = row["fixed_default_delta_vs_clean_default"]
        rc = row["fixed_catf_v2_rc_thresholds"]
        rc_delta = row["fixed_rc_delta_vs_clean_default"]
        lines.append(f"| {seed} | clean_default_0.25 | {f4(clean['precision'])} | {f4(clean['recall'])} | {f4(clean['map50'])} | {f4(clean['map50_95'])} | +0.0000 | +0.0000 | +0.0000 | +0.0000 | false |")
        lines.append(
            f"| {seed} | fixed_default_0.25 | {f4(fixed['precision'])} | {f4(fixed['recall'])} | {f4(fixed['map50'])} | {f4(fixed['map50_95'])} | "
            f"{fd(fixed_delta['precision'])} | {fd(fixed_delta['recall'])} | {fd(fixed_delta['map50'])} | {fd(fixed_delta['map50_95'])} | {str(row['fixed_default_constraint_failed']).lower()} |"
        )
        lines.append(
            f"| {seed} | fixed_catf_v2_rc | {f4(rc['precision'])} | {f4(rc['recall'])} | {f4(rc['map50'])} | {f4(rc['map50_95'])} | "
            f"{fd(rc_delta['precision'])} | {fd(rc_delta['recall'])} | {fd(rc_delta['map50'])} | {fd(rc_delta['map50_95'])} | {str(row['fixed_rc_constraint_failed']).lower()} |"
        )
    lines += [
        "",
        "## Constraint Summary",
        "",
        f"- CATF-v2-RC official-predict post-processing pass count: `{payload['summary']['rc_constraint_pass_count']}/3`.",
        f"- RC all seeds pass: `{str(payload['summary']['rc_all_seeds_pass']).lower()}`.",
        "- Constraint reference for RC is the clean model's `YOLO.predict(conf=0.10)` posthoc default-threshold evaluator for the same seed.",
    ]
    write_md(REPORTS / "catf_v2_rc_official_path_validation.md", lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names_from_data_yaml(DATA)
    if set(CANONICAL_CLASS_NAMES).issubset(set(class_names)):
        class_names = {**class_names, **CANONICAL_CLASS_NAMES}
    thresholds = load_rc_thresholds()
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data": str(DATA.resolve()),
        "threshold_config": str(THRESHOLD_CONFIG.resolve()),
        "thresholds": {str(k): v for k, v in sorted(thresholds.items())},
        "seeds": {},
    }
    pass_count = 0
    for seed in SEEDS:
        seed_payload = validate_seed(seed, thresholds)
        payload["seeds"][str(seed)] = seed_payload
        pass_count += int(not seed_payload["official_predict_posthoc"]["fixed_rc_constraint_failed"])
    payload["summary"] = {
        "rc_constraint_pass_count": pass_count,
        "rc_all_seeds_pass": pass_count == len(SEEDS),
    }
    write_report(payload, class_names)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
