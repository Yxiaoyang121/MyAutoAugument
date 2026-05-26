from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostic_pipeline import run_error_diagnosis, run_validation_prediction  # noqa: E402
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402


PYTHON_EXE = Path(r"D:\Anaconda\envs\pytorch\python.exe")
DATA_YAML = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml"
RUN_ID = "20260518_tiled1024_safe_no_ok_position_yolo_default_diagnosis_constrained_50ep"
RUN_ROOT = PROJECT_ROOT / "outputs/experiments" / RUN_ID
RUNS_ROOT = RUN_ROOT / "runs"
REPORTS_ROOT = RUN_ROOT / "reports"
POLICY_ROOT = PROJECT_ROOT / "configs/online_policies"
CUSTOM_YOLO_LIKE_RUN = (
    PROJECT_ROOT
    / "outputs/experiments/20260518_tiled1024_safe_no_ok_position_custom_yolo_like_base_yolo11n_50ep"
)

METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
PRECISION_DROP_LIMIT = -0.02
MAP50_DROP_LIMIT = -0.01
MAP50_95_DROP_LIMIT = -0.01


@dataclass(frozen=True)
class ExperimentGroup:
    key: str
    label: str
    policy_path: Path
    run_id: str
    train: bool = True
    run_dir: Path | None = None

    @property
    def output_dir(self) -> Path:
        return self.run_dir if self.run_dir is not None else RUNS_ROOT / self.run_id


GROUPS = [
    ExperimentGroup(
        key="yolo_default",
        label="YOLO default",
        policy_path=POLICY_ROOT / "yolo_default_passthrough_policy.json",
        run_id="yolo_default_seed42",
    ),
    ExperimentGroup(
        key="diagnosis_light",
        label="YOLO default + diagnosis_light",
        policy_path=POLICY_ROOT / "diagnosis_light_policy.json",
        run_id="diagnosis_light",
    ),
    ExperimentGroup(
        key="diagnosis_precision_safe",
        label="YOLO default + diagnosis_precision_safe",
        policy_path=POLICY_ROOT / "diagnosis_precision_safe_policy.json",
        run_id="diagnosis_precision_safe",
    ),
    ExperimentGroup(
        key="diagnosis_recall_safe",
        label="YOLO default + diagnosis_recall_safe",
        policy_path=POLICY_ROOT / "diagnosis_recall_safe_policy.json",
        run_id="diagnosis_recall_safe",
    ),
    ExperimentGroup(
        key="custom_yolo_like_base_old",
        label="custom_yolo_like_base old control",
        policy_path=POLICY_ROOT / "yolo_like_base_policy.json",
        run_id="custom_yolo_like_base_old",
        train=False,
        run_dir=CUSTOM_YOLO_LIKE_RUN,
    ),
]


def main() -> None:
    args = parse_args()
    configure_environment()
    prepare_dirs()
    selected = select_groups(args.groups)
    write_json(
        RUN_ROOT / "configs/experiment_config.json",
        {
            "run_id": RUN_ID,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data": rel(DATA_YAML),
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": str(args.device),
            "seed": args.seed,
            "constraint_reference": "yolo_default",
            "constraint_limits": {
                "precision_min_delta": PRECISION_DROP_LIMIT,
                "map50_min_delta": MAP50_DROP_LIMIT,
                "map50_95_min_delta": MAP50_95_DROP_LIMIT,
            },
            "groups": [group_to_dict(group) for group in GROUPS],
        },
    )

    for group in selected:
        if group.train and not args.skip_training:
            run_training_group(group, args=args, force=args.force_training)
        if not args.skip_diagnosis:
            run_group_diagnosis(group, args=args, force=args.force_diagnosis)

    if selected_keys(selected) == selected_keys(GROUPS):
        aggregate = build_aggregate_report(GROUPS)
        write_aggregate_outputs(aggregate)
        update_state_docs(aggregate)
        export_project_snapshot()
        print(json.dumps(aggregate["summary"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"completed_groups": [group.key for group in selected]}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO default + diagnosis-constrained augmentation experiments.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional subset of group keys to run.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-diagnosis", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    parser.add_argument("--force-diagnosis", action="store_true")
    return parser.parse_args()


def configure_environment() -> None:
    scripts_dir = PYTHON_EXE.parent / "Scripts"
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    yolo_config = RUN_ROOT / "configs/ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())


def prepare_dirs() -> None:
    for path in [RUN_ROOT / "configs", RUN_ROOT / "logs", REPORTS_ROOT, RUNS_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def select_groups(keys: list[str] | None) -> list[ExperimentGroup]:
    if not keys:
        return GROUPS
    allowed = {group.key: group for group in GROUPS}
    missing = [key for key in keys if key not in allowed]
    if missing:
        raise ValueError(f"unknown group key(s): {missing}; allowed={sorted(allowed)}")
    return [allowed[key] for key in keys]


def selected_keys(groups: list[ExperimentGroup]) -> set[str]:
    return {group.key for group in groups}


def run_training_group(group: ExperimentGroup, *, args: argparse.Namespace, force: bool = False) -> None:
    run_dir = group.output_dir
    if is_training_complete(run_dir) and not force:
        print(f"[skip] {group.key}: completed training already exists at {run_dir}")
        return
    command = [
        str(PYTHON_EXE),
        str(PROJECT_ROOT / "scripts/train_yolo_online_aug.py"),
        "--model",
        str(args.model),
        "--data",
        str(args.data),
        "--policy",
        str(group.policy_path),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--run-id",
        group.run_id,
        "--project",
        str(RUNS_ROOT),
        "--no-disable-yolo-aug",
        "--save-preview",
        "--preview-count",
        str(args.preview_count),
        "--skip-doc-update",
    ]
    log_prefix = RUN_ROOT / "logs" / group.key
    write_text(log_prefix.with_suffix(".train.command.txt"), subprocess.list2cmdline(command))
    print(f"[train:start] {group.key} -> {run_dir}")
    start = time.time()
    with log_prefix.with_suffix(".train.stdout.log").open("w", encoding="utf-8", errors="replace") as stdout:
        with log_prefix.with_suffix(".train.stderr.log").open("w", encoding="utf-8", errors="replace") as stderr:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout,
                stderr=stderr,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=os.environ.copy(),
            )
    wall = time.time() - start
    write_text(log_prefix.with_suffix(".train.wall_seconds.txt"), f"{wall:.3f}")
    write_text(log_prefix.with_suffix(".train.exitcode"), str(completed.returncode))
    if completed.returncode != 0:
        raise RuntimeError(f"training failed for {group.key}; see {log_prefix.with_suffix('.train.stderr.log')}")
    print(f"[train:done] {group.key} wall_seconds={wall:.1f}")


def is_training_complete(run_dir: Path) -> bool:
    stats_path = run_dir / "reports/online_aug_stats.json"
    if not stats_path.exists():
        return False
    stats = read_json(stats_path)
    return (
        bool(stats.get("train_success"))
        and bool(stats.get("val_success"))
        and (run_dir / "train/results.csv").exists()
        and ((run_dir / "train/weights/best.pt").exists() or (run_dir / "train/weights/last.pt").exists())
    )


def run_group_diagnosis(group: ExperimentGroup, *, args: argparse.Namespace, force: bool = False) -> None:
    run_dir = group.output_dir
    diagnosis_json = run_dir / "diagnosis/diagnosis.json"
    if diagnosis_json.exists() and not force:
        print(f"[skip] {group.key}: diagnosis already exists")
        return
    weights = best_weight(run_dir)
    if weights is None:
        raise FileNotFoundError(f"missing weights for diagnosis group {group.key}: {run_dir}")
    val_images, val_labels = resolve_val_image_label_dirs(Path(args.data))
    class_names = load_class_names_from_data_yaml(args.data)
    print(f"[diagnosis:start] {group.key}")
    prediction = run_validation_prediction(
        weights=weights,
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        output_dir=run_dir / "diagnosis_prediction",
        imgsz=int(args.imgsz),
        workers=int(args.workers),
        device=str(args.device),
        conf=0.25,
        iou=0.5,
        dry_run=False,
    )
    diagnosis = run_error_diagnosis(
        val_images_dir=val_images,
        val_labels_dir=val_labels,
        predictions_dir=prediction["predictions_dir"],
        output_dir=run_dir / "diagnosis",
        class_names=class_names,
        match_iou=0.5,
        localization_weak_iou=0.3,
    )
    metrics = load_group_metrics(group)
    write_markdown(run_dir / "reports/diagnosis_report.md", build_group_diagnosis_report(group, metrics, diagnosis))
    print(f"[diagnosis:done] {group.key}")


def best_weight(run_dir: Path) -> Path | None:
    for path in [run_dir / "train/weights/best.pt", run_dir / "train/weights/last.pt"]:
        if path.exists():
            return path
    return None


def resolve_val_image_label_dirs(data_yaml: Path) -> tuple[Path, Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig")) or {}
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    val_value = Path(str(data["val"]))
    val_images = val_value if val_value.is_absolute() else root / val_value
    try:
        rel_parts = list(val_images.relative_to(root).parts)
        if rel_parts and rel_parts[0] == "images":
            val_labels = root.joinpath("labels", *rel_parts[1:])
        else:
            val_labels = val_images.parent.parent / "labels" / val_images.name
    except ValueError:
        parts = list(val_images.parts)
        if "images" in parts:
            index = parts.index("images")
            parts[index] = "labels"
            val_labels = Path(*parts)
        else:
            val_labels = val_images.parent.parent / "labels" / val_images.name
    return val_images, val_labels


def load_group_metrics(group: ExperimentGroup) -> dict[str, Any]:
    if group.key == "custom_yolo_like_base_old":
        payload = read_json(group.output_dir / "reports/custom_yolo_like_base_50ep_metrics.json")
        return compact_metrics(payload.get("final_metrics", {}))
    stats = read_json(group.output_dir / "reports/online_aug_stats.json")
    return compact_metrics(stats.get("val_metrics", {}))


def load_group_stats(group: ExperimentGroup) -> dict[str, Any]:
    stats_path = group.output_dir / "reports/online_aug_stats.json"
    if stats_path.exists():
        return read_json(stats_path)
    return {}


def load_group_policy(group: ExperimentGroup) -> dict[str, Any]:
    policy_path = group.output_dir / "configs/policy.json"
    if policy_path.exists():
        return read_json(policy_path)
    return read_json(group.policy_path)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in ["images", "instances", *METRIC_KEYS]}


def build_aggregate_report(groups: list[ExperimentGroup]) -> dict[str, Any]:
    reference_metrics = load_group_metrics(groups[0])
    rows: list[dict[str, Any]] = []
    for group in groups:
        metrics = load_group_metrics(group)
        delta = metric_delta(metrics, reference_metrics)
        stats = load_group_stats(group)
        diagnosis = load_group_diagnosis(group)
        row = {
            "key": group.key,
            "label": group.label,
            "run_dir": str(group.output_dir.resolve()),
            "policy": str(group.policy_path),
            "metrics": metrics,
            "delta_vs_yolo_default": delta,
            "constraints": evaluate_constraints(delta),
            "diagnosis": summarize_diagnosis(diagnosis),
            "artifacts": group_artifacts(group),
            "online_aug_stats": {
                "train_image_count": stats.get("train_image_count"),
                "fixed_augmented_dataset_generated": stats.get("fixed_augmented_dataset_generated"),
                "yolo_builtin_augmentations_disabled": False,
                "ops": stats.get("ops", {}),
                "cutout_safe": stats.get("cutout_safe", {}),
                "mosaic4": stats.get("mosaic4", {}),
            },
        }
        rows.append(row)
    scoring = score_constraint_candidates(rows)
    operator_impact = build_operator_impact(rows, groups)
    summary = {
        "run_id": RUN_ID,
        "output_dir": str(RUN_ROOT.resolve()),
        "best_under_constraints": scoring.get("best_key"),
        "best_label": scoring.get("best_label"),
        "recall_improved_under_constraints": scoring.get("recall_improved_under_constraints"),
        "constraints_realized": scoring.get("constraints_realized"),
        "failure_driver_if_no_improvement": scoring.get("failure_driver_if_no_improvement"),
    }
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": rel(DATA_YAML),
        "reference": "yolo_default",
        "constraint_limits": {
            "precision_min_delta": PRECISION_DROP_LIMIT,
            "map50_min_delta": MAP50_DROP_LIMIT,
            "map50_95_min_delta": MAP50_95_DROP_LIMIT,
        },
        "rows": rows,
        "constraint_scoring": scoring,
        "operator_impact": operator_impact,
        "summary": summary,
    }


def metric_delta(current: dict[str, Any], reference: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        if current.get(key) is None or reference.get(key) is None:
            out[key] = None
        else:
            out[key] = float(current[key]) - float(reference[key])
    return out


def evaluate_constraints(delta: dict[str, float | None]) -> dict[str, Any]:
    failures = []
    if delta.get("precision") is not None and float(delta["precision"]) < PRECISION_DROP_LIMIT:
        failures.append("precision_drop_gt_0.02")
    if delta.get("map50") is not None and float(delta["map50"]) < MAP50_DROP_LIMIT:
        failures.append("map50_drop_gt_0.01")
    if delta.get("map50_95") is not None and float(delta["map50_95"]) < MAP50_95_DROP_LIMIT:
        failures.append("map50_95_drop_gt_0.01")
    return {"passed": not failures, "failures": failures}


def score_constraint_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_rows = [row for row in rows if row["key"] != "custom_yolo_like_base_old"]
    passed = [row for row in candidate_rows if row["constraints"]["passed"]]
    if not passed:
        return {
            "best_key": None,
            "best_label": None,
            "passed_candidates": [],
            "constraints_realized": False,
            "recall_improved_under_constraints": False,
            "failure_driver_if_no_improvement": infer_failure_driver(rows),
        }
    passed_sorted = sorted(
        passed,
        key=lambda row: (
            float(row["delta_vs_yolo_default"].get("recall") or 0.0),
            float(row["metrics"].get("map50_95") or 0.0),
        ),
        reverse=True,
    )
    best = passed_sorted[0]
    recall_gain = float(best["delta_vs_yolo_default"].get("recall") or 0.0)
    return {
        "best_key": best["key"],
        "best_label": best["label"],
        "passed_candidates": [row["key"] for row in passed_sorted],
        "constraints_realized": True,
        "recall_improved_under_constraints": recall_gain > 0.0 and best["key"] != "yolo_default",
        "failure_driver_if_no_improvement": None if recall_gain > 0.0 and best["key"] != "yolo_default" else infer_failure_driver(rows),
        "selection_rule": "maximize Recall delta under Precision/mAP50/mAP50-95 constraints; tie-break by mAP50-95",
    }


def infer_failure_driver(rows: list[dict[str, Any]]) -> str:
    drivers: list[str] = []
    reference = next((row for row in rows if row["key"] == "yolo_default"), None)
    if not reference:
        return "unknown"
    ref_diag = reference.get("diagnosis", {})
    ref_fp = int(ref_diag.get("global", {}).get("fp", 0) or 0)
    ref_loc = int(ref_diag.get("global", {}).get("localization_weak", 0) or 0)
    for row in rows:
        if row["key"] in {"yolo_default", "custom_yolo_like_base_old"}:
            continue
        delta = row.get("delta_vs_yolo_default", {})
        diag = row.get("diagnosis", {})
        fp_delta = int(diag.get("global", {}).get("fp", 0) or 0) - ref_fp
        loc_delta = int(diag.get("global", {}).get("localization_weak", 0) or 0) - ref_loc
        if delta.get("precision") is not None and float(delta["precision"]) < PRECISION_DROP_LIMIT and fp_delta > 0:
            drivers.append(f"{row['key']}: FP increased by {fp_delta}")
        if delta.get("map50_95") is not None and float(delta["map50_95"]) < MAP50_95_DROP_LIMIT:
            if loc_delta > 0:
                drivers.append(f"{row['key']}: localization weak increased by {loc_delta}")
            else:
                drivers.append(f"{row['key']}: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis")
    return "; ".join(drivers) if drivers else "no constrained recall improvement; metric variance or class tradeoff likely"


def load_group_diagnosis(group: ExperimentGroup) -> dict[str, Any]:
    path = group.output_dir / "diagnosis/diagnosis.json"
    return read_json(path) if path.exists() else {}


def summarize_diagnosis(diagnosis: dict[str, Any]) -> dict[str, Any]:
    if not diagnosis:
        return {}
    per_class = list((diagnosis.get("per_class") or {}).values())
    fn_by_class = sorted(per_class, key=lambda item: int(item.get("fn", 0) or 0), reverse=True)[:8]
    fp_by_class = sorted(per_class, key=lambda item: int(item.get("fp", 0) or 0), reverse=True)[:8]
    return {
        "global": diagnosis.get("global", {}),
        "fn_types": {
            "by_class_top": fn_by_class,
            "by_size": diagnosis.get("size_recall", {}),
            "by_quality": (diagnosis.get("quality", {}) or {}).get("false_negatives", {}),
            "by_position": diagnosis.get("position", {}),
        },
        "fp_types": {
            "by_class_top": fp_by_class,
            "by_quality": (diagnosis.get("quality", {}) or {}).get("false_positives", {}),
            "background": diagnosis.get("fp_background", {}),
        },
        "issues": diagnosis.get("issues", []),
    }


def build_operator_impact(rows: list[dict[str, Any]], groups: list[ExperimentGroup]) -> list[dict[str, Any]]:
    rows_by_key = {row["key"]: row for row in rows}
    output: list[dict[str, Any]] = []
    for group in groups:
        row = rows_by_key[group.key]
        policy = load_group_policy(group)
        stats_ops = row.get("online_aug_stats", {}).get("ops", {})
        operations = policy.get("operations", []) or []
        if not operations and group.key == "yolo_default":
            output.append(
                {
                    "group": group.key,
                    "op": "ultralytics_default_aug_pool",
                    "prob": None,
                    "strength": None,
                    "applied": None,
                    "delta_vs_yolo_default": row["delta_vs_yolo_default"],
                    "impact_scope": "reference_default_pool",
                }
            )
            continue
        for operation in operations:
            op_name = str(operation.get("name", ""))
            op_stats = stats_ops.get(op_name, {})
            output.append(
                {
                    "group": group.key,
                    "op": op_name,
                    "prob": operation.get("prob", operation.get("base_prob")),
                    "strength": operation.get("strength", operation.get("base_strength")),
                    "applied": op_stats.get("applied"),
                    "skipped_probability": op_stats.get("skipped_probability"),
                    "skipped_safety": op_stats.get("skipped_safety"),
                    "delta_vs_yolo_default": row["delta_vs_yolo_default"],
                    "impact_scope": "policy_group_not_isolated",
                }
            )
    return output


def group_artifacts(group: ExperimentGroup) -> dict[str, Any]:
    run_dir = group.output_dir
    return {
        "best_pt": str((run_dir / "train/weights/best.pt").resolve()) if (run_dir / "train/weights/best.pt").exists() else None,
        "last_pt": str((run_dir / "train/weights/last.pt").resolve()) if (run_dir / "train/weights/last.pt").exists() else None,
        "results_csv": str((run_dir / "train/results.csv").resolve()) if (run_dir / "train/results.csv").exists() else None,
        "val_dir": str((run_dir / "val").resolve()) if (run_dir / "val").exists() else None,
        "stats_json": str((run_dir / "reports/online_aug_stats.json").resolve()) if (run_dir / "reports/online_aug_stats.json").exists() else None,
        "diagnosis_json": str((run_dir / "diagnosis/diagnosis.json").resolve()) if (run_dir / "diagnosis/diagnosis.json").exists() else None,
        "diagnosis_report": str((run_dir / "reports/diagnosis_report.md").resolve()) if (run_dir / "reports/diagnosis_report.md").exists() else None,
    }


def write_aggregate_outputs(aggregate: dict[str, Any]) -> None:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_ROOT / "diagnosis_constrained_metrics.json", aggregate)
    write_json(REPORTS_ROOT / "constraint_scoring.json", aggregate["constraint_scoring"])
    write_json(REPORTS_ROOT / "operator_impact.json", aggregate["operator_impact"])
    write_comparison_csv(REPORTS_ROOT / "unified_comparison.csv", aggregate["rows"])
    write_markdown(REPORTS_ROOT / "diagnosis_constrained_experiment_report.md", build_aggregate_markdown(aggregate))
    write_markdown(REPORTS_ROOT / "constraint_scoring.md", build_constraint_markdown(aggregate))
    write_markdown(REPORTS_ROOT / "operator_impact.md", build_operator_impact_markdown(aggregate))


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "label",
                "precision",
                "recall",
                "map50",
                "map50_95",
                "d_precision",
                "d_recall",
                "d_map50",
                "d_map50_95",
                "constraints_passed",
            ],
        )
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            delta = row["delta_vs_yolo_default"]
            writer.writerow(
                {
                    "key": row["key"],
                    "label": row["label"],
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "map50": metrics.get("map50"),
                    "map50_95": metrics.get("map50_95"),
                    "d_precision": delta.get("precision"),
                    "d_recall": delta.get("recall"),
                    "d_map50": delta.get("map50"),
                    "d_map50_95": delta.get("map50_95"),
                    "constraints_passed": row["constraints"]["passed"],
                }
            )


def build_group_diagnosis_report(group: ExperimentGroup, metrics: dict[str, Any], diagnosis: dict[str, Any]) -> str:
    summary = summarize_diagnosis(diagnosis)
    global_diag = summary.get("global", {})
    lines = [
        f"# Diagnosis Report: {group.label}",
        "",
        f"- Run dir: `{group.output_dir}`",
        f"- Precision/Recall/mAP50/mAP50-95: `{fmt(metrics.get('precision'))}/{fmt(metrics.get('recall'))}/{fmt(metrics.get('map50'))}/{fmt(metrics.get('map50_95'))}`",
        f"- TP/FP/FN: `{global_diag.get('tp', 0)}/{global_diag.get('fp', 0)}/{global_diag.get('fn', 0)}`",
        f"- Localization weak: `{global_diag.get('localization_weak', 0)}`",
        "",
        "## FN Types",
        "",
        diagnosis_class_table(summary.get("fn_types", {}).get("by_class_top", []), count_key="fn"),
        "",
        "## FP Types",
        "",
        diagnosis_class_table(summary.get("fp_types", {}).get("by_class_top", []), count_key="fp"),
        "",
        "## Quality And Position",
        "",
        f"- FN quality: `{json.dumps(summary.get('fn_types', {}).get('by_quality', {}), ensure_ascii=False)}`",
        f"- FP quality: `{json.dumps(summary.get('fp_types', {}).get('by_quality', {}), ensure_ascii=False)}`",
        f"- FN size: `{json.dumps(summary.get('fn_types', {}).get('by_size', {}), ensure_ascii=False)}`",
        f"- FN position: `{json.dumps(summary.get('fn_types', {}).get('by_position', {}), ensure_ascii=False)}`",
    ]
    return "\n".join(lines) + "\n"


def build_aggregate_markdown(aggregate: dict[str, Any]) -> str:
    rows = aggregate["rows"]
    scoring = aggregate["constraint_scoring"]
    lines = [
        "# YOLO Default Diagnosis-Constrained Experiment",
        "",
        f"- Run ID: `{RUN_ID}`",
        "- Base policy: Ultralytics YOLO default augmentation remains enabled for A-D.",
        "- Custom diagnosis policies are added online in the dataloader; no fixed augmented dataset is generated.",
        f"- Constraint reference: `YOLO default`",
        f"- Constraint: Precision delta >= `{PRECISION_DROP_LIMIT:+.2f}`, mAP50 delta >= `{MAP50_DROP_LIMIT:+.2f}`, mAP50-95 delta >= `{MAP50_95_DROP_LIMIT:+.2f}`.",
        "",
        "## Unified Metrics",
        "",
        comparison_table(rows),
        "",
        "## Constraint Selection",
        "",
        f"- Best under constraints: `{scoring.get('best_label')}`",
        f"- Recall improved under constraints: `{str(scoring.get('recall_improved_under_constraints')).lower()}`",
        f"- Passed candidates: `{', '.join(scoring.get('passed_candidates', []))}`",
        f"- Failure driver if no improvement: `{scoring.get('failure_driver_if_no_improvement')}`",
        "",
        "## Diagnostics",
        "",
        "| run | TP | FP | FN | localization_weak | top FN class | top FP class |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        diag = row.get("diagnosis", {})
        global_diag = diag.get("global", {})
        top_fn = first_class(diag.get("fn_types", {}).get("by_class_top", []), "fn")
        top_fp = first_class(diag.get("fp_types", {}).get("by_class_top", []), "fp")
        lines.append(
            f"| {row['label']} | {global_diag.get('tp', 0)} | {global_diag.get('fp', 0)} | "
            f"{global_diag.get('fn', 0)} | {global_diag.get('localization_weak', 0)} | {top_fn} | {top_fp} |"
        )
    lines.extend(
        [
            "",
            "## Operator Impact",
            "",
            "Operator impact is reported at policy-group scope, not as isolated single-op ablation.",
            "",
            operator_impact_table(aggregate["operator_impact"]),
            "",
            "## Artifacts",
            "",
            f"- Metrics JSON: `{REPORTS_ROOT / 'diagnosis_constrained_metrics.json'}`",
            f"- Constraint scoring JSON: `{REPORTS_ROOT / 'constraint_scoring.json'}`",
            f"- Operator impact JSON: `{REPORTS_ROOT / 'operator_impact.json'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_constraint_markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# Constraint Scoring",
        "",
        "- Reference: YOLO default.",
        "- Reject if Precision drops by more than 0.02.",
        "- Reject if mAP50 drops by more than 0.01.",
        "- Reject if mAP50-95 drops by more than 0.01.",
        "- Among passing policies, maximize Recall delta; tie-break by mAP50-95.",
        "",
        comparison_table(aggregate["rows"]),
        "",
        "## Result",
        "",
        f"- Best: `{aggregate['constraint_scoring'].get('best_label')}`",
        f"- Recall improved under constraints: `{str(aggregate['constraint_scoring'].get('recall_improved_under_constraints')).lower()}`",
        f"- Failure driver if no improvement: `{aggregate['constraint_scoring'].get('failure_driver_if_no_improvement')}`",
    ]
    return "\n".join(lines) + "\n"


def build_operator_impact_markdown(aggregate: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Operator Impact",
            "",
            "Impact is measured at policy-group scope because this experiment does not run single-operator ablations.",
            "",
            operator_impact_table(aggregate["operator_impact"]),
        ]
    ) + "\n"


def comparison_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run | Precision | Recall | mAP50 | mAP50-95 | dP | dR | d_mAP50 | d_mAP50-95 | pass constraints |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row["metrics"]
        delta = row["delta_vs_yolo_default"]
        lines.append(
            f"| {row['label']} | {fmt(metrics.get('precision'))} | {fmt(metrics.get('recall'))} | "
            f"{fmt(metrics.get('map50'))} | {fmt(metrics.get('map50_95'))} | "
            f"{fmt(delta.get('precision'), signed=True)} | {fmt(delta.get('recall'), signed=True)} | "
            f"{fmt(delta.get('map50'), signed=True)} | {fmt(delta.get('map50_95'), signed=True)} | "
            f"{'yes' if row['constraints']['passed'] else 'no'} |"
        )
    return "\n".join(lines)


def operator_impact_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| group | op | prob | strength | applied | dP | dR | d_mAP50 | d_mAP50-95 | scope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        delta = row["delta_vs_yolo_default"]
        lines.append(
            f"| {row['group']} | {row['op']} | {fmt(row.get('prob'))} | {fmt(row.get('strength'))} | "
            f"{row.get('applied', 'n/a')} | {fmt(delta.get('precision'), signed=True)} | "
            f"{fmt(delta.get('recall'), signed=True)} | {fmt(delta.get('map50'), signed=True)} | "
            f"{fmt(delta.get('map50_95'), signed=True)} | {row.get('impact_scope')} |"
        )
    return "\n".join(lines)


def diagnosis_class_table(rows: list[dict[str, Any]], *, count_key: str) -> str:
    lines = [
        "| class_id | class_name | gt | tp | fp | fn | precision | recall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    filtered = [row for row in rows if int(row.get(count_key, 0) or 0) > 0]
    for row in filtered[:8]:
        lines.append(
            f"| {row.get('class_id')} | {row.get('class_name')} | {row.get('gt')} | {row.get('tp')} | "
            f"{row.get('fp')} | {row.get('fn')} | {fmt(row.get('precision'))} | {fmt(row.get('recall'))} |"
        )
    if len(lines) == 2:
        lines.append("| n/a | n/a | 0 | 0 | 0 | 0 | n/a | n/a |")
    return "\n".join(lines)


def first_class(rows: list[dict[str, Any]], key: str) -> str:
    for row in rows:
        if int(row.get(key, 0) or 0) > 0:
            return f"{row.get('class_name')} ({key}={row.get(key)})"
    return "n/a"


def update_state_docs(aggregate: dict[str, Any]) -> None:
    scoring = aggregate["constraint_scoring"]
    best = scoring.get("best_label")
    rows_by_key = {row["key"]: row for row in aggregate["rows"]}
    yolo = rows_by_key["yolo_default"]["metrics"]
    section = "\n".join(
        [
            "## YOLO Default Diagnosis-Constrained 50 Epoch",
            "",
            f"- Run ID: `{RUN_ID}`",
            "- Objective: keep Ultralytics YOLO default augmentation enabled and add only constrained diagnostic online augmentation.",
            "- Groups: YOLO default, diagnosis_light, diagnosis_precision_safe, diagnosis_recall_safe, plus old custom_yolo_like_base control.",
            "- All A-D groups use `yolo11n.pt`, tiled safe no-OK/no-position data, `epochs=50`, `imgsz=1024`, `batch=2`, `workers=0`, `device=0`, `seed=42`.",
            "- YOLO built-in augmentation: enabled for A-D; custom diagnosis policies are added online in the dataloader.",
            "- Fixed augmented dataset generated: `false`.",
            f"- YOLO default reference P/R/mAP50/mAP50-95: `{fmt(yolo.get('precision'))}/{fmt(yolo.get('recall'))}/{fmt(yolo.get('map50'))}/{fmt(yolo.get('map50_95'))}`",
            f"- Best under industrial constraints: `{best}`",
            f"- Recall improved while constraints hold: `{str(scoring.get('recall_improved_under_constraints')).lower()}`",
            f"- Failure driver if no improvement: `{scoring.get('failure_driver_if_no_improvement')}`",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/diagnosis_constrained_experiment_report.md`",
            f"- Metrics JSON: `outputs/experiments/{RUN_ID}/reports/diagnosis_constrained_metrics.json`",
            f"- Constraint scoring: `outputs/experiments/{RUN_ID}/reports/constraint_scoring.json`",
        ]
    )
    for rel_path in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
        upsert_section(PROJECT_ROOT / rel_path, "YOLO_DEFAULT_DIAGNOSIS_CONSTRAINED_50EP", section)


def upsert_section(path: Path, marker: str, section: str) -> None:
    start = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    block = f"{start}\n{section.rstrip()}\n{end}\n"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        path.write_text(before.rstrip() + "\n\n" + block + after.lstrip(), encoding="utf-8")
    else:
        path.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")


def export_project_snapshot() -> None:
    script = PROJECT_ROOT / "scripts/export_project_snapshot.py"
    if script.exists():
        subprocess.run([str(PYTHON_EXE), str(script)], cwd=str(PROJECT_ROOT), check=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    write_text(path, text)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def group_to_dict(group: ExperimentGroup) -> dict[str, Any]:
    return {
        "key": group.key,
        "label": group.label,
        "policy_path": rel(group.policy_path),
        "run_id": group.run_id,
        "train": group.train,
        "run_dir": str(group.output_dir),
    }


def fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if signed:
        return f"{number:+.4f}"
    return f"{number:.4f}"


if __name__ == "__main__":
    main()
