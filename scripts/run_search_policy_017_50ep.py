from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402
from AutoAugment.utils import find_yolo_records_from_dirs  # noqa: E402
from scripts.run_diagnosis_guided_policy_search import (  # noqa: E402
    build_trial_dataset,
    compute_scores,
    count_label_rows,
    format_operations,
    has_oom,
    parse_yolo_val_log,
    read_training_summary,
    write_dataset_report_md,
)


RUN_ID = "20260518_tiled1024_safe_no_ok_position_search_policy_017_yolo11n_50ep"
BASELINE_ID = "20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep"
DIAGAUG_ID = "20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep"
YOLO_DEFAULT_ID = "20260518_tiled1024_safe_no_ok_position_yolo_default_aug_yolo11n_50ep"
RANDOM_ID = "20260518_tiled1024_safe_no_ok_position_random_external_aug_yolo11n_50ep"
POLICY_SEARCH_ID = "20260518_tiled1024_safe_no_ok_position_diagnosis_guided_policy_search"

DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
DATA_YAML = DATASET_ROOT / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID
POLICY_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / POLICY_SEARCH_ID
    / "short_training"
    / "policies"
    / "search_policy_017"
    / "policy.json"
)
BEST_POLICY_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / POLICY_SEARCH_ID
    / "reports"
    / "best_policy_summary.md"
)

BASELINE_METRICS = PROJECT_ROOT / "outputs" / "experiments" / BASELINE_ID / "reports" / "baseline_50ep_metrics.json"
DIAGAUG_METRICS = PROJECT_ROOT / "outputs" / "experiments" / DIAGAUG_ID / "reports" / "diagaug_50ep_metrics.json"
YOLO_DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "experiments" / YOLO_DEFAULT_ID / "reports" / "yolo_default_aug_50ep_metrics.json"
RANDOM_METRICS = PROJECT_ROOT / "outputs" / "experiments" / RANDOM_ID / "reports" / "random_external_aug_50ep_metrics.json"

YOLO = Path("D:/Anaconda/envs/pytorch/Scripts/yolo.exe")
MODEL = "yolo11n.pt"
EPOCHS = 50
IMGSZ = 1024
DEFAULT_BATCH = 2
WORKERS = 0
DEVICE = "0"
SEED = 42
DISABLED_YOLO_AUGS = {
    "mosaic": 0,
    "mixup": 0,
    "copy_paste": 0,
    "hsv_h": 0,
    "hsv_s": 0,
    "hsv_v": 0,
    "degrees": 0,
    "translate": 0,
    "scale": 0,
    "shear": 0,
    "perspective": 0,
    "fliplr": 0,
    "flipud": 0,
}


def main() -> None:
    args = parse_args()
    configure_environment()
    prepare_output_dir(OUTPUT_DIR, force=args.force, resume=args.resume)

    policy = read_json(POLICY_JSON)
    baseline = read_json(BASELINE_METRICS)
    diagaug = read_json(DIAGAUG_METRICS)
    yolo_default = read_json(YOLO_DEFAULT_METRICS)
    random_external = read_json(RANDOM_METRICS)
    class_names = load_class_names_from_data_yaml(DATA_YAML)

    policy_dir = OUTPUT_DIR / "policies"
    policy_dir.mkdir(parents=True, exist_ok=True)
    write_json(policy_dir / "search_policy_017.json", policy)
    if BEST_POLICY_SUMMARY.exists():
        (policy_dir / "best_policy_summary_source.md").write_text(BEST_POLICY_SUMMARY.read_text(encoding="utf-8-sig"), encoding="utf-8")

    dataset_report = ensure_dataset(policy=policy, class_names=class_names, resume=args.resume)
    write_json(OUTPUT_DIR / "dataset" / "augmented_dataset_summary.json", dataset_report)
    write_dataset_report_md(OUTPUT_DIR / "dataset" / "augmented_dataset_summary.md", dataset_report)

    batch = DEFAULT_BATCH
    oom = False
    if args.skip_training:
        train_result = read_existing_command_result("final_train", batch=batch)
    else:
        print("[search017] training 50 epochs")
        train_result = run_train(dataset_report["data_yaml"], batch=batch)
        if train_result["returncode"] != 0 and has_oom(
            OUTPUT_DIR / "logs" / "final_train_batch2.stdout.log",
            OUTPUT_DIR / "logs" / "final_train_batch2.stderr.log",
        ):
            oom = True
            batch = 1
            remove_train_dir()
            print("[search017] OOM at batch=2; retrying batch=1")
            train_result = run_train(dataset_report["data_yaml"], batch=batch)
        if train_result["returncode"] != 0:
            raise RuntimeError(f"search_policy_017 final training failed; see {OUTPUT_DIR / 'logs'}")

    print("[search017] validating best.pt on base validation set")
    val_result = run_val(str(DATA_YAML.resolve()), batch=batch)
    if val_result["returncode"] != 0 and has_oom(
        OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stdout.log",
        OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stderr.log",
    ):
        oom = True
        batch = 1
        val_result = run_val(str(DATA_YAML.resolve()), batch=batch)
    if val_result["returncode"] != 0:
        raise RuntimeError(f"search_policy_017 validation failed; see {OUTPUT_DIR / 'logs'}")

    oom = oom or has_oom(
        OUTPUT_DIR / "logs" / f"final_train_batch{batch}.stdout.log",
        OUTPUT_DIR / "logs" / f"final_train_batch{batch}.stderr.log",
        OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stdout.log",
        OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stderr.log",
    )
    val_metrics = parse_yolo_val_log(OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stdout.log", class_names)
    train_summary = read_training_summary(OUTPUT_DIR / "train" / "results.csv")
    payload = build_metrics_payload(
        policy=policy,
        dataset_report=dataset_report,
        val_metrics=val_metrics,
        train_result=train_result,
        val_result=val_result,
        train_summary=train_summary,
        baseline=baseline,
        diagaug=diagaug,
        yolo_default=yolo_default,
        random_external=random_external,
        batch=batch,
        oom=oom,
        class_names=class_names,
    )
    write_json(OUTPUT_DIR / "reports" / "search_policy_017_50ep_metrics.json", payload)
    write_search017_report(OUTPUT_DIR / "reports" / "search_policy_017_50ep_report.md", payload)
    write_comparison_report(OUTPUT_DIR / "reports" / "compare_baseline_diagaug_random_yolo_search017.md", payload)
    write_final_training_artifacts(payload)
    update_state_docs(payload)
    export_project_snapshot()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal 50 epoch YOLO training for search_policy_017.")
    parser.add_argument("--force", action="store_true", help="Remove the existing search017 output directory before running.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing built dataset if present.")
    parser.add_argument("--skip-training", action="store_true", help="Reuse existing final training logs and run validation/reporting only.")
    return parser.parse_args()


def configure_environment() -> None:
    if YOLO.parent.exists():
        os.environ["PATH"] = str(YOLO.parent) + os.pathsep + os.environ.get("PATH", "")
    yolo_config = PROJECT_ROOT / "outputs" / "Ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"


def prepare_output_dir(path: Path, *, force: bool, resume: bool) -> None:
    resolved = path.resolve()
    experiments_root = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    if resolved.exists() and force:
        if not str(resolved).startswith(str(experiments_root)):
            raise RuntimeError(f"refusing to remove output outside experiments: {resolved}")
        shutil.rmtree(resolved)
    elif resolved.exists() and not resume:
        raise FileExistsError(f"output directory already exists; rerun with --force or --resume: {resolved}")
    for subdir in ["dataset", "final_training", "logs", "metrics", "policies", "reports"]:
        (resolved / subdir).mkdir(parents=True, exist_ok=True)


def ensure_dataset(*, policy: dict[str, Any], class_names: dict[int, str], resume: bool) -> dict[str, Any]:
    report_path = OUTPUT_DIR / "dataset" / "augmented_dataset_summary.json"
    if resume and report_path.exists():
        return read_json(report_path)
    train_records = find_yolo_records_from_dirs(DATASET_ROOT / "images" / "train", DATASET_ROOT / "labels" / "train", missing_label="empty")
    val_records = find_yolo_records_from_dirs(DATASET_ROOT / "images" / "val", DATASET_ROOT / "labels" / "val", missing_label="empty")
    print("[search017] building original + 1x search_policy_017 augmented dataset")
    _paths, report = build_trial_dataset(policy, OUTPUT_DIR, train_records, val_records, class_names)
    report["source_data_yaml"] = rel(DATA_YAML)
    report["policy_json"] = rel(POLICY_JSON)
    report["best_policy_summary"] = rel(BEST_POLICY_SUMMARY)
    report["train_bboxes"] = count_label_rows(Path(report["labels_train"]))
    report["val_bboxes"] = count_label_rows(Path(report["labels_val"]))
    return report


def run_train(data_yaml: str, *, batch: int) -> dict[str, Any]:
    command = [
        str(YOLO),
        "detect",
        "train",
        f"model={MODEL}",
        f"data={data_yaml}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"batch={batch}",
        f"workers={WORKERS}",
        f"device={DEVICE}",
        f"seed={SEED}",
        f"project={OUTPUT_DIR}",
        "name=train",
        "exist_ok=True",
        *[f"{key}={value}" for key, value in DISABLED_YOLO_AUGS.items()],
    ]
    return run_command(command, "final_train", batch)


def run_val(data_yaml: str, *, batch: int) -> dict[str, Any]:
    command = [
        str(YOLO),
        "detect",
        "val",
        f"model={OUTPUT_DIR / 'train' / 'weights' / 'best.pt'}",
        f"data={data_yaml}",
        f"imgsz={IMGSZ}",
        f"batch={batch}",
        f"workers={WORKERS}",
        f"device={DEVICE}",
        f"project={OUTPUT_DIR}",
        "name=val",
        "exist_ok=True",
    ]
    return run_command(command, "final_val", batch)


def run_command(command: list[str], prefix: str, batch: int) -> dict[str, Any]:
    logs_dir = OUTPUT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_text = subprocess.list2cmdline(command)
    stem = f"{prefix}_batch{batch}"
    (logs_dir / f"{stem}.command.txt").write_text(command_text + "\n", encoding="utf-8")
    stdout_path = logs_dir / f"{stem}.stdout.log"
    stderr_path = logs_dir / f"{stem}.stderr.log"
    start = time.time()
    start_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (logs_dir / f"{stem}.start_utc.txt").write_text(start_iso + "\n", encoding="utf-8")
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as stderr:
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, text=True, env=os.environ.copy())
        returncode = process.wait()
    end_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    wall = time.time() - start
    (logs_dir / f"{stem}.end_utc.txt").write_text(end_iso + "\n", encoding="utf-8")
    (logs_dir / f"{stem}.exitcode").write_text(str(returncode) + "\n", encoding="utf-8")
    return {
        "command": command_text,
        "returncode": returncode,
        "wall_seconds": wall,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "start_utc": start_iso,
        "end_utc": end_iso,
    }


def read_existing_command_result(prefix: str, *, batch: int) -> dict[str, Any]:
    logs_dir = OUTPUT_DIR / "logs"
    stem = f"{prefix}_batch{batch}"
    return {
        "command": read_text(logs_dir / f"{stem}.command.txt").strip(),
        "returncode": int(read_text(logs_dir / f"{stem}.exitcode").strip()),
        "wall_seconds": 0.0,
        "stdout": str((logs_dir / f"{stem}.stdout.log").resolve()),
        "stderr": str((logs_dir / f"{stem}.stderr.log").resolve()),
        "start_utc": read_text(logs_dir / f"{stem}.start_utc.txt").strip(),
        "end_utc": read_text(logs_dir / f"{stem}.end_utc.txt").strip(),
    }


def remove_train_dir() -> None:
    train_dir = OUTPUT_DIR / "train"
    if train_dir.exists():
        resolved = train_dir.resolve()
        if not str(resolved).startswith(str(OUTPUT_DIR.resolve())):
            raise RuntimeError(f"refusing to remove unexpected train dir: {resolved}")
        shutil.rmtree(train_dir)


def build_metrics_payload(
    *,
    policy: dict[str, Any],
    dataset_report: dict[str, Any],
    val_metrics: dict[str, Any],
    train_result: dict[str, Any],
    val_result: dict[str, Any],
    train_summary: dict[str, Any],
    baseline: dict[str, Any],
    diagaug: dict[str, Any],
    yolo_default: dict[str, Any],
    random_external: dict[str, Any],
    batch: int,
    oom: bool,
    class_names: dict[int, str],
) -> dict[str, Any]:
    final_overall = {**val_metrics["overall"], **compute_scores(val_metrics["overall"])}
    final_per_class = val_metrics["per_class"]
    baseline_overall = baseline["validation"]["overall"]
    baseline_per_class = baseline["validation"]["per_class"]
    diagaug_overall = diagaug["final_metrics"]
    diagaug_per_class = diagaug["final_per_class"]
    yolo_default_overall = yolo_default["final_metrics"]
    yolo_default_per_class = yolo_default["final_per_class"]
    random_overall = random_external["final_metrics"]
    random_per_class = random_external["final_per_class"]

    comparisons = {
        "vs_baseline": build_comparison(baseline_overall, baseline_per_class, final_overall, final_per_class, "baseline", "search_policy_017"),
        "vs_diagaug": build_comparison(diagaug_overall, diagaug_per_class, final_overall, final_per_class, "diagaug", "search_policy_017"),
        "vs_yolo_default": build_comparison(
            yolo_default_overall,
            yolo_default_per_class,
            final_overall,
            final_per_class,
            "yolo_default",
            "search_policy_017",
        ),
        "vs_random_external": build_comparison(
            random_overall,
            random_per_class,
            final_overall,
            final_per_class,
            "random_external",
            "search_policy_017",
        ),
        "five_way_per_class": build_five_way_per_class(
            baseline_per_class,
            yolo_default_per_class,
            diagaug_per_class,
            random_per_class,
            final_per_class,
            class_names,
        ),
    }
    formal_rank = build_formal_rank(
        {
            "baseline": baseline_overall,
            "diag_policy_001": diagaug_overall,
            "yolo_default": yolo_default_overall,
            "random_external": random_overall,
            "search_policy_017": final_overall,
        }
    )
    best_pt = OUTPUT_DIR / "train" / "weights" / "best.pt"
    summary = {
        "precision": final_overall["precision"],
        "recall": final_overall["recall"],
        "map50": final_overall["map50"],
        "map50_95": final_overall["map50_95"],
        "balanced_score": final_overall["balanced_score"],
        "delta_vs_baseline": comparisons["vs_baseline"]["overall_delta"],
        "delta_vs_diagaug": comparisons["vs_diagaug"]["overall_delta"],
        "delta_vs_random_external": comparisons["vs_random_external"]["overall_delta"],
        "delta_vs_yolo_default": comparisons["vs_yolo_default"]["overall_delta"],
        "beats_random_external_by_map50_95": final_overall["map50_95"] > random_overall["map50_95"],
        "beats_diag_policy_001_by_map50_95": final_overall["map50_95"] > diagaug_overall["map50_95"],
        "is_best_formal_by_map50_95": formal_rank["best_by_map50_95"] == "search_policy_017",
        "is_best_formal_by_balanced_score": formal_rank["best_by_balanced_score"] == "search_policy_017",
        "best_pt": str(best_pt.resolve()),
        "oom": oom,
        "training_wall_seconds": train_result["wall_seconds"],
    }
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": rel(DATA_YAML),
        "model": MODEL,
        "policy": policy,
        "policy_source_json": rel(POLICY_JSON),
        "best_policy_summary": rel(BEST_POLICY_SUMMARY),
        "augmented_dataset": dataset_report,
        "train_settings": {
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": batch,
            "requested_batch": DEFAULT_BATCH,
            "workers": WORKERS,
            "device": DEVICE,
            "seed": SEED,
            "yolo_builtin_augmentations_disabled": True,
            "disabled_augmentation_params": DISABLED_YOLO_AUGS,
        },
        "commands": {
            "final_train_command": train_result["command"],
            "final_val_command": val_result["command"],
            "final_train_exitcode": train_result["returncode"],
            "final_val_exitcode": val_result["returncode"],
        },
        "time": {
            "train_start_utc": train_result["start_utc"],
            "train_end_utc": train_result["end_utc"],
            "val_start_utc": val_result["start_utc"],
            "val_end_utc": val_result["end_utc"],
            "training_wall_seconds": train_result["wall_seconds"],
            "training_wall_hours": train_result["wall_seconds"] / 3600.0,
            "validation_wall_seconds": val_result["wall_seconds"],
            "training_loop_time_seconds": train_summary.get("last_epoch", {}).get("time_seconds"),
        },
        "artifacts": {
            "best_pt": str(best_pt.resolve()),
            "last_pt": str((OUTPUT_DIR / "train" / "weights" / "last.pt").resolve()),
            "policy_json": rel(OUTPUT_DIR / "policies" / "search_policy_017.json"),
            "augmented_dataset_report": rel(OUTPUT_DIR / "dataset" / "augmented_dataset_summary.json"),
            "train_log": rel(OUTPUT_DIR / "logs" / f"final_train_batch{batch}.stdout.log"),
            "val_log": rel(OUTPUT_DIR / "logs" / f"final_val_batch{batch}.stdout.log"),
            "reports_dir": rel(OUTPUT_DIR / "reports"),
        },
        "final_metrics": final_overall,
        "final_per_class": final_per_class,
        "baseline_metrics": baseline_overall,
        "baseline_per_class": baseline_per_class,
        "diagaug_metrics": diagaug_overall,
        "diagaug_per_class": diagaug_per_class,
        "yolo_default_metrics": yolo_default_overall,
        "yolo_default_per_class": yolo_default_per_class,
        "random_external_metrics": random_overall,
        "random_external_per_class": random_per_class,
        "comparisons": comparisons,
        "formal_rank": formal_rank,
        "training_results": train_summary,
        "oom": oom,
        "summary": summary,
        "notes": [
            "search_policy_017 is diagnosis-guided and was selected by balanced score in the policy search run.",
            "The augmented training set keeps all original train images and adds one search_policy_017 augmented copy per train image.",
            "YOLO built-in augmentations were disabled for final training.",
            "Final validation uses the original safe tiled no-OK/no-position validation set.",
            "Best-formal flags are reported by mAP50-95 and balanced score to avoid hiding metric tradeoffs.",
        ],
    }


def build_comparison(
    reference_overall: dict[str, Any],
    reference_per_class: list[dict[str, Any]],
    target_overall: dict[str, Any],
    target_per_class: list[dict[str, Any]],
    reference_name: str,
    target_name: str,
) -> dict[str, Any]:
    ref_by_id = {int(row["class_id"]): row for row in reference_per_class}
    per_class = []
    for row in target_per_class:
        class_id = int(row["class_id"])
        ref = ref_by_id[class_id]
        per_class.append(
            {
                "class_id": class_id,
                "name": row["name"],
                f"{reference_name}_recall": float(ref["recall"]),
                f"{target_name}_recall": float(row["recall"]),
                "delta_recall": float(row["recall"]) - float(ref["recall"]),
                f"{reference_name}_ap50": float(ref["ap50"]),
                f"{target_name}_ap50": float(row["ap50"]),
                "delta_ap50": float(row["ap50"]) - float(ref["ap50"]),
            }
        )
    return {
        "overall_delta": {
            "precision": float(target_overall["precision"]) - float(reference_overall["precision"]),
            "recall": float(target_overall["recall"]) - float(reference_overall["recall"]),
            "map50": float(target_overall["map50"]) - float(reference_overall["map50"]),
            "map50_95": float(target_overall["map50_95"]) - float(reference_overall["map50_95"]),
            "balanced_score": balanced_score(target_overall) - balanced_score(reference_overall),
        },
        "per_class_delta": per_class,
    }


def build_five_way_per_class(
    baseline: list[dict[str, Any]],
    yolo_default: list[dict[str, Any]],
    diagaug: list[dict[str, Any]],
    random_external: list[dict[str, Any]],
    search017: list[dict[str, Any]],
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    by_name = {
        "baseline": {int(row["class_id"]): row for row in baseline},
        "yolo_default": {int(row["class_id"]): row for row in yolo_default},
        "diagaug": {int(row["class_id"]): row for row in diagaug},
        "random_external": {int(row["class_id"]): row for row in random_external},
        "search017": {int(row["class_id"]): row for row in search017},
    }
    rows = []
    for class_id, name in class_names.items():
        rows.append(
            {
                "class_id": class_id,
                "name": name,
                "baseline_recall": by_name["baseline"][class_id]["recall"],
                "baseline_ap50": by_name["baseline"][class_id]["ap50"],
                "yolo_default_recall": by_name["yolo_default"][class_id]["recall"],
                "yolo_default_ap50": by_name["yolo_default"][class_id]["ap50"],
                "diagaug_recall": by_name["diagaug"][class_id]["recall"],
                "diagaug_ap50": by_name["diagaug"][class_id]["ap50"],
                "random_recall": by_name["random_external"][class_id]["recall"],
                "random_ap50": by_name["random_external"][class_id]["ap50"],
                "search017_recall": by_name["search017"][class_id]["recall"],
                "search017_ap50": by_name["search017"][class_id]["ap50"],
            }
        )
    return rows


def build_formal_rank(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored = {
        name: {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "map50": float(metrics["map50"]),
            "map50_95": float(metrics["map50_95"]),
            "balanced_score": balanced_score(metrics),
        }
        for name, metrics in rows.items()
    }
    return {
        "runs": scored,
        "best_by_map50_95": max(scored, key=lambda name: scored[name]["map50_95"]),
        "best_by_balanced_score": max(scored, key=lambda name: scored[name]["balanced_score"]),
    }


def write_search017_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["final_metrics"]
    dataset = payload["augmented_dataset"]
    summary = payload["summary"]
    lines = [
        "# search_policy_017 50 Epoch Report",
        "",
        "- Scope: formal 50 epoch YOLO training for diagnosis-guided `search_policy_017`.",
        f"- Dataset: `{payload['dataset']}`",
        f"- Augmented dataset: `{dataset['dataset_dir']}`",
        f"- Training set: original train images `{dataset['original_train_count']}` + augmented train images `{dataset['augmented_train_count']}`.",
        f"- Train images / bboxes: `{dataset['train_images']}` / `{dataset['train_bboxes']}`",
        f"- Val images / bboxes: `{dataset['val_images']}` / `{dataset['val_bboxes']}`",
        f"- Policy JSON: `{payload['policy_source_json']}`",
        f"- Operations: `{format_operations(payload['policy']['operations'])}`",
        f"- bbox_valid_rate: `{dataset['bbox_valid_rate']:.6f}`",
        f"- image_failures / label_failures: `{dataset['image_failures']}` / `{dataset['label_failures']}`",
        "- Train settings: `model=yolo11n.pt epochs=50 imgsz=1024 batch=2 workers=0 device=0 seed=42`",
        "- YOLO built-in augmentations disabled: `"
        + " ".join(f"{key}={value}" for key, value in DISABLED_YOLO_AUGS.items())
        + "`",
        f"- Precision: `{metrics['precision']:.3f}`",
        f"- Recall: `{metrics['recall']:.3f}`",
        f"- mAP50: `{metrics['map50']:.3f}`",
        f"- mAP50-95: `{metrics['map50_95']:.3f}`",
        f"- Balanced score: `{metrics['balanced_score']:.6f}`",
        f"- Exceeds random external by mAP50-95: `{str(summary['beats_random_external_by_map50_95']).lower()}`",
        f"- Exceeds diag_policy_001 by mAP50-95: `{str(summary['beats_diag_policy_001_by_map50_95']).lower()}`",
        f"- Current best formal by mAP50-95: `{str(summary['is_best_formal_by_map50_95']).lower()}`",
        f"- Current best formal by balanced score: `{str(summary['is_best_formal_by_balanced_score']).lower()}`",
        f"- best.pt: `{payload['artifacts']['best_pt']}`",
        f"- OOM: `{str(payload['oom']).lower()}`",
        f"- Training wall seconds: `{payload['time']['training_wall_seconds']:.1f}`",
        f"- Training wall hours: `{payload['time']['training_wall_hours']:.3f}`",
        "",
        "## Operation Details",
        "",
        "| operation | prob | strength | params |",
        "|---|---:|---:|---|",
    ]
    for op in payload["policy"]["operations"]:
        lines.append(f"| {op['name']} | {float(op['prob']):.4f} | {float(op['strength']):.4f} | `{json.dumps(op.get('params', {}), ensure_ascii=False)}` |")
    lines.extend(
        [
            "",
            "## Per-Class Recall/AP50",
            "",
            "| class id | class | Recall | AP50 |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in payload["final_per_class"]:
        lines.append(f"| {row['class_id']} | {row['name']} | {row['recall']:.3f} | {row['ap50']:.3f} |")
    lines.extend(
        [
            "",
            "## Overall Deltas",
            "",
            "| reference | delta Precision | delta Recall | delta mAP50 | delta mAP50-95 | delta balanced |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("vs_baseline", "baseline"),
        ("vs_diagaug", "diag_policy_001 DiagAug 50ep"),
        ("vs_yolo_default", "YOLO default"),
        ("vs_random_external", "random external"),
    ]:
        delta = payload["comparisons"][key]["overall_delta"]
        lines.append(
            f"| {label} | {delta['precision']:+.3f} | {delta['recall']:+.3f} | "
            f"{delta['map50']:+.3f} | {delta['map50_95']:+.3f} | {delta['balanced_score']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "Training:",
            "",
            "```powershell",
            payload["commands"]["final_train_command"],
            "```",
            "",
            "Validation:",
            "",
            "```powershell",
            payload["commands"]["final_val_command"],
            "```",
        ]
    )
    write_markdown(path, lines)


def write_comparison_report(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        ("No-YOLO-Aug baseline", payload["baseline_metrics"]),
        ("diag_policy_001 DiagAug", payload["diagaug_metrics"]),
        ("YOLO default aug", payload["yolo_default_metrics"]),
        ("Random external aug", payload["random_external_metrics"]),
        ("search_policy_017", payload["final_metrics"]),
    ]
    lines = [
        "# Baseline vs DiagAug vs Random External vs YOLO Default vs search_policy_017",
        "",
        "| run | Precision | Recall | mAP50 | mAP50-95 | balanced |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows:
        lines.append(
            f"| {name} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | "
            f"{float(row['map50']):.3f} | {float(row['map50_95']):.3f} | {balanced_score(row):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Overall Delta For search_policy_017",
            "",
            "| reference | delta Precision | delta Recall | delta mAP50 | delta mAP50-95 | delta balanced |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in [
        ("vs_baseline", "baseline"),
        ("vs_diagaug", "diag_policy_001 DiagAug"),
        ("vs_yolo_default", "YOLO default"),
        ("vs_random_external", "random external"),
    ]:
        delta = payload["comparisons"][key]["overall_delta"]
        lines.append(
            f"| {label} | {delta['precision']:+.3f} | {delta['recall']:+.3f} | "
            f"{delta['map50']:+.3f} | {delta['map50_95']:+.3f} | {delta['balanced_score']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Class Recall/AP50",
            "",
            "| class | baseline R/AP50 | DiagAug R/AP50 | YOLO default R/AP50 | random R/AP50 | search017 R/AP50 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["comparisons"]["five_way_per_class"]:
        lines.append(
            f"| {row['name']} | {row['baseline_recall']:.3f}/{row['baseline_ap50']:.3f} | "
            f"{row['diagaug_recall']:.3f}/{row['diagaug_ap50']:.3f} | "
            f"{row['yolo_default_recall']:.3f}/{row['yolo_default_ap50']:.3f} | "
            f"{row['random_recall']:.3f}/{row['random_ap50']:.3f} | "
            f"{row['search017_recall']:.3f}/{row['search017_ap50']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Formal Ranking",
            "",
            f"- Best by mAP50-95: `{payload['formal_rank']['best_by_map50_95']}`",
            f"- Best by balanced score: `{payload['formal_rank']['best_by_balanced_score']}`",
            f"- search_policy_017 exceeds random external by mAP50-95: `{str(payload['summary']['beats_random_external_by_map50_95']).lower()}`",
            f"- search_policy_017 exceeds diag_policy_001 by mAP50-95: `{str(payload['summary']['beats_diag_policy_001_by_map50_95']).lower()}`",
            "",
            "## Notes",
            "",
            "- All formal external/offline augmentation runs use original train images plus one augmented copy per train image.",
            "- YOLO built-in augmentation is disabled for baseline, DiagAug, random external, and search_policy_017 unless explicitly listed as the YOLO default control.",
            "- `search_policy_017` is compared as a formal 50 epoch result, not as the previous 5 epoch short-training trend check.",
        ]
    )
    write_markdown(path, lines)


def write_final_training_artifacts(payload: dict[str, Any]) -> None:
    final_dir = OUTPUT_DIR / "final_training"
    final_dir.mkdir(parents=True, exist_ok=True)
    write_text(final_dir / "train_command.txt", payload["commands"]["final_train_command"])
    write_text(final_dir / "val_command.txt", payload["commands"]["final_val_command"])
    write_json(
        final_dir / "final_train_metrics.json",
        {
            "stage": "final_training",
            "status": "completed",
            "data_yaml": payload["augmented_dataset"]["data_yaml"],
            "model": payload["model"],
            "train_settings": payload["train_settings"],
            "command": payload["commands"]["final_train_command"],
            "exitcode": payload["commands"]["final_train_exitcode"],
            "best_pt": payload["artifacts"]["best_pt"],
            "last_pt": payload["artifacts"]["last_pt"],
            "training_results": payload["training_results"],
            "oom": payload["oom"],
        },
    )
    write_json(
        final_dir / "final_val_metrics.json",
        {
            "stage": "final_validation",
            "status": "completed",
            "data_yaml": str(DATA_YAML.resolve()),
            "command": payload["commands"]["final_val_command"],
            "exitcode": payload["commands"]["final_val_exitcode"],
            "metrics": payload["final_metrics"],
            "per_class": payload["final_per_class"],
            "comparisons": payload["comparisons"],
            "oom": payload["oom"],
        },
    )


def update_state_docs(payload: dict[str, Any]) -> None:
    metrics = payload["final_metrics"]
    dataset = payload["augmented_dataset"]
    delta_baseline = payload["comparisons"]["vs_baseline"]["overall_delta"]
    delta_diag = payload["comparisons"]["vs_diagaug"]["overall_delta"]
    delta_random = payload["comparisons"]["vs_random_external"]["overall_delta"]
    section = "\n".join(
        [
            "## search_policy_017 Formal 50 Epoch Result",
            "",
            f"- Run ID: `{RUN_ID}`",
            "- Scope: diagnosis-guided policy search winner promoted to formal 50 epoch YOLO training.",
            "- Training set: original train images + 1x `search_policy_017` augmented train images.",
            "- YOLO built-in augmentations: disabled.",
            f"- Policy: `{format_operations(payload['policy']['operations'])}`",
            f"- Train images / bboxes: `{dataset['train_images']}` / `{dataset['train_bboxes']}`",
            f"- Safety: bbox_valid_rate=`{dataset['bbox_valid_rate']:.6f}`, image_failures=`{dataset['image_failures']}`, label_failures=`{dataset['label_failures']}`",
            f"- Precision: `{metrics['precision']:.3f}`",
            f"- Recall: `{metrics['recall']:.3f}`",
            f"- mAP50: `{metrics['map50']:.3f}`",
            f"- mAP50-95: `{metrics['map50_95']:.3f}`",
            f"- Balanced score: `{metrics['balanced_score']:.6f}`",
            f"- Delta vs baseline: P `{delta_baseline['precision']:+.3f}`, R `{delta_baseline['recall']:+.3f}`, mAP50 `{delta_baseline['map50']:+.3f}`, mAP50-95 `{delta_baseline['map50_95']:+.3f}`",
            f"- Delta vs diag_policy_001: P `{delta_diag['precision']:+.3f}`, R `{delta_diag['recall']:+.3f}`, mAP50 `{delta_diag['map50']:+.3f}`, mAP50-95 `{delta_diag['map50_95']:+.3f}`",
            f"- Delta vs random external: P `{delta_random['precision']:+.3f}`, R `{delta_random['recall']:+.3f}`, mAP50 `{delta_random['map50']:+.3f}`, mAP50-95 `{delta_random['map50_95']:+.3f}`",
            f"- Exceeds random external by mAP50-95: `{str(payload['summary']['beats_random_external_by_map50_95']).lower()}`",
            f"- Exceeds diag_policy_001 by mAP50-95: `{str(payload['summary']['beats_diag_policy_001_by_map50_95']).lower()}`",
            f"- Current best formal by mAP50-95: `{str(payload['summary']['is_best_formal_by_map50_95']).lower()}`",
            f"- Current best formal by balanced score: `{str(payload['summary']['is_best_formal_by_balanced_score']).lower()}`",
            f"- OOM: `{str(payload['oom']).lower()}`",
            f"- Training wall seconds: `{payload['time']['training_wall_seconds']:.1f}`",
            f"- best.pt: `{payload['artifacts']['best_pt']}`",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/search_policy_017_50ep_report.md`",
            f"- Comparison: `outputs/experiments/{RUN_ID}/reports/compare_baseline_diagaug_random_yolo_search017.md`",
            f"- Metrics JSON: `outputs/experiments/{RUN_ID}/reports/search_policy_017_50ep_metrics.json`",
        ]
    )
    for path in [PROJECT_ROOT / "PROJECT_STATE.md", PROJECT_ROOT / "CODEX_HANDOFF.md", PROJECT_ROOT / "EXPERIMENT_LOG.md"]:
        upsert_section(path, "SEARCH_POLICY_017_50EP", section)


def export_project_snapshot() -> None:
    script = PROJECT_ROOT / "scripts" / "export_project_snapshot.py"
    subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT), check=True)


def balanced_score(metrics: dict[str, Any]) -> float:
    return float(
        0.30 * float(metrics["map50_95"])
        + 0.25 * float(metrics["map50"])
        + 0.25 * float(metrics["recall"])
        + 0.20 * float(metrics["precision"])
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8-sig", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def upsert_section(path: Path, key: str, section: str) -> None:
    start = f"<!-- {key}_START -->"
    end = f"<!-- {key}_END -->"
    text = read_text(path)
    block = f"{start}\n{section.rstrip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _match: block, text)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


if __name__ == "__main__":
    main()
