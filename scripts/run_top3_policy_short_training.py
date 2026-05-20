from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostic_pipeline import run_error_diagnosis, run_validation_prediction  # noqa: E402
from AutoAugment.diagnostic_pipeline.common import write_json, write_markdown  # noqa: E402
from AutoAugment.diagnostic_pipeline.dataset_builder import build_final_augmented_dataset  # noqa: E402
from AutoAugment.diagnostics.yolo_error_analysis import load_class_names_from_data_yaml  # noqa: E402
from AutoAugment.utils import IMAGE_EXTENSIONS, resolve_yolo_train_val_records  # noqa: E402


RUN_ID = "20260518_tiled1024_safe_no_ok_position_policy_top3_shorttrain"
TOP3_RUN_ID = "20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline"
DIAGAUG_RUN_ID = "20260518_tiled1024_safe_no_ok_position_diagaug_yolo11n_50ep"

DATASET_ROOT = PROJECT_ROOT / "outputs" / "datasets" / "tiled" / "tiled_1024_ov20_full_safe_no_ok_position"
DATA_YAML = DATASET_ROOT / "data.yaml"
TOP3_JSON = PROJECT_ROOT / "outputs" / "experiments" / TOP3_RUN_ID / "top3_policies" / "top3_policies.json"
DIAGAUG_SELECTED = PROJECT_ROOT / "outputs" / "experiments" / DIAGAUG_RUN_ID / "short_training" / "selected_policy.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / RUN_ID

YOLO = Path("D:/Anaconda/envs/pytorch/Scripts/yolo.exe")
MODEL = "yolo11n.pt"
EPOCHS = 5
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
    configure_environment()
    prepare_output_dir(OUTPUT_DIR)
    class_names = load_class_names_from_data_yaml(DATA_YAML)
    split = resolve_yolo_train_val_records(DATASET_ROOT, seed=SEED)
    top3_payload = read_json(TOP3_JSON)
    previous_selected = read_json(DIAGAUG_SELECTED) if DIAGAUG_SELECTED.exists() else {}

    run_config = {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top3_source_run": TOP3_RUN_ID,
        "source_contract": "single baseline diagnosis -> proxy/safety top3; not multi-round closed-loop search",
        "purpose": "5 epoch short-training validation only; not a final model result",
        "data_yaml": rel(DATA_YAML),
        "model": MODEL,
        "train_settings": {
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": DEFAULT_BATCH,
            "workers": WORKERS,
            "device": DEVICE,
            "seed": SEED,
            "disabled_yolo_augmentations": DISABLED_YOLO_AUGS,
        },
        "score_formula_preferred": "0.30*mAP50 + 0.35*mAP50-95 + 0.20*Recall + 0.15*small_object_recall",
        "score_formula_fallback": "0.35*mAP50 + 0.40*mAP50-95 + 0.25*Recall",
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)

    trial_results = []
    for top_row in top3_payload["top3"]:
        result = run_policy_trial(
            top_row=top_row,
            train_records=split.train_records,
            val_records=split.val_records,
            class_names=class_names,
        )
        trial_results.append(result)

    summary = build_summary(trial_results, top3_payload, previous_selected)
    write_json(OUTPUT_DIR / "reports" / "top3_policy_shorttrain_results.json", summary)
    write_total_report(OUTPUT_DIR / "reports" / "top3_policy_shorttrain_report.md", summary)
    update_state_docs(summary)
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))


def configure_environment() -> None:
    if YOLO.parent.exists():
        os.environ["PATH"] = str(YOLO.parent) + os.pathsep + os.environ.get("PATH", "")
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
        for child in resolved.iterdir():
            if child.name.startswith("runner_"):
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)


def run_policy_trial(
    *,
    top_row: dict[str, Any],
    train_records: list[Any],
    val_records: list[Any],
    class_names: dict[int, str],
) -> dict[str, Any]:
    policy = top_row["policy"]
    policy_id = str(policy["policy_id"])
    policy_dir = OUTPUT_DIR / "policies" / policy_id
    policy_dir.mkdir(parents=True, exist_ok=True)
    write_json(policy_dir / "policy.json", policy)
    write_json(policy_dir / "proxy_metrics.json", slim_proxy_row(top_row))

    dataset_report = build_final_augmented_dataset(
        selected_policy=policy,
        train_records=train_records,
        val_records=val_records,
        output_dir=policy_dir / "dataset_builder",
        class_names=class_names,
        seed=SEED,
        augment_repeat=1,
        dry_run=False,
    )
    dataset_summary = summarize_dataset(dataset_report)
    write_json(policy_dir / "augmented_dataset_summary.json", dataset_summary)

    oom = False
    batch = DEFAULT_BATCH
    train_result = run_train(policy_dir, dataset_summary["data_yaml"], batch=batch)
    if train_result["returncode"] != 0 and has_oom(policy_dir / "train_stdout.log", policy_dir / "train_stderr.log"):
        oom = True
        batch = 1
        train_result = run_train(policy_dir, dataset_summary["data_yaml"], batch=batch)
        if train_result["returncode"] != 0 and has_oom(policy_dir / "train_stdout.log", policy_dir / "train_stderr.log"):
            raise RuntimeError(f"{policy_id} still OOM at batch=1; stopping as requested")
    if train_result["returncode"] != 0:
        raise RuntimeError(f"short training failed for {policy_id}; see {policy_dir / 'train_stderr.log'}")

    copy_weights(policy_dir)
    val_result = run_val(policy_dir, dataset_summary["data_yaml"], batch=batch)
    if val_result["returncode"] != 0 and has_oom(policy_dir / "val_stdout.log", policy_dir / "val_stderr.log"):
        oom = True
        batch = 1
        val_result = run_val(policy_dir, dataset_summary["data_yaml"], batch=batch)
    if val_result["returncode"] != 0:
        raise RuntimeError(f"short validation failed for {policy_id}; see {policy_dir / 'val_stderr.log'}")

    val_metrics = parse_yolo_val_log(policy_dir / "val_stdout.log", class_names)
    small_object = compute_small_object_recall(
        policy_dir=policy_dir,
        best_pt=policy_dir / "weights" / "best.pt",
        val_images_dir=Path(dataset_summary["images_val"]),
        val_labels_dir=Path(dataset_summary["labels_val"]),
        class_names=class_names,
        batch=batch,
    )
    score = compute_short_train_score(val_metrics["overall"], small_object)
    train_loop = read_training_summary(policy_dir / "train" / "results.csv")
    payload = {
        "policy_id": policy_id,
        "source_issue": policy.get("source_issue"),
        "source_issues": policy.get("source_issues", []),
        "operations": policy.get("operations", []),
        "contains_copy_paste": contains_copy_paste(policy),
        "proxy_rank": int(top_row.get("rank", 0) or 0),
        "proxy_score": float(top_row.get("proxy_score", 0.0) or 0.0),
        "safety_score": float(top_row.get("safety_score", 0.0) or 0.0),
        "combined_proxy_safety_score": float(top_row.get("combined_proxy_safety_score", 0.0) or 0.0),
        "proxy_hard_filter_pass": bool(top_row.get("hard_filter_pass")),
        "proxy_hard_filter_reasons": list(top_row.get("hard_filter_reasons", []) or []),
        "proxy_soft_penalty_reasons": list(top_row.get("safety_soft_penalty_reasons", []) or []),
        "copy_paste_audit": top_row.get("copy_paste_audit", {}),
        "augmented_dataset": dataset_summary,
        "train_settings": {
            "model": MODEL,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": batch,
            "workers": WORKERS,
            "device": DEVICE,
            "seed": SEED,
            "disabled_yolo_augmentations": DISABLED_YOLO_AUGS,
        },
        "commands": {
            "train_command": train_result["command"],
            "val_command": val_result["command"],
            "train_returncode": train_result["returncode"],
            "val_returncode": val_result["returncode"],
        },
        "artifacts": {
            "policy_dir": str(policy_dir.resolve()),
            "data_yaml": dataset_summary["data_yaml"],
            "best_pt": str((policy_dir / "weights" / "best.pt").resolve()),
            "last_pt": str((policy_dir / "weights" / "last.pt").resolve()),
            "train_stdout": str((policy_dir / "train_stdout.log").resolve()),
            "train_stderr": str((policy_dir / "train_stderr.log").resolve()),
            "val_stdout": str((policy_dir / "val_stdout.log").resolve()),
            "val_stderr": str((policy_dir / "val_stderr.log").resolve()),
        },
        "metrics": {
            **val_metrics["overall"],
            "per_class": val_metrics["per_class"],
            "small_object_recall": small_object.get("small_object_recall"),
            "small_object_recall_basis": small_object,
            "short_train_score": score["score"],
            "short_train_score_formula": score["formula"],
        },
        "last_epoch_metrics_from_training_loop": train_loop.get("last_epoch", {}),
        "best_epoch_by_map50_95_from_training_loop": train_loop.get("best_epoch_by_map50_95", {}),
        "oom": oom,
        "training_wall_seconds": train_result["wall_seconds"],
        "validation_wall_seconds": val_result["wall_seconds"],
    }
    write_json(policy_dir / "metrics.json", payload)
    write_policy_report(policy_dir / "report.md", payload)
    return payload


def run_train(policy_dir: Path, data_yaml: str, *, batch: int) -> dict[str, Any]:
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
        f"project={policy_dir}",
        "name=train",
        "exist_ok=True",
        *[f"{key}={value}" for key, value in DISABLED_YOLO_AUGS.items()],
    ]
    return run_command(command, policy_dir, "train")


def run_val(policy_dir: Path, data_yaml: str, *, batch: int) -> dict[str, Any]:
    command = [
        str(YOLO),
        "detect",
        "val",
        f"model={policy_dir / 'weights' / 'best.pt'}",
        f"data={data_yaml}",
        f"imgsz={IMGSZ}",
        f"batch={batch}",
        f"workers={WORKERS}",
        f"device={DEVICE}",
        f"project={policy_dir}",
        "name=val",
        "exist_ok=True",
    ]
    return run_command(command, policy_dir, "val")


def run_command(command: list[str], output_dir: Path, prefix: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command_text = subprocess.list2cmdline(command)
    (output_dir / f"{prefix}_command.txt").write_text(command_text + "\n", encoding="utf-8")
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )
    wall = time.time() - start
    (output_dir / f"{prefix}_stdout.log").write_text(completed.stdout, encoding="utf-8", errors="replace")
    (output_dir / f"{prefix}_stderr.log").write_text(completed.stderr, encoding="utf-8", errors="replace")
    return {
        "command": command_text,
        "returncode": completed.returncode,
        "wall_seconds": wall,
    }


def copy_weights(policy_dir: Path) -> None:
    src = policy_dir / "train" / "weights"
    dst = policy_dir / "weights"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["best.pt", "last.pt"]:
        source = src / name
        if not source.exists():
            raise FileNotFoundError(f"missing trained weight: {source}")
        shutil.copy2(source, dst / name)


def compute_small_object_recall(
    *,
    policy_dir: Path,
    best_pt: Path,
    val_images_dir: Path,
    val_labels_dir: Path,
    class_names: dict[int, str],
    batch: int,
) -> dict[str, Any]:
    try:
        pred_record = run_validation_prediction(
            weights=best_pt,
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
            output_dir=policy_dir / "diagnosis_prediction",
            imgsz=IMGSZ,
            workers=WORKERS,
            device=DEVICE,
            conf=0.25,
            iou=0.5,
            dry_run=False,
        )
        diagnosis = run_error_diagnosis(
            val_images_dir=val_images_dir,
            val_labels_dir=val_labels_dir,
            predictions_dir=pred_record["predictions_dir"],
            output_dir=policy_dir / "diagnosis",
            class_names=class_names,
            match_iou=0.5,
        )
        by_size = diagnosis.get("size_recall", {})
        tiny = by_size.get("tiny", {})
        small = by_size.get("small", {})
        gt = int(tiny.get("gt_count", 0) or 0) + int(small.get("gt_count", 0) or 0)
        tp = int(tiny.get("tp_count", 0) or 0) + int(small.get("tp_count", 0) or 0)
        return {
            "available": True,
            "formula_used": "preferred",
            "small_object_recall": tp / max(1, gt),
            "tiny_gt": int(tiny.get("gt_count", 0) or 0),
            "small_gt": int(small.get("gt_count", 0) or 0),
            "small_object_gt": gt,
            "small_object_tp": tp,
            "prediction_batch": batch,
        }
    except Exception as exc:
        return {
            "available": False,
            "formula_used": "fallback",
            "small_object_recall": None,
            "error": str(exc),
        }


def compute_short_train_score(overall: dict[str, Any], small_object: dict[str, Any]) -> dict[str, Any]:
    map50 = float(overall["map50"])
    map50_95 = float(overall["map50_95"])
    recall = float(overall["recall"])
    small_recall = small_object.get("small_object_recall")
    if small_object.get("available") and small_recall is not None:
        score = 0.30 * map50 + 0.35 * map50_95 + 0.20 * recall + 0.15 * float(small_recall)
        formula = "0.30*mAP50 + 0.35*mAP50-95 + 0.20*Recall + 0.15*small_object_recall"
    else:
        score = 0.35 * map50 + 0.40 * map50_95 + 0.25 * recall
        formula = "0.35*mAP50 + 0.40*mAP50-95 + 0.25*Recall"
    return {"score": float(score), "formula": formula}


def summarize_dataset(dataset_report: dict[str, Any]) -> dict[str, Any]:
    train_labels = Path(dataset_report["labels_train"])
    val_labels = Path(dataset_report["labels_val"])
    train_images = Path(dataset_report["images_train"])
    val_images = Path(dataset_report["images_val"])
    payload = {
        **dataset_report,
        "train_images": count_images(train_images),
        "train_bboxes": count_label_rows(train_labels),
        "val_images": count_images(val_images),
        "val_bboxes": count_label_rows(val_labels),
    }
    return payload


def count_images(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def count_label_rows(root: Path) -> int:
    total = 0
    for path in root.rglob("*.txt"):
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def parse_yolo_val_log(path: Path, class_names: dict[int, str]) -> dict[str, Any]:
    text = strip_ansi(path.read_text(encoding="utf-8", errors="replace"))
    rows: list[dict[str, Any]] = []
    overall: dict[str, Any] | None = None
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        values = numeric_tail(parts, 6)
        if values is None:
            continue
        images, instances, precision, recall, ap50, ap50_95 = values
        if parts[0] == "all":
            overall = {
                "images": int(images),
                "instances": int(instances),
                "precision": precision,
                "recall": recall,
                "map50": ap50,
                "map50_95": ap50_95,
            }
            continue
        if overall is None or len(rows) >= len(class_names):
            continue
        class_id = len(rows)
        rows.append(
            {
                "class_id": class_id,
                "name": class_names.get(class_id, str(class_id)),
                "log_name": parts[0],
                "images": int(images),
                "instances": int(instances),
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ap50_95": ap50_95,
            }
        )
    if overall is None:
        raise RuntimeError(f"could not parse YOLO val metrics from {path}")
    if len(rows) != len(class_names):
        raise RuntimeError(f"expected {len(class_names)} class rows from {path}, parsed {len(rows)}")
    return {"overall": overall, "per_class": rows}


def numeric_tail(parts: list[str], count: int) -> list[float] | None:
    values: list[float] = []
    for item in reversed(parts):
        try:
            values.append(float(item))
        except ValueError:
            continue
        if len(values) == count:
            return list(reversed(values))
    return None


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def read_training_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def metrics_from_row(row: dict[str, str]) -> dict[str, Any]:
        return {
            "epoch": int(float(row["epoch"])),
            "time_seconds": float(row["time"]),
            "precision": float(row["metrics/precision(B)"]),
            "recall": float(row["metrics/recall(B)"]),
            "map50": float(row["metrics/mAP50(B)"]),
            "map50_95": float(row["metrics/mAP50-95(B)"]),
        }

    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "last_epoch": metrics_from_row(rows[-1]),
        "best_epoch_by_map50_95": metrics_from_row(best),
        "epoch_count": len(rows),
    }


def build_summary(
    trial_results: list[dict[str, Any]],
    top3_payload: dict[str, Any],
    previous_selected: dict[str, Any],
) -> dict[str, Any]:
    best = max(trial_results, key=lambda item: float(item["metrics"]["short_train_score"]))
    copy_paste = next((item for item in trial_results if item["contains_copy_paste"]), None)
    low_contrast_best = max(
        [item for item in trial_results if item["source_issue"] == "low_contrast_missed_defect"],
        key=lambda item: float(item["metrics"]["short_train_score"]),
    )
    copy_paste_summary = None
    if copy_paste is not None:
        copy_paste_summary = {
            "policy_id": copy_paste["policy_id"],
            "score": copy_paste["metrics"]["short_train_score"],
            "beats_best_low_contrast_policy": copy_paste["metrics"]["short_train_score"]
            > low_contrast_best["metrics"]["short_train_score"],
            "bbox_safety_issue": bool(copy_paste.get("proxy_hard_filter_reasons")),
            "hard_filter_pass": bool(copy_paste.get("proxy_hard_filter_pass")),
            "worth_formal_50epoch": copy_paste["policy_id"] == best["policy_id"],
        }
    return {
        "run_id": RUN_ID,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top3_source_run": TOP3_RUN_ID,
        "source_scope": "same baseline diagnosis; proxy/safety top3; not multi-round closed-loop",
        "short_training_scope": "5 epochs per policy; strategy validation only; not final model result",
        "data_yaml": rel(DATA_YAML),
        "trial_count": len(trial_results),
        "trials": trial_results,
        "best_policy_id": best["policy_id"],
        "best_policy": best,
        "previous_formal_diagaug_policy_id": previous_selected.get("policy_id"),
        "best_matches_formal_diagaug_policy": best["policy_id"] == previous_selected.get("policy_id"),
        "copy_paste_policy_summary": copy_paste_summary,
        "recommendation": formal_retrain_recommendation(best, previous_selected, copy_paste_summary),
        "summary": {
            "best_policy_id": best["policy_id"],
            "best_matches_diag_policy_001": best["policy_id"] == "diag_policy_001",
            "best_matches_formal_diagaug_policy": best["policy_id"] == previous_selected.get("policy_id"),
            "scores": {
                item["policy_id"]: item["metrics"]["short_train_score"]
                for item in trial_results
            },
        },
    }


def formal_retrain_recommendation(
    best: dict[str, Any],
    previous_selected: dict[str, Any],
    copy_paste_summary: dict[str, Any] | None,
) -> str:
    if best["policy_id"] != previous_selected.get("policy_id"):
        return (
            "Short-training top1 differs from the current formal DiagAug policy. "
            "The existing 50 epoch DiagAug result represents only the earlier proxy top1; "
            "run a new formal 50 epoch experiment with the short-training top1 policy."
        )
    if copy_paste_summary and copy_paste_summary.get("beats_best_low_contrast_policy"):
        return (
            "copy_paste was competitive enough to justify an additional formal run, "
            "although the short-training top1 still matches the current formal DiagAug policy."
        )
    return (
        "Short-training top1 matches the current formal DiagAug policy. "
        "A rerun is not strictly required for strategy selection, but can be used for confirmation."
    )


def write_policy_report(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    dataset = payload["augmented_dataset"]
    lines = [
        f"# Short-Training Report: {payload['policy_id']}",
        "",
        f"- policy_id: `{payload['policy_id']}`",
        f"- source_issue: `{payload['source_issue']}`",
        f"- operations: `{format_operations(payload['operations'])}`",
        f"- contains_copy_paste: `{str(payload['contains_copy_paste']).lower()}`",
        f"- train images / bboxes: `{dataset['train_images']}` / `{dataset['train_bboxes']}`",
        f"- val images / bboxes: `{dataset['val_images']}` / `{dataset['val_bboxes']}`",
        f"- Precision: `{metrics['precision']:.3f}`",
        f"- Recall: `{metrics['recall']:.3f}`",
        f"- mAP50: `{metrics['map50']:.3f}`",
        f"- mAP50-95: `{metrics['map50_95']:.3f}`",
        f"- small_object_recall: `{metrics.get('small_object_recall')}`",
        f"- short_train_score: `{metrics['short_train_score']:.6f}`",
        f"- score formula: `{metrics['short_train_score_formula']}`",
        f"- OOM: `{str(payload['oom']).lower()}`",
        f"- training wall seconds: `{payload['training_wall_seconds']:.1f}`",
        "",
        "## Per-Class Recall/AP50",
        "",
        "| class id | class | Recall | AP50 |",
        "| ---: | --- | ---: | ---: |",
    ]
    for row in metrics["per_class"]:
        lines.append(f"| {row['class_id']} | {row['name']} | {row['recall']:.3f} | {row['ap50']:.3f} |")
    write_markdown(path, lines)


def write_total_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Top3 Policy Short-Training Report",
        "",
        "## Scope",
        "",
        "- Top3 policies come from the same baseline diagnosis.",
        "- This is not multi-round closed-loop optimization.",
        "- These are proxy/safety top3 candidates.",
        "- Each policy was trained for 5 epochs only.",
        "- This is not a final model result; it validates strategy selection and informs whether to rerun formal 50 epoch DiagAug.",
        "",
        "## Comparison",
        "",
        "| policy_id | source_issue | operations | copy_paste | proxy_rank | Precision | Recall | mAP50 | mAP50-95 | short_train_score |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["trials"]:
        metrics = item["metrics"]
        lines.append(
            "| {policy_id} | {source_issue} | {ops} | {copy_paste} | {proxy_rank} | {precision:.3f} | {recall:.3f} | {map50:.3f} | {map50_95:.3f} | {score:.6f} |".format(
                policy_id=item["policy_id"],
                source_issue=item["source_issue"],
                ops=escape_pipe(format_operations(item["operations"])),
                copy_paste=item["contains_copy_paste"],
                proxy_rank=item["proxy_rank"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                map50=metrics["map50"],
                map50_95=metrics["map50_95"],
                score=metrics["short_train_score"],
            )
        )
    best = summary["best_policy"]
    copy_paste_summary = summary.get("copy_paste_policy_summary") or {}
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Short-training best policy: `{summary['best_policy_id']}`",
            f"- Matches current formal DiagAug policy `{summary.get('previous_formal_diagaug_policy_id')}`: `{str(summary['best_matches_formal_diagaug_policy']).lower()}`",
            f"- Score formula used: `{best['metrics']['short_train_score_formula']}`",
            "",
            "## copy_paste Policy",
            "",
            f"- Policy: `{copy_paste_summary.get('policy_id')}`",
            f"- Beats best low-contrast policy: `{str(copy_paste_summary.get('beats_best_low_contrast_policy')).lower()}`",
            f"- Bbox safety issue: `{str(copy_paste_summary.get('bbox_safety_issue')).lower()}`",
            f"- Worth formal 50 epoch: `{str(copy_paste_summary.get('worth_formal_50epoch')).lower()}`",
            "",
            "## Recommendation",
            "",
            summary["recommendation"],
        ]
    )
    write_markdown(path, lines)


def update_state_docs(summary: dict[str, Any]) -> None:
    section = build_state_section(summary)
    for path in [PROJECT_ROOT / "PROJECT_STATE.md", PROJECT_ROOT / "CODEX_HANDOFF.md", PROJECT_ROOT / "EXPERIMENT_LOG.md"]:
        upsert_section(path, "TOP3_POLICY_SHORTTRAIN", section)


def build_state_section(summary: dict[str, Any]) -> str:
    score_bits = ", ".join(f"{pid}={score:.6f}" for pid, score in summary["summary"]["scores"].items())
    return "\n".join(
        [
            "## Top3 Policy Short-Training Validation",
            "",
            f"- Run ID: `{RUN_ID}`",
            f"- Source top3 run: `{TOP3_RUN_ID}`",
            "- Scope: top3 short-training strategy validation only; not a final model result.",
            "- Each policy trained for `5` epochs with YOLO built-in augmentations disabled.",
            "- Top3 came from one baseline diagnosis and proxy/safety ranking, not multi-round closed-loop search.",
            f"- Short-training scores: `{score_bits}`",
            f"- Best short-training policy: `{summary['best_policy_id']}`",
            f"- Matches current formal DiagAug policy: `{str(summary['best_matches_formal_diagaug_policy']).lower()}`",
            "- Result should decide whether a formal DiagAug 50 epoch rerun is needed.",
            f"- Report: `outputs/experiments/{RUN_ID}/reports/top3_policy_shorttrain_report.md`",
            f"- Results JSON: `outputs/experiments/{RUN_ID}/reports/top3_policy_shorttrain_results.json`",
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


def slim_proxy_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rank",
        "policy_id",
        "proxy_score",
        "safety_score",
        "combined_proxy_safety_score",
        "hard_filter_pass",
        "hard_filter_reasons",
        "safety_soft_penalty_reasons",
        "copy_paste_audit",
    ]
    return {key: row.get(key) for key in keys}


def has_oom(*paths: Path) -> bool:
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace").lower() for path in paths if path.exists())
    return any(pattern in text for pattern in ["out of memory", "cuda oom", "cuda out of memory"])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def contains_copy_paste(policy: dict[str, Any]) -> bool:
    return any(str(op.get("name", "")).lower() == "copy_paste" for op in policy.get("operations", []) or [])


def format_operations(operations: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{op.get('name')}(p={float(op.get('prob', 0.0)):.3f}, s={float(op.get('strength', 0.0)):.3f})"
        for op in operations
    )


def escape_pipe(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
