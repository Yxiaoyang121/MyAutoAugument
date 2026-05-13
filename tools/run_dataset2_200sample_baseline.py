from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostics import generate_augmentation_advice
from AutoAugment.diagnostics.yolo_error_analysis import analyze_yolo_errors, write_analysis_outputs
from AutoAugment.search import parse_yolo_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed 200-sample no-augmentation YOLO baseline.")
    parser.add_argument("--dataset", default="E:/TJGY/DataSet2_fixed_200sample")
    parser.add_argument("--output", default="outputs/baseline_dataset2_fixed_200sample_5epochs_workers0")
    parser.add_argument("--wrapper-source", default="outputs/baseline_dataset2_fixed_5epochs_workers0/yolo_single_thread.py")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    data_yaml = dataset / "data.yaml"
    if not dataset.exists():
        raise FileNotFoundError(f"dataset does not exist: {dataset}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml does not exist: {data_yaml}")
    if output.exists():
        raise FileExistsError(f"output already exists, refusing to overwrite: {output}")
    if args.workers != 0:
        raise ValueError("this comparison baseline must use workers=0")

    output.mkdir(parents=True)
    wrapper_source = Path(args.wrapper_source)
    if not wrapper_source.exists():
        raise FileNotFoundError(f"single-thread wrapper source does not exist: {wrapper_source}")
    wrapper = output / "yolo_single_thread.py"
    shutil.copyfile(wrapper_source, wrapper)

    config = {
        "dataset": str(dataset),
        "dataset_yaml": str(data_yaml),
        "output": str(output),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "single_thread_yolo_wrapper": True,
        "wrapper": str(wrapper),
        "custom_autoaugment_policy": False,
        "llm": False,
        "ablation": False,
        "final_full_dataset": False,
    }
    write_json(output / "baseline_stage_config.json", config)

    train_command = [
        sys.executable,
        "-X",
        "utf8",
        str(wrapper),
        "train",
        "--model",
        args.model,
        "--data",
        data_yaml.as_posix(),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--project",
        (output / "train_runs").as_posix(),
        "--name",
        "train",
        "--exist-ok",
    ]
    train_completed = run_command(train_command)
    write_text(output / "train_stdout.log", train_completed.stdout)
    write_text(output / "train_stderr.log", train_completed.stderr)
    if train_completed.returncode != 0:
        raise RuntimeError(f"baseline train failed; see {output / 'train_stderr.log'}")

    best_pt = copy_best_pt(output)
    val_command = [
        sys.executable,
        "-X",
        "utf8",
        str(wrapper),
        "val",
        "--model",
        best_pt.as_posix(),
        "--data",
        data_yaml.as_posix(),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--project",
        (output / "val_runs").as_posix(),
        "--name",
        "val",
        "--exist-ok",
    ]
    val_completed = run_command(val_command)
    write_text(output / "val_stdout.log", val_completed.stdout)
    write_text(output / "val_stderr.log", val_completed.stderr)
    if val_completed.returncode != 0:
        raise RuntimeError(f"baseline val failed; see {output / 'val_stderr.log'}")

    pred_labels_dir = run_predict(
        wrapper=wrapper,
        output=output,
        best_pt=best_pt,
        val_images=dataset / "images" / "val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        conf=args.conf,
        iou=args.iou,
    )
    analysis = analyze_yolo_errors(
        images_dir=dataset / "images" / "val",
        labels_dir=dataset / "labels" / "val",
        predictions_dir=pred_labels_dir,
        class_names={},
        match_iou=args.iou,
    )
    advice = generate_augmentation_advice(analysis["summary"])
    diagnosis_dir = output / "diagnosis"
    write_analysis_outputs(diagnosis_dir, analysis=analysis, advice=advice)

    parsed = parse_yolo_metrics(f"{val_completed.stdout}\n{val_completed.stderr}")
    yolo_pr = parse_precision_recall(f"{val_completed.stdout}\n{val_completed.stderr}")
    overall = analysis["summary"]["overall"]
    metrics = {
        "mAP50": parsed.get("map50"),
        "mAP50-95": parsed.get("map50_95"),
        "Precision": overall.get("precision"),
        "Recall": overall.get("recall"),
        "fitness": parsed.get("map50_95"),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "model": args.model,
        "dataset": str(dataset),
        "seed": args.seed,
        "precision_recall_source": "diagnostic_analyzer_conf0.25_iou0.5, matching closed-loop P/R source",
        "yolo_val_precision": yolo_pr.get("precision"),
        "yolo_val_recall": yolo_pr.get("recall"),
        "yolo_val_images": count_images(dataset / "images" / "val"),
        "yolo_val_instances": overall.get("gt_count"),
        "tp": overall.get("tp_count"),
        "fp": overall.get("fp_count"),
        "fn": overall.get("fn_count"),
        "localization_weak_count": overall.get("localization_weak_count"),
        "train_stdout_log": str(output / "train_stdout.log"),
        "val_stdout_log": str(output / "val_stdout.log"),
        "predict_stdout_log": str(output / "predict_stdout.log"),
        "weights_best_pt": str(best_pt),
        "wrapper": str(wrapper),
        "diagnosis_dir": str(diagnosis_dir),
        "single_thread_yolo_wrapper": True,
        "custom_autoaugment_policy": False,
        "llm": False,
        "ablation": False,
        "final_full_dataset": False,
    }
    write_json(output / "baseline_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def run_predict(
    *,
    wrapper: Path,
    output: Path,
    best_pt: Path,
    val_images: Path,
    imgsz: int,
    batch: int,
    workers: int,
    seed: int,
    conf: float,
    iou: float,
) -> Path:
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(wrapper),
        "predict",
        "--model",
        best_pt.as_posix(),
        "--source",
        val_images.as_posix(),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--seed",
        str(seed),
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--save-txt",
        "--save-conf",
        "--project",
        (output / "diagnosis_runs").as_posix(),
        "--name",
        "pred",
        "--exist-ok",
    ]
    completed = run_command(command)
    write_text(output / "predict_stdout.log", completed.stdout)
    write_text(output / "predict_stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"baseline predict failed; see {output / 'predict_stderr.log'}")
    labels_dir = output / "diagnosis_runs" / "pred" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    return labels_dir


def copy_best_pt(output: Path) -> Path:
    candidates = [
        output / "train_runs" / "train" / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / "detect" / output.relative_to(PROJECT_ROOT) / "train_runs" / "train" / "weights" / "best.pt",
    ]
    candidates.extend(sorted((output / "train_runs").rglob("best.pt"), key=lambda item: item.stat().st_mtime, reverse=True))
    for candidate in candidates:
        if candidate.exists():
            weights_dir = output / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            destination = weights_dir / "best.pt"
            if candidate.resolve() != destination.resolve():
                shutil.copyfile(candidate, destination)
            return destination.resolve()
    raise FileNotFoundError(f"could not find best.pt under {output / 'train_runs'}")


def parse_precision_recall(text: str) -> dict[str, float]:
    lines = [strip_ansi(line).strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        tokens = line.split()
        if not tokens or tokens[0].lower() != "all":
            continue
        if not near_metric_header(lines, index):
            continue
        numeric_values = [value for value in (float_or_none(token) for token in tokens[1:]) if value is not None]
        if len(numeric_values) >= 6:
            return {"precision": numeric_values[-4], "recall": numeric_values[-3]}
    return {}


def near_metric_header(lines: list[str], index: int) -> bool:
    for candidate in lines[max(0, index - 4) : index]:
        normalized = "".join(char for char in candidate.lower() if char.isalnum())
        if "map50" in normalized and "map5095" in normalized:
            return True
    return False


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def count_images(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"})


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
