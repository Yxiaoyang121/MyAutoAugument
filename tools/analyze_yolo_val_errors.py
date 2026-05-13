from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.diagnostics import generate_augmentation_advice
from AutoAugment.diagnostics.yolo_error_analysis import (
    analyze_yolo_errors,
    load_class_names_from_data_yaml,
    write_analysis_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze YOLO validation TP/FP/FN errors and generate augmentation advice.")
    parser.add_argument("--dataset", required=True, help="YOLO dataset root containing images/val and labels/val.")
    parser.add_argument("--weights", required=True, help="YOLO weights for prediction, usually baseline best.pt.")
    parser.add_argument("--output", required=True, help="Diagnostic output directory.")
    parser.add_argument("--data-yaml", default=None, help="Optional data.yaml with class names. Defaults to dataset/data.yaml.")
    parser.add_argument("--predictions", default=None, help="Optional existing YOLO prediction labels directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO predict NMS IoU and default matching IoU.")
    parser.add_argument("--match-iou", type=float, default=None, help="IoU threshold for TP matching. Defaults to --iou.")
    parser.add_argument("--localization-weak-iou", type=float, default=0.3)
    parser.add_argument("--max-visualizations", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    weights = Path(args.weights).resolve()
    output = Path(args.output).resolve()
    images_dir = dataset / "images" / "val"
    labels_dir = dataset / "labels" / "val"
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if not images_dir.exists():
        raise FileNotFoundError(f"validation images directory does not exist: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"validation labels directory does not exist: {labels_dir}")
    if not weights.exists():
        raise FileNotFoundError(f"weights do not exist: {weights}")
    output.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data_yaml).resolve() if args.data_yaml else dataset / "data.yaml"
    class_names = load_class_names_from_data_yaml(data_yaml) if data_yaml.exists() else {}
    predictions_dir = Path(args.predictions).resolve() if args.predictions else run_yolo_predict(
        weights=weights,
        images_dir=images_dir,
        output_dir=output,
        imgsz=args.imgsz,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
    )
    match_iou = args.match_iou if args.match_iou is not None else args.iou
    analysis = analyze_yolo_errors(
        images_dir=images_dir,
        labels_dir=labels_dir,
        predictions_dir=predictions_dir,
        class_names=class_names,
        match_iou=match_iou,
        localization_weak_iou=args.localization_weak_iou,
    )
    advice = generate_augmentation_advice(analysis["summary"])
    write_analysis_outputs(output, analysis=analysis, advice=advice)
    write_visualizations(
        output / "visualizations",
        per_object_errors=analysis["per_object_errors"],
        max_per_group=args.max_visualizations,
    )

    print("YOLO validation error analysis completed:")
    print(f"- Dataset: {dataset}")
    print(f"- Weights: {weights}")
    print(f"- Predictions: {predictions_dir}")
    print(f"- Output: {output}")
    print(f"- Summary: {output / 'error_summary.json'}")
    print(f"- Advice: {output / 'augmentation_advice.json'}")
    print(f"- Advisor search space: {output / 'advisor_search_space.json'}")


def run_yolo_predict(
    *,
    weights: Path,
    images_dir: Path,
    output_dir: Path,
    imgsz: int,
    workers: int,
    conf: float,
    iou: float,
) -> Path:
    project = output_dir / "predict_runs"
    command = (
        f"yolo detect predict model={quote_path(weights)} source={quote_path(images_dir)} "
        f"imgsz={imgsz} conf={conf} iou={iou} save_txt=True save_conf=True workers={workers} "
        f"project={quote_path(project)} name=pred exist_ok=True"
    )
    completed = run_command(command)
    (output_dir / "predict_stdout.log").write_text(completed.stdout, encoding="utf-8", errors="replace")
    (output_dir / "predict_stderr.log").write_text(completed.stderr, encoding="utf-8", errors="replace")
    (output_dir / "predict_command.txt").write_text(command + "\n", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"YOLO predict failed. See {output_dir / 'predict_stderr.log'}")
    labels_dir = project / "pred" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    return labels_dir


def write_visualizations(output_dir: Path, *, per_object_errors: list[dict], max_per_group: int) -> None:
    groups = {
        "FN": "false_negatives",
        "FP": "false_positives",
        "TP": "true_positives",
        "localization_weak": "low_confidence",
    }
    counters = {name: 0 for name in groups.values()}
    edge_counter = 0
    for record in per_object_errors:
        group = groups.get(record["error_type"])
        if group and counters[group] < max_per_group:
            _write_record_visualization(output_dir / group, record)
            counters[group] += 1
        if record.get("near_edge") and edge_counter < max_per_group:
            _write_record_visualization(output_dir / "edge_cases", record)
            edge_counter += 1


def _write_record_visualization(output_dir: Path, record: dict) -> None:
    image_path = Path(record["image_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    error_type = record["error_type"]
    if record.get("gt_bbox") is not None:
        color = (0, 0, 255) if error_type == "FN" else (0, 255, 0)
        _draw_box(image, record["gt_bbox"], color, f"GT {record['class_name']} {error_type}")
    if record.get("pred_bbox") is not None:
        color = (0, 165, 255) if error_type == "FP" else (0, 255, 0)
        conf = record.get("confidence")
        iou = record.get("iou")
        label = f"P {record['class_name']}"
        if conf is not None:
            label += f" c={conf:.2f}"
        if iou is not None:
            label += f" iou={iou:.2f}"
        _draw_box(image, record["pred_bbox"], color, label)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem.replace(" ", "_")
    index = len(list(output_dir.glob(f"{stem}_*.jpg")))
    cv2.imwrite(str(output_dir / f"{stem}_{index:03d}.jpg"), image)


def _draw_box(image: np.ndarray, box: list[float], color: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def quote_path(path: str | Path) -> str:
    text = Path(path).resolve().as_posix() if isinstance(path, Path) else str(path)
    return f'"{text}"' if any(char.isspace() for char in text) else text


def run_command(command: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
