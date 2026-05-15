from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from AutoAugment.diagnostics.yolo_error_analysis import load_yolo_predictions
from AutoAugment.formats.yolo import load_yolo_labels
from AutoAugment.diagnostic_pipeline.common import (
    maybe_device_arg,
    quote_path,
    run_logged_command,
    write_json,
)
from AutoAugment.utils import IMAGE_EXTENSIONS


def run_validation_prediction(
    *,
    weights: str | Path,
    val_images_dir: str | Path,
    val_labels_dir: str | Path,
    output_dir: str | Path,
    imgsz: int,
    workers: int = 0,
    device: str | int | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
    dry_run: bool = False,
    existing_predictions_dir: str | Path | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run YOLO prediction on the validation set and record per-image boxes."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights_path = Path(weights)
    val_images = Path(val_images_dir)
    val_labels = Path(val_labels_dir)
    predictions_dir = Path(existing_predictions_dir) if existing_predictions_dir else output / "predict_runs" / "pred" / "labels"
    command = (
        f"yolo detect predict model={quote_path(weights_path)} source={quote_path(val_images)} "
        f"imgsz={imgsz} conf={conf} iou={iou} save_txt=True save_conf=True workers={workers}"
        f"{maybe_device_arg(device)} project={quote_path(output / 'predict_runs')} name=pred exist_ok=True"
    )
    if existing_predictions_dir is None:
        command_result = run_logged_command(
            command,
            output_dir=output,
            prefix="predict",
            dry_run=dry_run,
            timeout=timeout,
        )
        if command_result.returncode not in {0, None}:
            raise RuntimeError(f"YOLO validation predict failed; see {command_result.stderr_log}")
    else:
        (output / "predict_command.txt").write_text(f"[existing predictions] {predictions_dir}\n", encoding="utf-8")
        (output / "predict_stdout.log").write_text("", encoding="utf-8")
        (output / "predict_stderr.log").write_text("", encoding="utf-8")

    if dry_run:
        records: list[dict[str, Any]] = []
    else:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        records = collect_validation_prediction_records(
            val_images_dir=val_images,
            val_labels_dir=val_labels,
            predictions_dir=predictions_dir,
        )

    record = {
        "stage": "validation_prediction",
        "status": "planned" if dry_run else "completed",
        "dry_run": dry_run,
        "weights": str(weights_path.resolve()),
        "val_images_dir": str(val_images.resolve()),
        "val_labels_dir": str(val_labels.resolve()),
        "predictions_dir": str(predictions_dir.resolve()),
        "imgsz": int(imgsz),
        "workers": int(workers),
        "conf": float(conf),
        "iou": float(iou),
        "prediction_count": sum(len(item["predictions"]) for item in records),
        "image_count": len(records),
        "records": records,
    }
    write_json(output / "validation_predictions.json", record)
    return record


def collect_validation_prediction_records(
    *,
    val_images_dir: str | Path,
    val_labels_dir: str | Path,
    predictions_dir: str | Path,
) -> list[dict[str, Any]]:
    """Collect GT and prediction labels for each validation image."""

    images_root = Path(val_images_dir)
    labels_root = Path(val_labels_dir)
    predictions_root = Path(predictions_dir)
    records: list[dict[str, Any]] = []
    image_paths = sorted(path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        pred_path = predictions_root / relative.with_suffix(".txt")
        gt_labels, gt_boxes = load_yolo_labels(label_path, width, height)
        pred_labels, pred_boxes, pred_confs = load_yolo_predictions(pred_path, width, height)
        records.append(
            {
                "image_path": str(image_path.resolve()),
                "relative_path": relative.as_posix(),
                "ground_truth": [
                    {"class_id": int(label), "bbox_xyxy": [float(value) for value in box]}
                    for label, box in zip(gt_labels, gt_boxes)
                ],
                "predictions": [
                    {
                        "class_id": int(label),
                        "bbox_xyxy": [float(value) for value in box],
                        "confidence": float(confidence),
                    }
                    for label, box, confidence in zip(pred_labels, pred_boxes, pred_confs)
                ],
            }
        )
    return records
