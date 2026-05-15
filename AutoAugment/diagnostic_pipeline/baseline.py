from __future__ import annotations

from pathlib import Path
from typing import Any

from AutoAugment.diagnostic_pipeline.common import (
    find_yolo_weight,
    maybe_device_arg,
    metrics_from_logs,
    quote_path,
    run_logged_command,
    write_json,
    write_markdown,
)


def run_baseline_training(
    *,
    data_yaml: str | Path,
    output_dir: str | Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int = 0,
    device: str | int | None = None,
    dataset_root: str | Path | None = None,
    dry_run: bool = False,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Train and validate the baseline YOLO model with full command auditing."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(data_yaml)
    project = output / "train_runs"
    train_command = (
        f"yolo detect train model={model} data={quote_path(data_yaml)} epochs={epochs} "
        f"imgsz={imgsz} batch={batch} workers={workers}{maybe_device_arg(device)} "
        f"project={quote_path(project)} name=baseline exist_ok=True"
    )
    train_result = run_logged_command(
        train_command,
        output_dir=output,
        prefix="train",
        dry_run=dry_run,
        timeout=timeout,
    )
    if train_result.returncode not in {0, None}:
        raise RuntimeError(f"Baseline YOLO train failed; see {train_result.stderr_log}")

    best_pt = find_yolo_weight(project / "baseline", "best.pt")
    last_pt = find_yolo_weight(project / "baseline", "last.pt")
    if dry_run:
        best_pt = output / "train_runs" / "baseline" / "weights" / "best.pt"
        last_pt = output / "train_runs" / "baseline" / "weights" / "last.pt"
    elif best_pt is None:
        raise FileNotFoundError(f"Could not locate baseline best.pt under {project}")

    val_command = (
        f"yolo detect val model={quote_path(best_pt)} data={quote_path(data_yaml)} "
        f"imgsz={imgsz} batch={batch} workers={workers}{maybe_device_arg(device)} "
        f"project={quote_path(output / 'val_runs')} name=baseline_val exist_ok=True"
    )
    val_result = run_logged_command(
        val_command,
        output_dir=output,
        prefix="val",
        dry_run=dry_run,
        timeout=timeout,
    )
    if val_result.returncode not in {0, None}:
        raise RuntimeError(f"Baseline YOLO val failed; see {val_result.stderr_log}")

    metrics = {} if dry_run else metrics_from_logs(val_result.stdout_log, val_result.stderr_log)
    record = {
        "stage": "baseline_training",
        "status": "planned" if dry_run else "completed",
        "dry_run": dry_run,
        "dataset_root": str(Path(dataset_root).resolve()) if dataset_root is not None else None,
        "data_yaml": str(data_yaml.resolve()),
        "model": model,
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "device": None if device is None else str(device),
        "best_pt": str(best_pt),
        "last_pt": str(last_pt) if last_pt is not None else None,
        "train_command": train_result.command,
        "val_command": val_result.command,
        "train_stdout_log": str(train_result.stdout_log),
        "train_stderr_log": str(train_result.stderr_log),
        "val_stdout_log": str(val_result.stdout_log),
        "val_stderr_log": str(val_result.stderr_log),
        "metrics": metrics,
    }
    write_json(output / "baseline_metrics.json", record)
    write_markdown(
        output / "baseline_summary.md",
        [
            "# Baseline YOLO Training",
            "",
            f"- Status: {record['status']}",
            f"- Data YAML: {record['data_yaml']}",
            f"- Epochs: {epochs}",
            f"- Workers: {workers}",
            f"- Best weights: {record['best_pt']}",
        ],
    )
    return record
