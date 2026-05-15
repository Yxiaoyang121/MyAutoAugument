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


def run_final_training(
    *,
    data_yaml: str | Path,
    output_dir: str | Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int = 0,
    device: str | int | None = None,
    dry_run: bool = False,
    skip_training: bool = False,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Train and validate the final YOLO detector on the augmented dataset."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data_yaml = Path(data_yaml)
    if dry_run or skip_training:
        payload = {
            "stage": "final_training",
            "status": "skipped" if skip_training else "planned",
            "dry_run": dry_run,
            "skipped": skip_training,
            "data_yaml": str(data_yaml.resolve()),
            "model": model,
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "workers": int(workers),
            "train_command": _train_command(output, data_yaml, model, epochs, imgsz, batch, workers, device),
            "val_command": None,
            "best_pt": None,
            "last_pt": None,
            "metrics": {},
        }
        (output / "train_command.txt").write_text(payload["train_command"] + "\n", encoding="utf-8")
        (output / "train_stdout.log").write_text("[planned] final training not executed\n", encoding="utf-8")
        (output / "train_stderr.log").write_text("", encoding="utf-8")
        write_json(output / "final_train_metrics.json", payload)
        write_json(output / "final_val_metrics.json", {"status": payload["status"], "metrics": {}})
        write_final_report(output / "final_report.md", payload)
        return payload

    train_command = _train_command(output, data_yaml, model, epochs, imgsz, batch, workers, device)
    train_result = run_logged_command(
        train_command,
        output_dir=output,
        prefix="train",
        timeout=timeout,
    )
    if train_result.returncode != 0:
        raise RuntimeError(f"Final YOLO training failed; see {train_result.stderr_log}")
    best_pt = find_yolo_weight(output / "train_runs" / "final", "best.pt")
    last_pt = find_yolo_weight(output / "train_runs" / "final", "last.pt")
    if best_pt is None:
        raise FileNotFoundError(f"Could not find final best.pt under {output / 'train_runs'}")
    val_command = (
        f"yolo detect val model={quote_path(best_pt)} data={quote_path(data_yaml)} imgsz={imgsz} "
        f"batch={batch} workers={workers}{maybe_device_arg(device)} "
        f"project={quote_path(output / 'val_runs')} name=final_val exist_ok=True"
    )
    val_result = run_logged_command(
        val_command,
        output_dir=output,
        prefix="val",
        timeout=timeout,
    )
    if val_result.returncode != 0:
        raise RuntimeError(f"Final YOLO validation failed; see {val_result.stderr_log}")
    metrics = metrics_from_logs(val_result.stdout_log, val_result.stderr_log)
    payload = {
        "stage": "final_training",
        "status": "completed",
        "dry_run": False,
        "skipped": False,
        "data_yaml": str(data_yaml.resolve()),
        "model": model,
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "workers": int(workers),
        "train_command": train_command,
        "val_command": val_command,
        "best_pt": str(best_pt),
        "last_pt": str(last_pt) if last_pt else None,
        "train_stdout_log": str(train_result.stdout_log),
        "train_stderr_log": str(train_result.stderr_log),
        "val_stdout_log": str(val_result.stdout_log),
        "val_stderr_log": str(val_result.stderr_log),
        "metrics": metrics,
    }
    write_json(output / "final_train_metrics.json", payload)
    write_json(output / "final_val_metrics.json", metrics)
    write_final_report(output / "final_report.md", payload)
    return payload


def write_final_report(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a compact final training report."""

    metrics = payload.get("metrics", {})
    write_markdown(
        path,
        [
            "# Final YOLO Training Report",
            "",
            f"- Status: {payload.get('status')}",
            f"- Data YAML: {payload.get('data_yaml')}",
            f"- Epochs: {payload.get('epochs')}",
            f"- Workers: {payload.get('workers')}",
            f"- Best weights: {payload.get('best_pt')}",
            f"- mAP50: {metrics.get('map50')}",
            f"- mAP50-95: {metrics.get('map50_95')}",
            f"- Precision: {metrics.get('precision')}",
            f"- Recall: {metrics.get('recall')}",
        ],
    )


def _train_command(
    output: Path,
    data_yaml: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str | int | None,
) -> str:
    return (
        f"yolo detect train model={model} data={quote_path(data_yaml)} epochs={epochs} "
        f"imgsz={imgsz} batch={batch} workers={workers}{maybe_device_arg(device)} "
        f"project={quote_path(output / 'train_runs')} name=final exist_ok=True"
    )
