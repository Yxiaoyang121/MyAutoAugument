from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_yolo_online_aug import count_split_images, metrics_to_dict  # noqa: E402


PYTHON_EXE = Path(r"D:\Anaconda\envs\pytorch\python.exe")
DEFAULT_DATA = PROJECT_ROOT / "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "outputs/experiments"
DEFAULT_RUN_ID = "clean_native_yolo_default_seed42_50ep"
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")
AUG_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "copy_paste_mode",
    "auto_augment",
    "erasing",
    "close_mosaic",
)


def main() -> None:
    args = parse_args()
    output_dir = (Path(args.project) / args.run_id).resolve()
    configure_environment(output_dir)
    prepare_dirs(output_dir)
    preload_ultralytics_font(output_dir)

    payload = run_clean_native(args, output_dir)
    write_outputs(payload)
    if not args.skip_doc_update:
        update_state_docs(payload)
        export_project_snapshot()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    if not payload["train"]["success"]:
        raise RuntimeError(f"clean native training failed: {payload['train']['error']}")
    if not payload["val"]["success"]:
        raise RuntimeError(f"clean native validation failed: {payload['val']['error']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a clean native Ultralytics YOLO default training reference.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default=str(DEFAULT_PROJECT))
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--skip-doc-update", action="store_true")
    return parser.parse_args()


def configure_environment(output_dir: Path) -> None:
    scripts_dir = PYTHON_EXE.parent / "Scripts"
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    yolo_config = output_dir / "configs" / "ultralytics"
    yolo_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config.resolve())


def prepare_dirs(output_dir: Path) -> None:
    for subdir in ["configs", "logs", "reports"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def preload_ultralytics_font(output_dir: Path) -> None:
    source = PROJECT_ROOT / "outputs/Ultralytics/Arial.Unicode.ttf"
    if not source.exists():
        return
    target = output_dir / "configs/ultralytics/Ultralytics/Arial.Unicode.ttf"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)


def run_clean_native(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    data_path = Path(args.data).resolve()
    train_kwargs = {
        "data": str(data_path),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "workers": int(args.workers),
        "device": str(args.device),
        "seed": int(args.seed),
        "project": str(output_dir),
        "name": "train",
        "exist_ok": True,
        "plots": False,
    }
    train_command = build_train_command(args, output_dir)
    write_text(output_dir / "configs/train_command.txt", train_command)
    write_text(output_dir / "logs/train.command.txt", train_command)

    train_stdout = output_dir / "logs/train.stdout.log"
    train_stderr = output_dir / "logs/train.stderr.log"
    train_success = False
    train_error: str | None = None
    train_result_type: str | None = None
    train_start = time.time()
    try:
        with train_stdout.open("w", encoding="utf-8", errors="replace") as stdout:
            with train_stderr.open("w", encoding="utf-8", errors="replace") as stderr:
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = YOLO(args.model).train(**train_kwargs)
        train_result_type = type(result).__name__
        train_success = True
    except Exception as exc:
        train_error = f"{type(exc).__name__}: {exc}"
    train_end = time.time()

    best_pt = output_dir / "train/weights/best.pt"
    last_pt = output_dir / "train/weights/last.pt"
    weights_for_val = best_pt if best_pt.exists() else last_pt
    val_metrics: dict[str, Any] = {}
    val_success = False
    val_error: str | None = None
    val_command = ""
    val_start = None
    val_end = None
    if train_success and weights_for_val.exists():
        val_command = build_val_command(args, output_dir, weights_for_val)
        write_text(output_dir / "configs/val_command.txt", val_command)
        write_text(output_dir / "logs/val.command.txt", val_command)
        val_stdout = output_dir / "logs/val.stdout.log"
        val_stderr = output_dir / "logs/val.stderr.log"
        val_start = time.time()
        try:
            with val_stdout.open("w", encoding="utf-8", errors="replace") as stdout:
                with val_stderr.open("w", encoding="utf-8", errors="replace") as stderr:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        val_result = YOLO(str(weights_for_val)).val(
                            data=str(data_path),
                            imgsz=int(args.imgsz),
                            batch=int(args.batch),
                            workers=int(args.workers),
                            device=str(args.device),
                            project=str(output_dir),
                            name="val",
                            exist_ok=True,
                            plots=False,
                        )
            val_metrics = metrics_to_dict(val_result)
            if val_metrics.get("images") is None:
                val_metrics["images"] = count_split_images(data_path, "val")
            val_success = True
        except Exception as exc:
            val_error = f"{type(exc).__name__}: {exc}"
        val_end = time.time()
    elif train_success:
        val_error = "no best.pt or last.pt found after training"
    else:
        val_error = "validation skipped because training failed"

    args_yaml = read_yaml(output_dir / "train/args.yaml")
    train_images = count_split_images(data_path, "train")
    metrics = compact_metrics(val_metrics)
    return {
        "run_id": args.run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "dataset": str(data_path),
        "model": args.model,
        "mode": "clean_native_ultralytics_yolo_default",
        "pure_native_yolo_train": True,
        "custom_trainer_used": False,
        "online_aug_detection_trainer_used": False,
        "feedback_callback_registered": False,
        "industrial_augmentation_registered": False,
        "fixed_augmented_dataset_generated": fixed_augmented_dataset_generated(output_dir),
        "train_image_count": train_images,
        "train_image_count_matches_original": train_images == 2301,
        "train_settings": {
            "epochs": int(args.epochs),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "workers": int(args.workers),
            "device": str(args.device),
            "seed": int(args.seed),
            "yolo_default_augmentations_enabled": True,
            "actual_augmentation_params_from_args_yaml": {key: args_yaml.get(key) for key in AUG_KEYS},
        },
        "commands": {
            "train_command": train_command,
            "val_command": val_command,
        },
        "time": {
            "train_start": train_start,
            "train_end": train_end,
            "train_wall_seconds": train_end - train_start,
            "val_start": val_start,
            "val_end": val_end,
            "val_wall_seconds": None if val_start is None or val_end is None else val_end - val_start,
        },
        "artifacts": {
            "args_yaml": str((output_dir / "train/args.yaml").resolve()),
            "results_csv": str((output_dir / "train/results.csv").resolve()),
            "best_pt": str(best_pt.resolve()),
            "last_pt": str(last_pt.resolve()),
            "train_log": str(train_stdout.resolve()),
            "val_log": str((output_dir / "logs/val.stdout.log").resolve()),
        },
        "train": {
            "success": train_success,
            "error": train_error,
            "result_type": train_result_type,
        },
        "val": {
            "success": val_success,
            "error": val_error,
            "metrics": val_metrics,
        },
        "metrics": metrics,
        "rows": [
            {
                "key": "clean_native_yolo_default",
                "label": "Clean native YOLO default seed42",
                "run_dir": str(output_dir),
                "metrics": metrics,
            }
        ],
        "close_mosaic": close_mosaic_summary(args_yaml, args.epochs, train_stdout),
        "summary": {
            "train_success": train_success,
            "val_success": val_success,
            "train_image_count": train_images,
            "fixed_augmented_dataset_generated": fixed_augmented_dataset_generated(output_dir),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "map50": metrics.get("map50"),
            "map50_95": metrics.get("map50_95"),
            "report": str((output_dir / "reports/clean_native_yolo_default_report.md").resolve()),
            "metrics_json": str((output_dir / "reports/clean_native_yolo_default_metrics.json").resolve()),
        },
    }


def build_train_command(args: argparse.Namespace, output_dir: Path) -> str:
    parts = [
        "YOLO.train",
        f"model={args.model}",
        f"data={Path(args.data).resolve()}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"device={args.device}",
        f"seed={args.seed}",
        f"project={output_dir}",
        "name=train",
        "exist_ok=True",
        "plots=False",
        "trainer=native_ultralytics_default",
    ]
    return subprocess.list2cmdline([str(part) for part in parts])


def build_val_command(args: argparse.Namespace, output_dir: Path, weights: Path) -> str:
    parts = [
        "YOLO.val",
        f"model={weights.resolve()}",
        f"data={Path(args.data).resolve()}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"device={args.device}",
        f"project={output_dir}",
        "name=val",
        "exist_ok=True",
        "plots=False",
    ]
    return subprocess.list2cmdline([str(part) for part in parts])


def write_outputs(payload: dict[str, Any]) -> None:
    output_dir = Path(payload["output_dir"])
    write_json(output_dir / "reports/clean_native_yolo_default_metrics.json", payload)
    write_text(output_dir / "reports/clean_native_yolo_default_report.md", build_report(payload))


def build_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    aug = payload["train_settings"]["actual_augmentation_params_from_args_yaml"]
    lines = [
        "# Clean Native YOLO Default 50 Epoch Report",
        "",
        "## Integrity",
        "",
        f"- Pure native YOLO.train: `{str(payload['pure_native_yolo_train']).lower()}`",
        f"- Custom trainer used: `{str(payload['custom_trainer_used']).lower()}`",
        f"- OnlineAugDetectionTrainer used: `{str(payload['online_aug_detection_trainer_used']).lower()}`",
        f"- Feedback callback registered: `{str(payload['feedback_callback_registered']).lower()}`",
        f"- Industrial augmentation registered: `{str(payload['industrial_augmentation_registered']).lower()}`",
        f"- Train image count: `{payload['train_image_count']}`",
        f"- Fixed augmented dataset generated: `{str(payload['fixed_augmented_dataset_generated']).lower()}`",
        f"- close_mosaic arg: `{payload['close_mosaic']['arg']}`",
        f"- close_mosaic official schedule expected: `{str(payload['close_mosaic']['official_schedule_expected']).lower()}`",
        f"- close_mosaic expected start epoch: `{payload['close_mosaic']['expected_start_epoch']}`",
        f"- close_mosaic log triggered: `{str(payload['close_mosaic']['log_triggered']).lower()}`",
        "",
        "## Metrics",
        "",
        f"- Precision: `{fmt(metrics.get('precision'))}`",
        f"- Recall: `{fmt(metrics.get('recall'))}`",
        f"- mAP50: `{fmt(metrics.get('map50'))}`",
        f"- mAP50-95: `{fmt(metrics.get('map50_95'))}`",
        "",
        "## YOLO Default Augmentation Args",
        "",
        "| arg | value |",
        "|---|---:|",
    ]
    for key in AUG_KEYS:
        lines.append(f"| `{key}` | `{aug.get(key)}` |")
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "Training:",
            "",
            f"```text\n{payload['commands']['train_command']}\n```",
            "",
            "Validation:",
            "",
            f"```text\n{payload['commands']['val_command']}\n```",
        ]
    )
    return "\n".join(lines) + "\n"


def update_state_docs(payload: dict[str, Any]) -> None:
    content = "\n".join(
        [
            "## Clean Native YOLO Default Reference",
            "",
            "- The old YOLO default reference is no longer treated as the final baseline because parity audit found it used `OnlineAugDetectionTrainer` with an empty passthrough policy.",
            "- New baseline: `outputs/experiments/clean_native_yolo_default_seed42_50ep/`.",
            "- Training mode: pure native Ultralytics `YOLO.train(**same_args)`.",
            "- Custom trainer / callback / dataset / transform / industrial augmentation: `false`.",
            f"- Train image count: `{payload['train_image_count']}`",
            f"- Precision/Recall/mAP50/mAP50-95: `{fmt(payload['metrics'].get('precision'))}/{fmt(payload['metrics'].get('recall'))}/{fmt(payload['metrics'].get('map50'))}/{fmt(payload['metrics'].get('map50_95'))}`",
            f"- close_mosaic official schedule expected: `{str(payload['close_mosaic']['official_schedule_expected']).lower()}`",
            f"- close_mosaic expected start epoch: `{payload['close_mosaic']['expected_start_epoch']}`",
            "- Future feedback experiments should compare only against this clean native reference.",
            "- Report: `outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_report.md`",
            "- Metrics JSON: `outputs/experiments/clean_native_yolo_default_seed42_50ep/reports/clean_native_yolo_default_metrics.json`",
        ]
    )
    for name in ["PROJECT_STATE.md", "CODEX_HANDOFF.md", "EXPERIMENT_LOG.md"]:
        update_marked_section(PROJECT_ROOT / name, "CLEAN_NATIVE_YOLO_DEFAULT_REFERENCE", content)


def update_marked_section(path: Path, marker: str, content: str) -> None:
    start = f"<!-- {marker}_START -->"
    end = f"<!-- {marker}_END -->"
    block = f"{start}\n{content.rstrip()}\n{end}"
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in original and end in original:
        before = original.split(start, 1)[0].rstrip()
        after = original.split(end, 1)[1].lstrip()
        text = f"{before}\n{block}\n{after}".rstrip() + "\n"
    else:
        text = original.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def export_project_snapshot() -> None:
    subprocess.run([str(PYTHON_EXE), str(PROJECT_ROOT / "scripts/export_project_snapshot.py")], cwd=str(PROJECT_ROOT), check=False)


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in ["images", "instances", *METRIC_KEYS]}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def fixed_augmented_dataset_generated(output_dir: Path) -> bool:
    return any(
        path.exists()
        for path in [
            output_dir / "dataset/final_dataset/images",
            output_dir / "dataset_builder/final_dataset/images",
        ]
    )


def contains_text(path: Path, text: str) -> bool:
    return path.exists() and text in path.read_text(encoding="utf-8-sig", errors="replace")


def close_mosaic_summary(args_yaml: dict[str, Any], epochs: int, train_stdout: Path) -> dict[str, Any]:
    raw_arg = args_yaml.get("close_mosaic")
    try:
        close_arg = int(raw_arg)
    except (TypeError, ValueError):
        close_arg = None
    scheduled = close_arg is not None and close_arg > 0 and epochs > close_arg
    return {
        "arg": raw_arg,
        "official_schedule_expected": scheduled,
        "official_schedule_last_n_epochs": close_arg if scheduled else 0,
        "expected_start_epoch": epochs - close_arg + 1 if scheduled and close_arg is not None else None,
        "log_triggered": contains_text(train_stdout, "Closing dataloader mosaic"),
    }


def fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
