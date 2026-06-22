from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.models.experiment_config import (
    ExperimentConfig,
    PRESERVE_WEAK_ALGORITHM_LOCK,
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_PRESERVE_WEAK,
    TRAINING_MODE_YOLO_DEFAULT,
)


PRESERVE_WEAK_FLAGS = [
    "--image-only-mainline",
    "--preserve-original-enabled",
    "--weak-image-aug-enabled",
    "--weak-only-for-moderate-risk",
    "--disable-sampler-only",
    "--causal-probe-mode",
    "--precision-aware-accept-gate",
    "--class-aware-feedback",
    "--roi-aware-aug",
    "--sample-aware-routing",
    "--feedback-enabled",
    "--industrial-aug-enabled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a GUI configured training task.")
    parser.add_argument("--config", required=True, help="Path to gui_experiment_config.json.")
    parser.add_argument("--dry-run", action="store_true", help="Write command files but do not execute.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    output = _resolve_output_dir(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output)
    config.save(output / "gui_experiment_config.json")

    command = build_training_command(config, output)
    write_launch_artifacts(config, output, command)

    print(f"[GUI] Training mode: {config.training_profile().get('display_name', config.run_mode)}", flush=True)
    print(f"[GUI] Command: {subprocess.list2cmdline(command)}", flush=True)
    print(
        "GUI_EVENT "
        + json.dumps(
            {"type": "phase", "current": 0, "total": 1, "stage": "启动训练任务"},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    if args.dry_run:
        print("[GUI] Dry run complete. Command was not executed.", flush=True)
        return

    exit_code = run_command_streaming(command)
    if exit_code != 0:
        raise SystemExit(exit_code)
    write_success_summary(config, output)
    print(
        "GUI_EVENT "
        + json.dumps(
            {
                "type": "experiment_finished",
                "total_trials": 1,
                "best_trial": 0,
                "best_score": None,
                "output_dir": str(output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def build_training_command(config: ExperimentConfig, output_dir: Path) -> list[str]:
    mode = config.run_mode
    if mode == TRAINING_MODE_CUSTOM:
        return build_yolo_train_command(config, output_dir, include_aug=True)
    if mode == TRAINING_MODE_YOLO_DEFAULT:
        return build_yolo_train_command(config, output_dir, include_aug=False)
    if mode == TRAINING_MODE_PRESERVE_WEAK:
        return build_preserve_weak_command(config, output_dir)
    raise ValueError(f"Unsupported training mode: {mode}")


def build_yolo_train_command(config: ExperimentConfig, output_dir: Path, *, include_aug: bool) -> list[str]:
    command = [
        "yolo",
        "detect",
        "train",
        f"model={config.model}",
        f"data={_resolve_data_path(config.dataset_path)}",
        f"epochs={config.epochs}",
        f"imgsz={config.imgsz}",
        f"batch={config.batch}",
        f"workers={config.workers}",
        f"device={config.device}",
        f"seed={config.seed}",
        f"project={output_dir.parent}",
        f"name={output_dir.name}",
        "exist_ok=True",
    ]
    if include_aug:
        for key in sorted(config.yolo_aug_params):
            command.append(f"{key}={config.yolo_aug_params[key]}")
    return command


def build_preserve_weak_command(config: ExperimentConfig, output_dir: Path) -> list[str]:
    script = PROJECT_ROOT / "scripts" / "train_yolo_default_with_inloop_feedback.py"
    command = [
        sys.executable,
        str(script),
        "--model",
        config.model,
        "--data",
        str(_resolve_data_path(config.dataset_path)),
        "--epochs",
        str(config.epochs),
        "--imgsz",
        str(config.imgsz),
        "--batch",
        str(config.batch),
        "--workers",
        str(config.workers),
        "--device",
        str(config.device),
        "--seed",
        str(config.seed),
        "--project",
        str(output_dir.parent),
        "--run-id",
        output_dir.name,
        "--catf-version",
        "v2",
        "--attenuation-ratio",
        "0.25",
        "--feedback-interval",
        "5",
        "--feedback-start-epoch",
        "5",
    ]
    command.extend(PRESERVE_WEAK_FLAGS)
    return command


def write_launch_artifacts(config: ExperimentConfig, output_dir: Path, command: list[str]) -> None:
    command_text = subprocess.list2cmdline(command)
    (output_dir / "generated_command.txt").write_text(command_text, encoding="utf-8")
    (output_dir / "generated_command.json").write_text(
        json.dumps({"argv": command, "cmdline": command_text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "algorithm_lock.json").write_text(
        json.dumps(config.algorithm_lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps({"script": "gui/adapters/gui_experiment_launcher.py", "config": config.to_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_success_summary(config: ExperimentConfig, output_dir: Path) -> None:
    summary = {
        "mode": config.run_mode,
        "training_profile": config.training_profile(),
        "output_dir": str(output_dir.resolve()),
        "status": "completed",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "trial_record.json").write_text(json.dumps({"trials": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "diagnosis.json").write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command_streaming(command: list[str]) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip("\n"), flush=True)
    return process.wait()


def _resolve_output_dir(output_dir: str) -> Path:
    output = Path(output_dir)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    return output


def _resolve_data_path(data_path: str) -> Path:
    path = Path(data_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
