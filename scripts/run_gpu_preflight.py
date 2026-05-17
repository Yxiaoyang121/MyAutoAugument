from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_JSON = OUTPUT_DIR / "gpu_preflight_report.json"
REPORT_MD = OUTPUT_DIR / "gpu_preflight_report.md"
SMOKE_DIR = OUTPUT_DIR / "gpu_preflight_smoke"
SMOKE_DATA_YAML = OUTPUT_DIR / "tiled_dataset_smoke" / "data.yaml"
MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"


TORCH_INFO_CODE = (
    "import torch; "
    "print('torch:', torch.__version__); "
    "print('cuda_available:', torch.cuda.is_available()); "
    "print('cuda_version:', torch.version.cuda); "
    "print('device_count:', torch.cuda.device_count()); "
    "print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {REPORT_JSON}")
    print(f"Wrote {REPORT_MD}")
    print(f"torch.cuda.is_available(): {report['torch'].get('cuda_available')}")
    print(f"GPU device: {report['torch'].get('device_name')}")
    print(f"YOLO GPU smoke status: {report['yolo_gpu_smoke']['status']}")
    if report["yolo_gpu_smoke"].get("success") is False:
        sys.exit(2)


def build_report() -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    commands: dict[str, dict[str, Any]] = {}
    commands["python_version"] = run_command(
        ["python", "--version"],
        "python --version",
        env=env,
        timeout=60,
    )
    commands["torch_cuda_info"] = run_command(
        ["python", "-c", TORCH_INFO_CODE],
        (
            'python -c "import torch; print(\'torch:\', torch.__version__); '
            "print('cuda_available:', torch.cuda.is_available()); "
            "print('cuda_version:', torch.version.cuda); "
            "print('device_count:', torch.cuda.device_count()); "
            "print('device_name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')\""
        ),
        env=env,
        timeout=120,
    )
    commands["ultralytics_version"] = run_command(
        ["python", "-c", "import ultralytics; print('ultralytics:', ultralytics.__version__)"],
        'python -c "import ultralytics; print(\'ultralytics:\', ultralytics.__version__)"',
        env=env,
        timeout=120,
    )
    commands["yolo_checks"] = run_command(
        ["yolo", "checks"],
        "yolo checks",
        env=env,
        timeout=240,
    )
    commands["nvidia_smi"] = run_command(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        env=env,
        timeout=60,
    )

    torch_info = parse_colon_lines(commands["torch_cuda_info"]["stdout"])
    ultralytics_info = parse_colon_lines(commands["ultralytics_version"]["stdout"])
    cuda_available = torch_info.get("cuda_available") == "True"

    nvidia_smi = summarize_nvidia_smi(commands["nvidia_smi"])

    yolo_gpu_smoke: dict[str, Any]
    if cuda_available:
        yolo_gpu_smoke = run_yolo_gpu_smoke(env=env)
    else:
        yolo_gpu_smoke = {
            "status": "not_run_cuda_unavailable",
            "success": None,
            "reason": "torch.cuda.is_available() = False",
            "formal_training_allowed": False,
        }

    finished_at = datetime.now().isoformat(timespec="seconds")
    conclusion = build_conclusion(cuda_available, yolo_gpu_smoke, torch_info, nvidia_smi)
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "project_root": str(PROJECT_ROOT),
        "commands": commands,
        "python": {
            "version_stdout": commands["python_version"]["stdout"].strip(),
            "returncode": commands["python_version"]["returncode"],
        },
        "torch": {
            "version": torch_info.get("torch"),
            "cuda_available": cuda_available,
            "cuda_version": torch_info.get("cuda_version"),
            "device_count": parse_int(torch_info.get("device_count")),
            "device_name": torch_info.get("device_name", "NO CUDA"),
            "raw": torch_info,
        },
        "ultralytics": {
            "version": ultralytics_info.get("ultralytics"),
            "returncode": commands["ultralytics_version"]["returncode"],
        },
        "nvidia_smi": nvidia_smi,
        "yolo_checks": {
            "returncode": commands["yolo_checks"]["returncode"],
            "success": commands["yolo_checks"]["returncode"] == 0,
        },
        "yolo_gpu_smoke": yolo_gpu_smoke,
        "conclusion": conclusion,
    }


def run_yolo_gpu_smoke(env: dict[str, str]) -> dict[str, Any]:
    if not SMOKE_DATA_YAML.exists():
        return {
            "status": "not_run_missing_dataset",
            "success": None,
            "reason": f"Missing smoke data yaml: {SMOKE_DATA_YAML}",
            "formal_training_allowed": False,
        }
    if not MODEL_PATH.exists():
        return {
            "status": "not_run_missing_model",
            "success": None,
            "reason": f"Missing local model: {MODEL_PATH}",
            "formal_training_allowed": False,
        }

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    train_command = [
        "yolo",
        "detect",
        "train",
        f"model={MODEL_PATH}",
        f"data={SMOKE_DATA_YAML}",
        "imgsz=640",
        "batch=1",
        "epochs=1",
        "workers=0",
        "device=0",
        f"project={SMOKE_DIR}",
        "name=train",
        "exist_ok=True",
        "mosaic=0",
        "mixup=0",
        "copy_paste=0",
        "hsv_h=0",
        "hsv_s=0",
        "hsv_v=0",
        "degrees=0",
        "translate=0",
        "scale=0",
        "shear=0",
        "perspective=0",
        "fliplr=0",
        "flipud=0",
    ]
    command_text = command_display(train_command)
    (SMOKE_DIR / "train_command.txt").write_text(command_text + "\n", encoding="utf-8")
    completed = run_command(train_command, command_text, env=env, timeout=900)
    (SMOKE_DIR / "train_stdout.log").write_text(completed["stdout"], encoding="utf-8")
    (SMOKE_DIR / "train_stderr.log").write_text(completed["stderr"], encoding="utf-8")

    success = completed["returncode"] == 0
    status = "success" if success else "failed"
    result: dict[str, Any] = {
        "status": status,
        "success": success,
        "returncode": completed["returncode"],
        "command": command_text,
        "data_yaml": str(SMOKE_DATA_YAML),
        "model": str(MODEL_PATH),
        "output_dir": str(SMOKE_DIR),
        "train_command_txt": str(SMOKE_DIR / "train_command.txt"),
        "train_stdout_log": str(SMOKE_DIR / "train_stdout.log"),
        "train_stderr_log": str(SMOKE_DIR / "train_stderr.log"),
        "formal_training_allowed": success,
    }
    if not success:
        result["error_stack"] = completed["stderr"] or completed["stdout"]
        result["reason_judgement"] = judge_yolo_failure(completed)
    return result


def run_command(
    args: list[str],
    display: str,
    *,
    env: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            args,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        duration = round(time.monotonic() - start, 3)
        return {
            "command": display,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_seconds": duration,
            "timeout_seconds": timeout,
        }
    except FileNotFoundError as exc:
        duration = round(time.monotonic() - start, 3)
        return {
            "command": display,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
            "duration_seconds": duration,
            "timeout_seconds": timeout,
        }
    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - start, 3)
        return {
            "command": display,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"Command timed out after {timeout} seconds.",
            "duration_seconds": duration,
            "timeout_seconds": timeout,
        }


def parse_colon_lines(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def summarize_nvidia_smi(command: dict[str, Any]) -> dict[str, Any]:
    first_line = next((line.strip() for line in command["stdout"].splitlines() if line.strip()), "")
    parts = [part.strip() for part in first_line.split(",")] if first_line else []
    return {
        "returncode": command["returncode"],
        "available": command["returncode"] == 0,
        "summary": first_line or None,
        "gpu_name": parts[0] if len(parts) >= 1 else None,
        "driver_version": parts[1] if len(parts) >= 2 else None,
        "memory_total": parts[2] if len(parts) >= 3 else None,
    }


def build_conclusion(
    cuda_available: bool,
    smoke: dict[str, Any],
    torch_info: dict[str, str],
    nvidia_smi: dict[str, Any],
) -> dict[str, Any]:
    if not cuda_available:
        inference = (
            "nvidia-smi detects a GPU, but the active PyTorch build reports "
            f"{torch_info.get('torch')} and cuda_version={torch_info.get('cuda_version')}; "
            "the primary blocker is likely CPU-only PyTorch in the active Python environment."
            if nvidia_smi.get("available")
            else "nvidia-smi did not detect a usable NVIDIA GPU/driver from this shell."
        )
        return {
            "status": "gpu_unavailable",
            "formal_training_allowed": False,
            "inference": inference,
            "message": (
                "torch.cuda.is_available() = False. Current environment may be CPU-only PyTorch, "
                "missing CUDA runtime, or unavailable GPU driver. Only CPU smoke tests can run here, "
                "and they cannot be used as formal experiment results."
            ),
        }
    if smoke.get("success") is True:
        return {
            "status": "gpu_smoke_passed",
            "formal_training_allowed": True,
            "message": "CUDA is available and the minimal YOLO GPU smoke test completed successfully.",
        }
    return {
        "status": "gpu_smoke_failed",
        "formal_training_allowed": False,
        "message": (
            "CUDA is visible to PyTorch, but the minimal YOLO GPU smoke test failed. "
            "Do not start formal training until the failure is fixed."
        ),
    }


def judge_yolo_failure(command: dict[str, Any]) -> str:
    combined = f"{command.get('stdout', '')}\n{command.get('stderr', '')}".lower()
    if "out of memory" in combined or "cuda oom" in combined:
        return "Likely GPU memory exhaustion during YOLO startup."
    if "cuda" in combined and ("driver" in combined or "runtime" in combined):
        return "Likely CUDA runtime or driver compatibility issue."
    if "not compiled with cuda" in combined or "cuda is not available" in combined:
        return "Likely CPU-only PyTorch or CUDA unavailable inside the active Python environment."
    if "dataset" in combined or "data.yaml" in combined:
        return "Likely smoke dataset or data.yaml path/format issue."
    return "Cause not obvious from captured logs; inspect train_stdout.log and train_stderr.log."


def command_display(args: list[str]) -> str:
    return subprocess.list2cmdline([str(arg) for arg in args])


def render_markdown(report: dict[str, Any]) -> str:
    torch_info = report["torch"]
    smoke = report["yolo_gpu_smoke"]
    conclusion = report["conclusion"]
    nvidia_smi = report["nvidia_smi"]
    commands = report["commands"]
    lines = [
        "# GPU Preflight Report",
        "",
        f"- Started: {report['started_at']}",
        f"- Finished: {report['finished_at']}",
        f"- Project root: `{report['project_root']}`",
        "",
        "## Summary",
        "",
        f"- `torch.cuda.is_available()` = {torch_info.get('cuda_available')}",
        f"- PyTorch: {torch_info.get('version')}",
        f"- CUDA version reported by PyTorch: {torch_info.get('cuda_version')}",
        f"- CUDA device count: {torch_info.get('device_count')}",
        f"- GPU device name: {torch_info.get('device_name')}",
        f"- `nvidia-smi` GPU: {nvidia_smi.get('gpu_name')}",
        f"- `nvidia-smi` driver: {nvidia_smi.get('driver_version')}",
        f"- `nvidia-smi` memory: {nvidia_smi.get('memory_total')}",
        f"- Ultralytics: {report['ultralytics'].get('version')}",
        f"- `yolo checks` success: {report['yolo_checks'].get('success')}",
        f"- YOLO GPU smoke status: {smoke.get('status')}",
        f"- Formal training allowed: {conclusion.get('formal_training_allowed')}",
        "",
        "## Conclusion",
        "",
        conclusion.get("message", ""),
        "",
        f"Inference: {conclusion.get('inference', 'n/a')}",
        "",
    ]

    if torch_info.get("cuda_available") is False:
        lines.extend(
            [
                "Required no-GPU note:",
                "",
                "- `torch.cuda.is_available() = False`",
                "- Current environment may be CPU-only PyTorch, CUDA not installed, or driver unavailable.",
                "- Only CPU smoke tests can run here; CPU smoke results cannot be used as formal experiment results.",
                "",
            ]
        )

    lines.extend(
        [
            "## Commands",
            "",
        ]
    )
    for name in [
        "python_version",
        "torch_cuda_info",
        "ultralytics_version",
        "yolo_checks",
        "nvidia_smi",
    ]:
        command = commands[name]
        lines.extend(render_command(name, command))

    lines.extend(
        [
            "## YOLO GPU Smoke",
            "",
        ]
    )
    for key in [
        "status",
        "success",
        "returncode",
        "command",
        "data_yaml",
        "model",
        "output_dir",
        "train_command_txt",
        "train_stdout_log",
        "train_stderr_log",
        "reason",
        "reason_judgement",
    ]:
        if key in smoke:
            lines.append(f"- {key}: `{smoke[key]}`")
    lines.append("")
    if smoke.get("success") is False:
        lines.extend(
            [
                "### Captured Error Stack",
                "",
                "```text",
                str(smoke.get("error_stack", "")).rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_command(name: str, command: dict[str, Any]) -> list[str]:
    return [
        f"### {name}",
        "",
        f"- Command: `{command['command']}`",
        f"- Return code: `{command['returncode']}`",
        f"- Duration seconds: `{command['duration_seconds']}`",
        "",
        "stdout:",
        "",
        "```text",
        command["stdout"].rstrip(),
        "```",
        "",
        "stderr:",
        "",
        "```text",
        command["stderr"].rstrip(),
        "```",
        "",
    ]


if __name__ == "__main__":
    main()
