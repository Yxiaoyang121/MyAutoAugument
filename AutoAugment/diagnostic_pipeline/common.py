from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.search import parse_yolo_metrics


@dataclass(frozen=True)
class CommandResult:
    """Audited result of an external command."""

    command: str
    returncode: int | None
    stdout_log: Path
    stderr_log: Path
    command_file: Path
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "stdout_log": str(self.stdout_log),
            "stderr_log": str(self.stderr_log),
            "command_file": str(self.command_file),
            "dry_run": self.dry_run,
        }


def write_json(path: str | Path, data: Any) -> Path:
    """Write stable UTF-8 JSON and return the path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    """Convert common runtime values to JSON-serializable structures."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def quote_path(path: str | Path) -> str:
    """Quote a path for the YOLO CLI command string."""

    text = Path(path).resolve().as_posix() if not isinstance(path, str) else path
    return f'"{text}"' if any(char.isspace() for char in text) else text


def maybe_device_arg(device: str | int | None) -> str:
    """Return the YOLO CLI device argument or an empty string."""

    if device is None or str(device).strip() == "":
        return ""
    return f" device={device}"


def run_logged_command(
    command: str,
    *,
    output_dir: str | Path,
    prefix: str,
    cwd: str | Path | None = None,
    dry_run: bool = False,
    timeout: int | None = None,
) -> CommandResult:
    """Run a command with stdout, stderr, and command text written to files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command_file = output / f"{prefix}_command.txt"
    stdout_log = output / f"{prefix}_stdout.log"
    stderr_log = output / f"{prefix}_stderr.log"
    command_file.write_text(command + "\n", encoding="utf-8")
    if dry_run:
        stdout_log.write_text("[dry-run] command was not executed\n", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return CommandResult(command, None, stdout_log, stderr_log, command_file, True)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )
    stdout_log.write_text(completed.stdout, encoding="utf-8", errors="replace")
    stderr_log.write_text(completed.stderr, encoding="utf-8", errors="replace")
    return CommandResult(command, completed.returncode, stdout_log, stderr_log, command_file, False)


def parse_yolo_metrics_extended(text: str) -> dict[str, float]:
    """Parse common Ultralytics validation metrics from stdout/stderr text."""

    metrics = dict(parse_yolo_metrics(text))
    lines = [_strip_ansi(line).strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        tokens = line.split()
        if not tokens or tokens[0].lower() != "all":
            continue
        if not _near_yolo_header(lines, index):
            continue
        numeric = [_float_or_none(token) for token in tokens[1:]]
        values = [value for value in numeric if value is not None]
        if len(values) >= 6:
            metrics.setdefault("precision", values[-4])
            metrics.setdefault("recall", values[-3])
            metrics.setdefault("map50", values[-2])
            metrics.setdefault("map50_95", values[-1])
            break
    return metrics


def metrics_from_logs(stdout_log: str | Path, stderr_log: str | Path) -> dict[str, float]:
    """Parse metrics from two command log files."""

    stdout = Path(stdout_log).read_text(encoding="utf-8", errors="replace") if Path(stdout_log).exists() else ""
    stderr = Path(stderr_log).read_text(encoding="utf-8", errors="replace") if Path(stderr_log).exists() else ""
    return parse_yolo_metrics_extended(f"{stdout}\n{stderr}")


def find_yolo_weight(run_dir: str | Path, name: str) -> Path | None:
    """Find the newest YOLO weight file with a given filename under a run directory."""

    root = Path(run_dir)
    if not root.exists():
        return None
    candidates = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].resolve() if candidates else None


def write_markdown(path: str | Path, lines: list[str]) -> Path:
    """Write Markdown lines with a trailing newline."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def _near_yolo_header(lines: list[str], index: int) -> bool:
    start = max(0, index - 5)
    for candidate in lines[start:index]:
        normalized = re.sub(r"[^a-zA-Z0-9]", "", candidate).lower()
        if "map50" in normalized and ("precision" in normalized or "boxp" in normalized or "class" in normalized):
            return True
    return False


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
