from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, Signal


class LocalExperimentRunner(QObject):
    """Run the local AutoAugment experiment backend through QProcess."""

    log_received = Signal(str)
    started = Signal()
    finished = Signal(int, str)
    error = Signal(str)
    state_changed = Signal(str)

    def __init__(self, project_root: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root)
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.started.connect(self._on_started)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        self._process.stateChanged.connect(self._on_state_changed)

    def is_running(self) -> bool:
        return self._process.state() != QProcess.ProcessState.NotRunning

    def start(self, config: dict[str, Any]) -> bool:
        if self.is_running():
            self.error.emit("已有实验正在运行。")
            return False

        script = self._resolve_backend_script(config.get("entry_script"))
        if script is None:
            message = "未找到本地后端脚本。请检查 gui/adapters/gui_experiment_launcher.py 是否存在。"
            self.log_received.emit(message)
            self.finished.emit(127, "missing backend script")
            return False

        args = [str(script), *self._build_cli_args(config, script)]
        self._process.setWorkingDirectory(str(self.project_root))
        self.log_received.emit(f"[GUI] Launch: {sys.executable} {' '.join(args)}")
        self._process.start(sys.executable, args)
        return True

    def stop(self) -> None:
        if not self.is_running():
            self.log_received.emit("[GUI] 当前没有正在运行的实验。")
            return

        pid = int(self._process.processId())
        self.log_received.emit("[GUI] 正在停止实验进程...")
        if os.name == "nt" and pid > 0:
            killer = QProcess(self)
            killer.start("taskkill", ["/PID", str(pid), "/T", "/F"])
            killer.waitForFinished(3000)
        else:
            self._process.terminate()
        if not self._process.waitForFinished(3000):
            self.log_received.emit("[GUI] 停止超时，强制结束进程。")
            self._process.kill()

    def _resolve_backend_script(self, configured: Any) -> Path | None:
        candidates: list[Path] = []
        if configured:
            candidates.append(Path(str(configured)).expanduser())
        env_script = os.environ.get("AUTOAUGMENT_EXPERIMENT_SCRIPT")
        if env_script:
            candidates.append(Path(env_script).expanduser())
        candidates.extend(
            [
                self.project_root / "gui" / "adapters" / "gui_experiment_launcher.py",
                self.project_root / "examples" / "run_policy_search.py",
                self.project_root / "scripts" / "run_experiment.py",
                self.project_root / "scripts" / "run_closed_loop_experiment.py",
                self.project_root / "src" / "run_experiment.py",
                self.project_root / "src" / "experiment.py",
            ]
        )
        for candidate in candidates:
            path = candidate if candidate.is_absolute() else self.project_root / candidate
            if path.exists() and path.is_file():
                return path.resolve()
        return None

    def _build_cli_args(self, config: dict[str, Any], script: Path) -> list[str]:
        if script.name == "gui_experiment_launcher.py" and config.get("config_path"):
            return ["--config", str(config["config_path"])]
        mapping = {
            "dataset_path": "--dataset",
            "output_dir": "--output",
            "trials": "--trials",
            "epochs": "--epochs",
            "imgsz": "--imgsz",
            "workers": "--workers",
            "batch": "--batch",
            "seed": "--seed",
            "samples": "--samples",
        }
        args: list[str] = []
        for key, flag in mapping.items():
            value = config.get(key)
            if value is None or value == "":
                continue
            args.extend([flag, str(value)])
        if config.get("proxy_prefilter"):
            args.append("--proxy-prefilter")
        args.extend(["--evaluator", "train_yolo" if config.get("real_yolo_validation") else "proxy"])
        return args

    def _read_stdout(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log_received.emit(data.rstrip())

    def _read_stderr(self) -> None:
        data = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        if data:
            self.log_received.emit(data.rstrip())

    def _on_started(self) -> None:
        self.started.emit()
        self.log_received.emit("[GUI] 实验进程已启动。")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        status = "正常退出" if exit_status == QProcess.ExitStatus.NormalExit else "异常退出"
        self.log_received.emit(f"[GUI] 实验结束，退出码 {exit_code}（{status}）。")
        self.finished.emit(exit_code, status)

    def _on_error(self, process_error: QProcess.ProcessError) -> None:
        message = f"QProcess 错误: {process_error.name}"
        self.log_received.emit(message)
        self.error.emit(message)

    def _on_state_changed(self, state: QProcess.ProcessState) -> None:
        self.state_changed.emit(state.name)
