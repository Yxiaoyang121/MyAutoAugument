from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from gui.models import ExperimentConfig
from gui.services import LocalExperimentRunner, ResultRepository
from gui.services.app_state import AppState


class ExperimentController(QObject):
    log_received = Signal(str)
    started = Signal(str)
    finished = Signal(int, str, str)
    error = Signal(str)
    state_changed = Signal(str)
    trial_event = Signal(dict)
    progress_changed = Signal(int, int, str)
    trials_updated = Signal(list)

    def __init__(self, project_root: Path, app_state: AppState | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.app_state = app_state
        self.runner = LocalExperimentRunner(self.project_root)
        self.repository = ResultRepository(self.project_root)
        self.current_output_dir = ""
        self.total_tasks = 1
        self._last_record_count = 0

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1500)
        self._poll_timer.timeout.connect(self._poll_results)

        self.runner.log_received.connect(self._handle_log)
        self.runner.started.connect(self._on_started)
        self.runner.finished.connect(self._on_finished)
        self.runner.error.connect(self.error.emit)
        self.runner.state_changed.connect(self.state_changed.emit)

    def is_running(self) -> bool:
        return self.runner.is_running()

    def start(self, config: ExperimentConfig) -> bool:
        output_dir = self._resolve_run_output_dir(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir = str(output_dir)
        config_path = config.save(output_dir / "gui_experiment_config.json")
        self.current_output_dir = str(output_dir)
        self.total_tasks = 1
        self._last_record_count = 0

        if self.app_state is not None:
            self.app_state.set_output_dir(str(output_dir))
            self.app_state.set_latest_result_dir(str(output_dir))
            self.app_state.set_backend_status("运行中")

        started = self.runner.start({**config.to_dict(), "config_path": str(config_path)})
        if started:
            self._poll_timer.start()
            self.started.emit(str(output_dir))
        return started

    def stop(self) -> None:
        self.runner.stop()

    def _resolve_run_output_dir(self, configured_output: str) -> Path:
        base = Path(configured_output or "outputs")
        if not base.is_absolute():
            base = self.project_root / base
        if base.name.startswith("run_"):
            return base
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        return base / stamp

    def _handle_log(self, message: str) -> None:
        for raw_line in message.splitlines() or [message]:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("GUI_EVENT "):
                self._handle_event_line(line[len("GUI_EVENT ") :])
                continue
            self.log_received.emit(line)

    def _handle_event_line(self, payload: str) -> None:
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            self.log_received.emit(f"[GUI] 无法解析后端事件: {payload}")
            return

        event_type = str(event.get("type", ""))
        if event_type == "phase":
            current = int(event.get("current") or 0)
            total = int(event.get("total") or self.total_tasks or 1)
            stage = str(event.get("stage") or "运行中")
            self.progress_changed.emit(current, total, stage)
            return

        if event_type == "trial_complete":
            current = int(event.get("trial_number") or int(event.get("trial_index", 0)) + 1)
            total = int(event.get("total_trials") or current)
            score = event.get("score")
            self.log_received.emit(f"[GUI] 训练记录 {current}/{total} 完成，score={self._fmt(score)}")
            self.trial_event.emit(event)
            self.progress_changed.emit(current, total, "训练记录更新")
            self._poll_results()
            return

        if event_type == "experiment_finished":
            total = int(event.get("total_trials") or self.total_tasks or 1)
            best_score = event.get("best_score")
            self.log_received.emit(f"[GUI] 训练任务完成，total={total}，best_score={self._fmt(best_score)}")
            self.progress_changed.emit(total, total, "训练任务完成")
            self._poll_results()
            return

        self.log_received.emit(f"[GUI] 后端事件: {event_type or payload}")

    def _poll_results(self) -> None:
        if not self.current_output_dir:
            return
        if not self.repository.has_results(self.current_output_dir):
            return
        data = self.repository.load(self.current_output_dir)
        records = data.get("trials") if isinstance(data.get("trials"), list) else []
        if len(records) != self._last_record_count:
            self._last_record_count = len(records)
            self.trials_updated.emit(records)
            if self.total_tasks:
                self.progress_changed.emit(min(len(records), self.total_tasks), self.total_tasks, "结果文件更新")

    def _on_started(self) -> None:
        if self.app_state is not None:
            self.app_state.set_backend_status("运行中")

    def _on_finished(self, exit_code: int, status: str) -> None:
        self._poll_timer.stop()
        self._poll_results()
        if self.app_state is not None:
            self.app_state.set_backend_status("正常" if exit_code == 0 else "失败")
            if self.current_output_dir:
                self.app_state.set_latest_result_dir(self.current_output_dir)
        self.finished.emit(exit_code, status, self.current_output_dir)

    def _fmt(self, value) -> str:
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return "-"
