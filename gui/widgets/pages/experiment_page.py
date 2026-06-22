from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from gui.controllers import ExperimentController
from gui.models import ExperimentConfig
from gui.models.experiment_config import (
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_PRESERVE_WEAK,
    TRAINING_MODE_YOLO_DEFAULT,
)
from gui.services.app_state import AppState
from gui.widgets.experiment_config_panel import ExperimentConfigPanel
from gui.widgets.experiment_status_panel import ExperimentStatusPanel


class ExperimentPage(QWidget):
    output_dir_changed = Signal(str)

    def __init__(self, project_root: Path, app_state: AppState | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.app_state = app_state
        self.controller = ExperimentController(self.project_root, app_state)
        self.current_policy: dict = {"name": "gui_policy", "operations": []}
        self._build_ui()
        self._connect_controller()
        self.config_panel.set_output_dir(str(self.project_root / "outputs"))

    def _build_ui(self) -> None:
        self.setObjectName("experimentPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.config_panel = ExperimentConfigPanel()
        self.status_panel = ExperimentStatusPanel()
        grid.addWidget(self.config_panel, 0, 0)
        grid.addWidget(self.status_panel, 0, 1)
        grid.setColumnStretch(0, 6)
        grid.setColumnStretch(1, 11)
        layout.addLayout(grid, 1)

        self.config_panel.start_requested.connect(self._start_experiment)
        self.config_panel.stop_requested.connect(self._stop_experiment)
        self.config_panel.open_output_requested.connect(self._open_output_dir)
        self.config_panel.output_dir_changed.connect(self.output_dir_changed.emit)

    def set_policy(self, policy: dict) -> None:
        self.current_policy = policy or {"name": "gui_policy", "operations": []}

    def set_dataset_path(self, path: str, inspection: dict | None = None) -> None:
        dataset = Path(path)
        if dataset.is_dir() and (dataset / "data.yaml").exists():
            self.config_panel.set_dataset_path(str(dataset / "data.yaml"))
        else:
            self.config_panel.set_dataset_path(path)

    def _config(self) -> ExperimentConfig:
        values = self.config_panel.values()
        metric = str(values["metric"]).lower().replace("-", "_")
        return ExperimentConfig.from_gui(
            dataset_path=values["dataset_path"],
            policy=self.current_policy,
            params={
                "dataset_path": values["dataset_path"],
                "output_dir": values["output_dir"],
                "epochs": values["epochs"],
                "imgsz": values["imgsz"],
                "workers": values["workers"],
                "batch": values["batch"],
                "seed": values["seed"],
                "device": values["device"],
                "model": values["model"],
                "run_mode": values["run_mode"],
                "metric": metric,
                "yolo_aug_params": values["yolo_aug_params"],
                "proxy_prefilter": False,
                "real_yolo_validation": True,
                "smoke_test": False,
                "adaptive_policy": False,
                "diagnose_trial_errors": False,
            },
        )

    def _start_experiment(self) -> None:
        config = self._config()
        self.status_panel.clear_log()
        self.status_panel.set_process_running(config.trials)
        self.config_panel.set_running(True)
        mode_text = {
            TRAINING_MODE_CUSTOM: "custom YOLO augmentation training",
            TRAINING_MODE_YOLO_DEFAULT: "YOLO default training",
            TRAINING_MODE_PRESERVE_WEAK: "Preserve-Weak Image-only CATF",
        }.get(config.run_mode, config.run_mode)
        self.status_panel.append_log(f"[GUI] Start requested: {mode_text}.")
        if not self.controller.start(config):
            self.config_panel.set_running(False)

    def _stop_experiment(self) -> None:
        self.status_panel.set_stopped()
        self.status_panel.append_log("[GUI] Stop requested by user.")
        self.controller.stop()

    def _open_output_dir(self) -> None:
        output = Path(self.config_panel.values()["output_dir"])
        if not output.is_absolute():
            output = self.project_root / output
        output.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

    def _connect_controller(self) -> None:
        self.controller.log_received.connect(self.status_panel.append_log)
        self.controller.started.connect(self._on_backend_started)
        self.controller.finished.connect(self._on_backend_finished)
        self.controller.error.connect(self._on_backend_error)
        self.controller.progress_changed.connect(self._on_progress_changed)
        self.controller.trials_updated.connect(self._on_trials_updated)

    def _on_backend_started(self, output_dir: str) -> None:
        self.output_dir_changed.emit(output_dir)
        self.status_panel.append_log(f"[GUI] Output dir: {output_dir}")

    def _on_backend_finished(self, exit_code: int, status: str, output_dir: str) -> None:
        self.config_panel.set_running(False)
        self.output_dir_changed.emit(output_dir)
        if status == "user stopped":
            self.status_panel.set_stopped()
        else:
            self.status_panel.set_finished(exit_code == 0)
        self.status_panel.append_log(f"[GUI] Backend finished: exit_code={exit_code}, status={status}")

    def _on_backend_error(self, message: str) -> None:
        self.config_panel.set_running(False)
        self.status_panel.set_finished(False)
        self.status_panel.append_log(f"[GUI] Backend error: {message}")

    def _on_progress_changed(self, current: int, total: int, stage: str) -> None:
        self.status_panel.set_trial_status(current, total, stage)

    def _on_trials_updated(self, records: list) -> None:
        self.status_panel.set_trial_records(records[-6:])
