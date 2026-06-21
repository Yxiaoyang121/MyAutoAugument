from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, Signal


@dataclass
class AppStateSnapshot:
    dataset_path: str = ""
    output_dir: str = ""
    latest_result_dir: str = ""
    current_policy: dict[str, Any] = field(default_factory=lambda: {"name": "gui_policy", "operations": []})
    backend_status: str = "正常"
    last_updated: str = ""


class AppState(QObject):
    """Shared GUI state used to keep pages synchronized without coupling pages."""

    dataset_changed = Signal(str, dict)
    output_dir_changed = Signal(str)
    result_dir_changed = Signal(str)
    policy_changed = Signal(dict)
    backend_status_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = AppStateSnapshot()

    def set_dataset(self, path: str, inspection: dict | None = None) -> None:
        self.snapshot.dataset_path = path
        self.dataset_changed.emit(path, inspection or {})

    def set_output_dir(self, path: str) -> None:
        self.snapshot.output_dir = path
        self.output_dir_changed.emit(path)

    def set_latest_result_dir(self, path: str) -> None:
        self.snapshot.latest_result_dir = path
        self.result_dir_changed.emit(path)

    def set_policy(self, policy: dict) -> None:
        self.snapshot.current_policy = policy or {"name": "gui_policy", "operations": []}
        self.policy_changed.emit(self.snapshot.current_policy)

    def set_backend_status(self, status: str) -> None:
        self.snapshot.backend_status = status
        self.backend_status_changed.emit(status)
