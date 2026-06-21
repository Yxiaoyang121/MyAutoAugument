from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject

from gui.adapters import DatasetAdapter
from gui.services.app_state import AppState


class DatasetController(QObject):
    def __init__(self, project_root: Path, app_state: AppState | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.app_state = app_state
        self.adapter = DatasetAdapter()

    def inspect(self, dataset_path: str | Path, progress_callback: Callable[[int, str], None] | None = None) -> dict[str, Any]:
        info = self.adapter.inspect_yolo_dataset(dataset_path, progress_callback=progress_callback).to_dict()
        if self.app_state is not None:
            self.app_state.set_dataset(str(info.get("root", dataset_path)), info)
        return info
