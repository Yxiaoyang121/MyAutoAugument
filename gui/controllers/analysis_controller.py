from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject

from gui.services import ResultRepository
from gui.services.app_state import AppState


class AnalysisController(QObject):
    def __init__(self, project_root: Path, app_state: AppState | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.app_state = app_state
        self.repository = ResultRepository(self.project_root)

    def load(self, result_dir: str | Path) -> dict[str, Any]:
        data = self.repository.load(result_dir)
        if self.app_state is not None and data.get("output_dir"):
            self.app_state.set_latest_result_dir(str(data["output_dir"]))
        return data

    def has_results(self, result_dir: str | Path) -> bool:
        return self.repository.has_results(result_dir)
