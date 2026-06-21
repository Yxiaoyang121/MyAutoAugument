from __future__ import annotations

from pathlib import Path
from typing import Any

from gui.services.result_loader import ResultLoader


class ResultRepository:
    """Single read entry point for experiment result files used by controllers."""

    def __init__(self, project_root: Path) -> None:
        self.loader = ResultLoader(project_root)

    def load(self, output_dir: str | Path) -> dict[str, Any]:
        return self.loader.load(output_dir)

    def has_results(self, output_dir: str | Path) -> bool:
        return self.loader.has_result_files(output_dir)

    def load_mock(self) -> dict[str, Any]:
        return self.loader.load_mock()
