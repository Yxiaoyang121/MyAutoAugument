from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject

from gui.adapters import BackendCapabilityAdapter, PolicyAdapter
from gui.services.app_state import AppState


class PolicyController(QObject):
    def __init__(self, project_root: Path, app_state: AppState | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.app_state = app_state
        self.capabilities = BackendCapabilityAdapter()
        self.adapter = PolicyAdapter(self.capabilities)

    def load(self, path: str | Path) -> dict[str, Any]:
        policy = self.adapter.load_policy_dict(path)
        if self.app_state is not None:
            self.app_state.set_policy(policy)
        return policy

    def save(self, path: str | Path, operations: list[dict[str, Any]], *, name: str) -> Path:
        saved = self.adapter.save_policy(path, operations, name=name)
        if self.app_state is not None:
            self.app_state.set_policy(self.adapter.policy_dict(operations, name=name))
        return saved

    def policy_dict(self, operations: list[dict[str, Any]], *, name: str) -> dict[str, Any]:
        policy = self.adapter.policy_dict(operations, name=name)
        if self.app_state is not None:
            self.app_state.set_policy(policy)
        return policy
