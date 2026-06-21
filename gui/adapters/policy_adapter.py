from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from AutoAugment.augmentations import list_augmentations
from AutoAugment.policies import OperationSpec, Policy

from gui.adapters.backend_capability_adapter import BackendCapabilityAdapter


class PolicyAdapter:
    def __init__(self, capability_adapter: BackendCapabilityAdapter | None = None) -> None:
        self.capabilities = capability_adapter or BackendCapabilityAdapter()

    def default_operation(self, name: str, *, prob: float = 1.0, strength: float = 1.0) -> dict[str, Any]:
        schema = self.capabilities.schema_by_name()[name]
        return {
            "enabled": True,
            "name": schema.name,
            "prob": float(prob),
            "strength": float(strength),
            "params": schema.default_params(),
        }

    def build_policy(self, operations: list[dict[str, Any]], *, name: str = "gui_policy") -> Policy:
        registered = set(list_augmentations())
        specs: list[OperationSpec] = []
        for raw in operations:
            if not bool(raw.get("enabled", True)):
                continue
            op_name = str(raw.get("name", "")).strip().lower()
            if op_name not in registered:
                raise ValueError(f"未知后端增强方法: {op_name}")
            specs.append(
                OperationSpec(
                    name=op_name,
                    prob=float(raw.get("prob", raw.get("probability", 1.0))),
                    strength=float(raw.get("strength", 1.0)),
                    params=dict(raw.get("params", {}) or {}),
                )
            )
        return Policy(name=name, operations=specs, metadata={"source": "qt_gui"})

    def policy_dict(self, operations: list[dict[str, Any]], *, name: str = "gui_policy") -> dict[str, Any]:
        return self.build_policy(operations, name=name).to_dict()

    def policy_json(self, operations: list[dict[str, Any]], *, name: str = "gui_policy") -> str:
        return self.build_policy(operations, name=name).to_json()

    def save_policy(self, path: str | Path, operations: list[dict[str, Any]], *, name: str = "gui_policy") -> Path:
        policy = self.build_policy(operations, name=name)
        target = Path(path)
        policy.save(target)
        return target

    def load_policy_dict(self, path: str | Path) -> dict[str, Any]:
        return Policy.load(path).to_dict()

    def load_policy(self, path: str | Path) -> list[dict[str, Any]]:
        return self.operations_from_policy_dict(self.load_policy_dict(path))

    def operations_from_policy_dict(self, policy: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "enabled": True,
                "name": str(operation.get("name", "")),
                "prob": float(operation.get("prob", operation.get("probability", 1.0))),
                "strength": float(operation.get("strength", 1.0)),
                "params": dict(operation.get("params", {}) or {}),
            }
            for operation in policy.get("operations", []) or []
        ]

    def parse_params_text(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            return {}
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError("操作参数必须是 JSON 对象")
        return data
