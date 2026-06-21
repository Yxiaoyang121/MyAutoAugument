from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AugmentationParamSchema:
    name: str
    default: Any = None
    value_type: str = "float"
    min_value: float | int | None = None
    max_value: float | int | None = None
    include_in_policy: bool = True
    note: str = ""

    def range_text(self) -> str:
        if self.min_value is None and self.max_value is None:
            return "后端动态"
        if self.min_value is None:
            return f"<= {self.max_value}"
        if self.max_value is None:
            return f">= {self.min_value}"
        return f"{self.min_value}..{self.max_value}"

    def default_for_json(self) -> Any:
        if isinstance(self.default, tuple):
            return list(self.default)
        return self.default


@dataclass(frozen=True)
class AugmentationSchema:
    name: str
    category: str
    description_zh: str
    params: tuple[AugmentationParamSchema, ...] = field(default_factory=tuple)
    changes_bboxes: bool = False
    detection_suitable: bool = True
    backend_reference: str = "AutoAugment.augmentations.ops"
    notes: str = ""

    def default_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for item in self.params:
            if not item.include_in_policy or item.default is None:
                continue
            params[item.name] = item.default_for_json()
        return params

    def params_text(self) -> str:
        if not self.params:
            return "-"
        parts = []
        for param in self.params:
            default = "动态" if param.default is None else repr(param.default)
            parts.append(f"{param.name}={default} [{param.range_text()}]")
        return "; ".join(parts)
