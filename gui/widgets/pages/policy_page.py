from __future__ import annotations

import copy
import json
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QApplication, QFileDialog, QGridLayout, QMessageBox, QVBoxLayout, QWidget

from gui.adapters import BackendCapabilityAdapter, PolicyAdapter
from gui.widgets.operation_info_panel import OperationInfoPanel
from gui.widgets.operation_param_editor import OperationParamEditor
from gui.widgets.operator_library import OperatorLibrary
from gui.widgets.policy_management_panel import PolicyManagementPanel
from gui.widgets.policy_pipeline import PolicyPipeline


MOCK_POLICY = {
    "name": "gui_policy",
    "created_at": "2025-06-16 14:15:22",
    "updated_at": "2026-06-16 14:30:22",
    "description": "通用策略：光照增强 + 几何变换，提升模型泛化能力。",
    "operations": [
        {
            "name": "brightness",
            "category": "光照 / 颜色",
            "prob": 0.5,
            "strength": 0.5,
            "bbox_effect": "不改变",
            "params": {"max_delta": 0.25},
        },
        {
            "name": "contrast",
            "category": "光照 / 颜色",
            "prob": 0.5,
            "strength": 0.5,
            "bbox_effect": "不改变",
            "params": {"max_delta": 0.5},
        },
        {
            "name": "horizontal_flip",
            "category": "几何变换",
            "prob": 0.5,
            "strength": 0.5,
            "bbox_effect": "不改变",
            "params": {},
        },
        {
            "name": "rotate",
            "category": "几何变换",
            "prob": 0.5,
            "strength": 0.5,
            "bbox_effect": "会改变",
            "params": {
                "max_angle": 15.0,
                "fill_mode": "border",
                "interpolation": "bilinear",
            },
        },
    ],
}

OPERATION_ALIASES = {
    "blur": "gaussian_blur",
    "shear": "affine",
}


class PolicyPage(QWidget):
    policy_changed = Signal(dict)
    preview_requested = Signal()
    experiment_requested = Signal()

    def __init__(self, project_root: Path | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root or Path.cwd())
        self.capabilities = BackendCapabilityAdapter()
        self.schema_lookup = self.capabilities.schema_by_name()
        self.policy_adapter = PolicyAdapter(self.capabilities)
        self.policy_name = "gui_policy"
        self.default_policy_path = self.project_root / "outputs" / "gui_policies" / "current_policy.json"
        self.operations: list[dict] = copy.deepcopy(MOCK_POLICY["operations"])
        self.selected_index = 3
        self._loading = False
        self.policy_emit_timer = QTimer(self)
        self.policy_emit_timer.setSingleShot(True)
        self.policy_emit_timer.setInterval(180)
        self.policy_emit_timer.timeout.connect(self._emit_policy_changed)
        self._build_ui()
        self._sync_ui(emit=False)

    def _build_ui(self) -> None:
        self.setObjectName("policyPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        workspace = QGridLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setHorizontalSpacing(10)
        workspace.setVerticalSpacing(10)

        self.operator_library = OperatorLibrary()
        self.pipeline = PolicyPipeline()
        self.param_editor = OperationParamEditor()
        self.info_panel = OperationInfoPanel()

        workspace.addWidget(self.operator_library, 0, 0)
        workspace.addWidget(self.pipeline, 0, 1)
        workspace.addWidget(self.param_editor, 0, 2)
        workspace.addWidget(self.info_panel, 0, 3)
        workspace.setColumnStretch(0, 7)
        workspace.setColumnStretch(1, 11)
        workspace.setColumnStretch(2, 12)
        workspace.setColumnStretch(3, 8)
        layout.addLayout(workspace, 1)

        self.management = PolicyManagementPanel()
        layout.addWidget(self.management)

        self.operator_library.operator_activated.connect(self._add_operation)
        self.pipeline.selection_changed.connect(self._select_operation)
        self.pipeline.move_requested.connect(self._move_selected)
        self.pipeline.clear_requested.connect(self._clear_operations)
        self.param_editor.operation_changed.connect(self._update_selected_operation)
        self.management.save_requested.connect(self._save_default_policy)
        self.management.save_as_requested.connect(self._save_policy_as)
        self.management.import_requested.connect(self._load_policy)
        self.management.export_requested.connect(self._save_policy_as)
        self.management.copy_requested.connect(self._copy_policy_json)
        self.management.reset_requested.connect(self._reset_policy)

    def current_policy_dict(self) -> dict:
        return self.policy_adapter.policy_dict(self._backend_operations(), name=self.policy_name)

    def set_policy(self, policy: dict) -> None:
        if self._loading:
            return
        self._loading = True
        self.policy_name = str(policy.get("name") or "gui_policy")
        self.operations = [self._decorate_operation(operation) for operation in self.policy_adapter.operations_from_policy_dict(policy)]
        self.selected_index = min(max(len(self.operations) - 1, 0), 3 if len(self.operations) > 3 else 0)
        self._loading = False
        self._sync_ui(emit=False)

    def _sync_ui(self, *, emit: bool = True) -> None:
        self.pipeline.set_operations(self.operations, self.selected_index)
        operation = self.operations[self.selected_index] if self.operations else None
        self.param_editor.set_operation(operation)
        self.info_panel.set_operation(operation)
        self.management.set_summary(self.policy_name, len(self.operations))
        if emit and not self._loading:
            self._schedule_policy_changed()

    def _add_operation(self, name: str) -> None:
        self.operations.append(self._make_operation(name))
        self.selected_index = len(self.operations) - 1
        self._sync_ui()

    def _select_operation(self, index: int) -> None:
        if index < 0 or index >= len(self.operations):
            return
        self.selected_index = index
        self._sync_ui(emit=False)

    def _move_selected(self, delta: int) -> None:
        if not self.operations:
            return
        target = self.selected_index + delta
        if target < 0 or target >= len(self.operations):
            return
        self.operations[self.selected_index], self.operations[target] = self.operations[target], self.operations[self.selected_index]
        self.selected_index = target
        self._sync_ui()

    def _clear_operations(self) -> None:
        self.operations = []
        self.selected_index = 0
        self._sync_ui()

    def _update_selected_operation(self, operation: dict) -> None:
        if self.selected_index < 0 or self.selected_index >= len(self.operations):
            return
        self.operations[self.selected_index] = operation
        self.pipeline.set_operations(self.operations, self.selected_index)
        self.info_panel.set_operation(operation)
        self.management.set_summary(self.policy_name, len(self.operations))
        self._schedule_policy_changed()

    def _schedule_policy_changed(self) -> None:
        if not self._loading:
            self.policy_emit_timer.start()

    def _emit_policy_changed(self) -> None:
        if not self._loading:
            self.policy_changed.emit(self.current_policy_dict())

    def _reset_policy(self) -> None:
        self.policy_name = "gui_policy"
        self.operations = copy.deepcopy(MOCK_POLICY["operations"])
        self.selected_index = 3
        self._sync_ui()

    def _save_default_policy(self) -> None:
        try:
            self._commit_pending_edits()
            operations = self._backend_operations()
            if not operations:
                QMessageBox.warning(self, "保存失败", "当前策略为空，不能覆盖默认策略。请先从左侧添加增强操作。")
                return
            self.default_policy_path.parent.mkdir(parents=True, exist_ok=True)
            saved = self.policy_adapter.save_policy(self.default_policy_path, operations, name=self.policy_name)
            QMessageBox.information(self, "保存策略", f"已保存到：{saved}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def _save_policy_as(self) -> None:
        self._commit_pending_edits()
        path, _ = QFileDialog.getSaveFileName(self, "导出策略 JSON", str(self.default_policy_path), "JSON (*.json)")
        if not path:
            return
        try:
            operations = self._backend_operations()
            if not operations:
                QMessageBox.warning(self, "导出失败", "当前策略为空，请先从左侧添加增强操作。")
                return
            saved = self.policy_adapter.save_policy(path, operations, name=self.policy_name)
            QMessageBox.information(self, "导出策略", f"已导出到：{saved}")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _load_policy(self) -> None:
        start_dir = self.default_policy_path.parent if self.default_policy_path.parent.exists() else self.project_root
        path, _ = QFileDialog.getOpenFileName(self, "导入策略", str(start_dir), "策略文件 (*.json *.yaml *.yml)")
        if not path:
            return
        try:
            self.set_policy(self.policy_adapter.load_policy_dict(path))
            self._sync_ui()
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))

    def _copy_policy_json(self) -> None:
        self._commit_pending_edits()
        QApplication.clipboard().setText(json.dumps(self.current_policy_dict(), ensure_ascii=False, indent=2))

    def _commit_pending_edits(self) -> None:
        self.param_editor.commit_pending_change()
        if self.policy_emit_timer.isActive():
            self.policy_emit_timer.stop()
            self._emit_policy_changed()

    def _backend_operations(self) -> list[dict]:
        operations: list[dict] = []
        for operation in self.operations:
            name = str(operation.get("name", ""))
            backend_name = self._backend_name(name)
            if backend_name not in self.schema_lookup:
                continue
            params = dict(operation.get("params", {}) or {})
            if not params:
                params = self.schema_lookup[backend_name].default_params()
            operations.append(
                {
                    "enabled": True,
                    "name": backend_name,
                    "prob": float(operation.get("prob", 1.0)),
                    "strength": float(operation.get("strength", 1.0)),
                    "params": params,
                }
            )
        return operations

    def _make_operation(self, name: str) -> dict:
        backend_name = self._backend_name(name)
        if backend_name == "rotate":
            return copy.deepcopy(MOCK_POLICY["operations"][-1])
        category = self._category_for(backend_name)
        params = {}
        if backend_name in self.schema_lookup:
            params = self.schema_lookup[backend_name].default_params()
        return {
            "name": backend_name,
            "category": category,
            "prob": 0.5,
            "strength": 0.5,
            "bbox_effect": "会改变" if backend_name in {"horizontal_flip", "vertical_flip", "rotate", "scale", "translate", "affine", "crop"} else "不改变",
            "params": params,
        }

    def _decorate_operation(self, operation: dict) -> dict:
        decorated = dict(operation)
        name = str(decorated.get("name", ""))
        backend_name = self._backend_name(name)
        decorated["name"] = backend_name
        decorated.setdefault("category", self._category_for(backend_name))
        schema = self.schema_lookup.get(backend_name)
        decorated.setdefault("bbox_effect", "会改变" if schema and schema.changes_bboxes else "不改变")
        decorated.setdefault("params", {})
        return decorated

    def _category_for(self, name: str) -> str:
        backend_name = self._backend_name(name)
        if backend_name in {"brightness", "contrast", "gamma"}:
            return "光照 / 颜色"
        if backend_name in {"horizontal_flip", "vertical_flip", "rotate", "scale", "translate", "affine", "crop", "resize_letterbox"}:
            return "几何变换"
        if backend_name in {"gaussian_noise", "salt_pepper_noise", "gaussian_blur", "motion_blur", "median_blur"}:
            return "噪声 / 模糊"
        return "工业增强"

    def _backend_name(self, name: str) -> str:
        normalized = str(name).strip().lower()
        return OPERATION_ALIASES.get(normalized, normalized)
