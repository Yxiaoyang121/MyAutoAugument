from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from gui.widgets.section_card import SectionCard


class PolicyManagementPanel(SectionCard):
    save_requested = Signal()
    save_as_requested = Signal()
    import_requested = Signal()
    export_requested = Signal()
    copy_requested = Signal()
    reset_requested = Signal()

    def __init__(self) -> None:
        super().__init__("策略管理")
        self.setFixedHeight(104)
        self.outer_layout.setContentsMargins(12, 8, 12, 8)
        self.outer_layout.setSpacing(6)
        if self.title_label:
            self.title_label.setMaximumHeight(18)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        summary, summary_body = self._panel("policyMiniCard")
        summary_body.addWidget(self._title("当前策略"))
        name_row, self.name_value = self._meta("策略名称", "gui_policy")
        count_row, self.count_value = self._meta("操作数量", "4")
        created_row, self.created_value = self._meta("创建时间", "2025-06-16 14:15:22")
        summary_body.addLayout(name_row)
        summary_body.addLayout(count_row)
        summary_body.addLayout(created_row)

        desc, desc_body = self._panel("policyDescCard")
        desc_body.addWidget(self._title("策略描述（可选）"))
        detail = QLabel("通用策略：光照增强 + 几何变换，提升模型泛化能力。")
        detail.setObjectName("policyDescText")
        detail.setWordWrap(True)
        desc_body.addWidget(detail, 1)

        buttons = QGridLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setHorizontalSpacing(6)
        buttons.setVerticalSpacing(0)
        specs = [
            ("保存策略", "policySaveButton", self.save_requested),
            ("另存为", "policyActionButton", self.save_as_requested),
            ("导入策略", "policyImportButton", self.import_requested),
            ("导出 JSON", "policyJsonButton", self.export_requested),
            ("复制 JSON", "policyJsonButton", self.copy_requested),
            ("重置策略", "policyResetButton", self.reset_requested),
        ]
        for index, (text, object_name, signal) in enumerate(specs):
            button = QPushButton(text)
            button.setObjectName(object_name)
            button.clicked.connect(signal.emit)
            buttons.addWidget(button, 0, index)

        row.addWidget(summary, 2)
        row.addWidget(desc, 3)
        row.addLayout(buttons, 6)
        self.body.addLayout(row)

    def set_summary(self, name: str, count: int) -> None:
        self.name_value.setText(name)
        self.count_value.setText(str(count))

    def _panel(self, object_name: str) -> tuple[QFrame, QVBoxLayout]:
        panel = QFrame()
        panel.setObjectName(object_name)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(4)
        return panel, layout

    def _title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("policyMiniTitle")
        return label

    def _meta(self, label: str, value: str) -> tuple[QHBoxLayout, QLabel]:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        key = QLabel(label)
        key.setObjectName("policyMetaLabel")
        val = QLabel(value)
        val.setObjectName("policyMetaValue")
        layout.addWidget(key)
        layout.addStretch(1)
        layout.addWidget(val)
        return layout, val
