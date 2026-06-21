from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from gui.adapters import BackendCapabilityAdapter
from gui.widgets.common import Section, configure_table


class AugmentationLibraryPage(QWidget):
    def __init__(self, capability_adapter: BackendCapabilityAdapter | None = None) -> None:
        super().__init__()
        self.capabilities = capability_adapter or BackendCapabilityAdapter()
        self.schemas = self.capabilities.list_augmentations()
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(16)

        title = QLabel("增强库")
        title.setObjectName("pageTitle")
        subtitle = QLabel("展示后端已注册的增强方法，参数与 AutoAugment.augmentations.ops 保持一致。")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        section = Section("已注册增强方法")
        row = QHBoxLayout()
        row.addWidget(QLabel("类别"))
        self.category_filter = QComboBox()
        self.category_filter.addItem("全部", "All")
        for category in sorted({schema.category for schema in self.schemas}):
            self.category_filter.addItem(category, category)
        self.category_filter.currentTextChanged.connect(self._populate)
        row.addWidget(self.category_filter)
        row.addStretch(1)
        section.layout.addLayout(row)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["类别", "方法", "说明", "参数 / 默认值 / 范围", "BBox", "检测适用性"])
        configure_table(
            self.table,
            stretch_columns=(2, 3),
            fixed_columns={0: 112, 4: 112, 5: 96},
            min_column_widths={1: 150, 2: 240, 3: 300},
        )
        section.layout.addWidget(self.table, 1)
        layout.addWidget(section, 1)

    def _populate(self) -> None:
        selected = self.category_filter.currentData() if hasattr(self, "category_filter") else "All"
        rows = [schema for schema in self.schemas if selected == "All" or schema.category == selected]
        self.table.setRowCount(len(rows))
        for row, schema in enumerate(rows):
            values = [
                schema.category,
                schema.name,
                schema.description_zh,
                schema.params_text(),
                "会改变 bbox" if schema.changes_bboxes else "不改变 bbox",
                "适用" if schema.detection_suitable else "需复核",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.table.setItem(row, column, item)
