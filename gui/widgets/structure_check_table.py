from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class StructureCheckTable(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._rows: list[QFrame] = []
        self._build_header()

    def _build_header(self) -> None:
        header = QFrame()
        header.setObjectName("structureHeader")
        grid = QGridLayout(header)
        grid.setContentsMargins(14, 8, 14, 8)
        grid.setColumnStretch(0, 1)
        item = QLabel("检查项")
        item.setObjectName("structureHeaderText")
        status = QLabel("状态")
        status.setObjectName("structureHeaderText")
        grid.addWidget(item, 0, 0)
        grid.addWidget(status, 0, 1)
        self._layout.addWidget(header)

    def set_checks(self, checks: list[tuple[str, str]]) -> None:
        for row in self._rows:
            row.setParent(None)
        self._rows.clear()
        for name, status in checks:
            row = QFrame()
            row.setObjectName("structureRow")
            grid = QGridLayout(row)
            grid.setContentsMargins(14, 8, 14, 8)
            grid.setColumnStretch(0, 1)
            name_label = QLabel(name)
            name_label.setObjectName("structureItem")
            status_label = QLabel(self._status_text(status))
            status_label.setObjectName("structureStatus")
            status_label.setProperty("tone", self._tone(status))
            grid.addWidget(name_label, 0, 0)
            grid.addWidget(status_label, 0, 1)
            self._layout.addWidget(row)
            self._rows.append(row)
        self._layout.addStretch(1)

    def _status_text(self, status: str) -> str:
        if status == "正常":
            return "●  正常"
        if status == "警告":
            return "●  警告"
        return "●  缺失"

    def _tone(self, status: str) -> str:
        if status == "正常":
            return "success"
        if status == "警告":
            return "warning"
        return "error"
