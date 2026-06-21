from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from gui.widgets.common import configure_table
from gui.widgets.status_badge import StatusBadge


class AnalysisTrialTable(QWidget):
    def __init__(self, rows: list[dict]) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("试验记录")
        title.setObjectName("analysisBlockTitle")
        layout.addWidget(title)

        self.table = QTableWidget(0, 8)
        self.table.setObjectName("analysisTrialTable")
        self.table.setHorizontalHeaderLabels(["Trial", "Score", "mAP50", "mAP50-95", "bbox 保留率", "状态", "时间", "操作"])
        configure_table(
            self.table,
            fixed_columns={0: 72, 1: 88, 2: 88, 3: 104, 4: 110, 5: 94, 7: 72},
            stretch_columns=(6,),
            row_height=30,
        )
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self.table, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)
        info = QLabel("共 24 条")
        info.setObjectName("analysisPagerText")
        page_size = QPushButton("10条/页")
        page_size.setObjectName("secondaryButton")
        previous = QPushButton("‹")
        previous.setObjectName("secondaryButton")
        current = QLabel("1")
        current.setObjectName("analysisPagerCurrent")
        page2 = QLabel("2")
        page2.setObjectName("analysisPagerText")
        page3 = QLabel("3")
        page3.setObjectName("analysisPagerText")
        next_button = QPushButton("›")
        next_button.setObjectName("secondaryButton")
        jump = QLabel("前往  1  页")
        jump.setObjectName("analysisPagerText")
        footer.addWidget(info)
        footer.addWidget(page_size)
        footer.addStretch(1)
        footer.addWidget(previous)
        footer.addWidget(current)
        footer.addWidget(page2)
        footer.addWidget(page3)
        footer.addWidget(next_button)
        footer.addStretch(1)
        footer.addWidget(jump)
        layout.addLayout(footer)

        self.set_rows(rows)

    def set_rows(self, rows: list[dict]) -> None:
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row["trial"],
                f"{row['score']:.3f}",
                f"{row['map50']:.3f}",
                f"{row['map5095']:.3f}",
                row["bbox_retention"],
                row["status"],
                row["time"],
                "查看",
            ]
            for column, value in enumerate(values):
                if column == 5:
                    badge = StatusBadge(str(value), self._status_tone(str(value)))
                    self.table.setCellWidget(row_index, column, badge)
                    continue
                if column == 7:
                    button = QPushButton("◉")
                    button.setToolTip("查看 Trial 详情")
                    button.setObjectName("analysisViewButton")
                    self.table.setCellWidget(row_index, column, button)
                    continue
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if row["status"] == "最优":
                    item.setBackground(QColor("#FFFBEB"))
                self.table.setItem(row_index, column, item)

    def _status_tone(self, status: str) -> str:
        if status == "最优":
            return "warning"
        if status == "已接受":
            return "success"
        if status == "拒绝":
            return "error"
        return "neutral"
