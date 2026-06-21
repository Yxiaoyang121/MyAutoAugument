from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.widgets.diagnosis_issue_card import DiagnosisIssueCard


class DiagnosisSummaryPanel(QWidget):
    def __init__(self, counts: dict[str, int], issues: list[dict]) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = QLabel("诊断摘要")
        title.setObjectName("analysisBlockTitle")
        layout.addWidget(title)

        grid_box = QFrame()
        grid_box.setObjectName("diagnosisCountBox")
        grid = QGridLayout(grid_box)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setSpacing(0)
        for column, (name, value) in enumerate(counts.items()):
            cell = self._count_cell(name, value)
            grid.addWidget(cell, 0, column)
        layout.addWidget(grid_box)

        for issue in issues[:3]:
            layout.addWidget(
                DiagnosisIssueCard(
                    issue["title"],
                    issue["severity"],
                    issue["description"],
                    issue["evidence"],
                    issue["suggestion"],
                )
            )

        more = QPushButton("查看更多诊断")
        more.setObjectName("secondaryButton")
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(more)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)

    def _count_cell(self, name: str, value: int) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(3, 2, 3, 2)
        layout.setSpacing(1)
        title = QLabel(name)
        title.setObjectName("diagnosisCountTitle")
        number = QLabel(str(value))
        number.setObjectName("diagnosisCountValue")
        number.setProperty("tone", self._tone(name))
        layout.addWidget(title)
        layout.addWidget(number)
        return widget

    def _tone(self, name: str) -> str:
        if "严重" in name:
            return "error"
        if "中等" in name:
            return "warning"
        if "轻微" in name:
            return "success"
        return "blue"
