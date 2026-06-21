from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout


class AnalysisSummaryCard(QFrame):
    """Compact summary metric used on the analysis center."""

    def __init__(self, title: str, value: str, icon: str, tone: str = "green") -> None:
        super().__init__()
        self.setObjectName("analysisSummaryCard")
        self.setMinimumHeight(76)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(8)

        icon_label = QLabel(icon)
        icon_label.setObjectName("analysisSummaryIcon")
        icon_label.setProperty("tone", tone)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("analysisSummaryTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("analysisSummaryValue")
        self.value_label.setWordWrap(True)

        text_layout.addWidget(title_label)
        text_layout.addWidget(self.value_label, 1)

        layout.addWidget(icon_label)
        layout.addLayout(text_layout, 1)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)
