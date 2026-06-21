from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout


class StatCard(QFrame):
    """Dashboard metric card used by the overview console."""

    def __init__(self, title: str, value: str, icon: str, tone: str = "green") -> None:
        super().__init__()
        self.setObjectName("statCard")
        self.setMinimumHeight(86)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 16, 12)
        layout.setSpacing(12)

        self.icon_label = QLabel(icon)
        self.icon_label.setObjectName("statIconBox")
        self.icon_label.setProperty("tone", tone)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("statTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statValue")

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.value_label)
        text_layout.addStretch(1)

        layout.addWidget(self.icon_label)
        layout.addLayout(text_layout, 1)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)
