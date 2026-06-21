from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class SectionCard(QFrame):
    """White rounded card with a compact title and a reusable body layout."""

    def __init__(self, title: str = "") -> None:
        super().__init__()
        self.setObjectName("sectionCard")
        self.outer_layout = QVBoxLayout(self)
        self.outer_layout.setContentsMargins(14, 12, 14, 14)
        self.outer_layout.setSpacing(9)

        if title:
            self.title_label = QLabel(title)
            self.title_label.setObjectName("sectionCardTitle")
            self.outer_layout.addWidget(self.title_label)
        else:
            self.title_label = None

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(9)
        self.outer_layout.addLayout(self.body, 1)
