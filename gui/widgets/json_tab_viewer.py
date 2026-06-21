from __future__ import annotations

import json

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


class JsonTabViewer(QWidget):
    def __init__(self, payloads: dict[str, object]) -> None:
        super().__init__()
        self.payloads = payloads
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        toolbar = QHBoxLayout()
        self.selector = QComboBox()
        self.selector.addItems(payloads.keys())
        self.selector.currentTextChanged.connect(self._render)
        toolbar.addWidget(self.selector)
        toolbar.addStretch(1)
        for text in ["格式化", "复制", "打开文件"]:
            button = QPushButton(text)
            button.setObjectName("secondaryButton")
            toolbar.addWidget(button)
        layout.addLayout(toolbar)

        self.viewer = QPlainTextEdit()
        self.viewer.setObjectName("jsonTabViewer")
        self.viewer.setReadOnly(True)
        layout.addWidget(self.viewer, 1)
        self._render(self.selector.currentText())

    def _render(self, name: str) -> None:
        self.viewer.setPlainText(json.dumps(self.payloads.get(name, {}), ensure_ascii=False, indent=2))
