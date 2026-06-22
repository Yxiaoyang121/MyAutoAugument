from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


MOCK_LOGS = [
    "[14:28:10] > Waiting for training task.",
    "[14:28:10] > Configure mode, data.yaml, model and training parameters.",
]


class LogViewer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.addStretch(1)
        self.auto_scroll = QCheckBox("自动滚动")
        self.auto_scroll.setObjectName("logAutoScroll")
        self.auto_scroll.setChecked(True)
        clear = QPushButton("清空日志")
        clear.setObjectName("secondaryButton")
        clear.clicked.connect(self.clear)
        toolbar.addWidget(self.auto_scroll)
        toolbar.addWidget(clear)

        self.text = QPlainTextEdit()
        self.text.setObjectName("terminalLog")
        self.text.setReadOnly(True)
        self.text.setMaximumHeight(128)
        self.text.setPlainText("\n".join(MOCK_LOGS))
        layout.addLayout(toolbar)
        layout.addWidget(self.text)

    def append(self, message: str) -> None:
        self.text.appendPlainText(message)
        if self.auto_scroll.isChecked():
            self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def clear(self) -> None:
        self.text.clear()
