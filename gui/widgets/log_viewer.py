from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


MOCK_LOGS = [
    "[14:28:10] > Experiment started.",
    r"[14:28:10] > Dataset: D:\AutoAugment\dataset",
    "[14:28:10] > Trials: 24, Epochs: 50, ImgSize: 640",
    "[14:28:11] > Trial 7 started.",
    "[14:28:11] > Augmenting 500 samples ...",
    "[14:28:13] > Augment completed. Time: 1.82s",
    "[14:28:13] > Training model (Epoch 32/50) ...",
    "[14:28:15] > Epoch 32/50 - Loss: 0.8421 - mAP50: 0.8763",
    "...",
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
