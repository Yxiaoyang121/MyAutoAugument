from __future__ import annotations

import json

from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget


LOG_FILES = ["summary.json", "trial_record.json", "diagnosis.json", "policy.json", "metrics.json", "stdout.log", "stderr.log"]


class AnalysisLogPanel(QWidget):
    def __init__(self, json_payloads: dict[str, object], stdout_text: str, stderr_text: str) -> None:
        super().__init__()
        self.json_payloads = json_payloads
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        left = QVBoxLayout()
        title = QLabel("日志文件")
        title.setObjectName("analysisBlockTitle")
        self.file_list = QListWidget()
        self.file_list.setObjectName("analysisFileList")
        self.file_list.addItems(LOG_FILES)
        self.file_list.currentTextChanged.connect(self._show_file)
        left.addWidget(title)
        left.addWidget(self.file_list, 1)

        right = QVBoxLayout()
        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        for text in ["复制", "打开文件", "打开目录"]:
            button = QPushButton(text)
            button.setObjectName("secondaryButton")
            toolbar.addWidget(button)
        self.viewer = QPlainTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setObjectName("analysisLogViewer")
        right.addLayout(toolbar)
        right.addWidget(self.viewer, 1)

        layout.addLayout(left, 1)
        layout.addLayout(right, 3)
        self.file_list.setCurrentRow(0)

    def _show_file(self, name: str) -> None:
        if name.endswith(".log"):
            self.viewer.setObjectName("analysisTerminalViewer")
            self.viewer.setPlainText(self.stderr_text if name == "stderr.log" else self.stdout_text)
        else:
            self.viewer.setObjectName("analysisLogViewer")
            payload = self.json_payloads.get(name, {})
            self.viewer.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self.viewer.style().unpolish(self.viewer)
        self.viewer.style().polish(self.viewer)
