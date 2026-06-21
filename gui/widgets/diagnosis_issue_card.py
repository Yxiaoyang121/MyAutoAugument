from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from gui.widgets.status_badge import StatusBadge


class DiagnosisIssueCard(QFrame):
    def __init__(self, title: str, severity: str, description: str, evidence: str, suggestion: str) -> None:
        super().__init__()
        self.setObjectName("diagnosisIssueCard")
        self.setProperty("tone", self._tone(severity))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 7, 8, 7)
        layout.setSpacing(8)

        icon = QLabel(self._icon(severity))
        icon.setObjectName("diagnosisIssueIcon")
        icon.setProperty("tone", self._tone(severity))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setObjectName("diagnosisIssueTitle")
        badge = StatusBadge(severity, self._tone(severity))
        header.addWidget(title_label, 1)
        header.addWidget(badge)

        for text, obj in [
            (description, "diagnosisIssueDescription"),
            (f"证据：{evidence}", "diagnosisIssueMeta"),
            (f"建议：{suggestion}", "diagnosisIssueMeta"),
        ]:
            label = QLabel(text)
            label.setObjectName(obj)
            label.setWordWrap(True)
            body.addWidget(label)

        body.insertLayout(0, header)

        arrow = QLabel("›")
        arrow.setObjectName("diagnosisIssueArrow")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(icon)
        layout.addLayout(body, 1)
        layout.addWidget(arrow)

    def _tone(self, severity: str) -> str:
        if severity == "严重":
            return "error"
        if severity == "中等":
            return "warning"
        return "success"

    def _icon(self, severity: str) -> str:
        if severity == "严重":
            return "!"
        if severity == "中等":
            return "i"
        return "✓"
