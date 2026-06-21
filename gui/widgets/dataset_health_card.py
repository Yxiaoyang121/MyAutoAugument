from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from gui.widgets.section_card import SectionCard


class HealthRing(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(124, 124)
        self.score = 0
        self.tone = "success"

    def set_score(self, score: int, tone: str = "success") -> None:
        self.score = max(0, min(100, score))
        self.tone = tone
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(16, 16, self.width() - 32, self.height() - 32)
        color = QColor({"success": "#10B981", "warning": "#F59E0B", "error": "#EF4444"}.get(self.tone, "#10B981"))
        bg_color = QColor({"success": "#D1FAE5", "warning": "#FEF3C7", "error": "#FEE2E2"}.get(self.tone, "#D1FAE5"))

        painter.setPen(QPen(bg_color, 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 0, 360 * 16)
        painter.setPen(QPen(color, 15, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 90 * 16, int(-360 * 16 * self.score / 100))

        painter.setPen(QColor("#0F172A"))
        painter.setFont(QFont("Microsoft YaHei", 19, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(0, -12, 0, 0), Qt.AlignmentFlag.AlignCenter, str(self.score))
        painter.setPen(QColor("#334155"))
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.DemiBold))
        painter.drawText(rect.adjusted(0, 36, 0, 0), Qt.AlignmentFlag.AlignCenter, "健康评分")


class DatasetHealthCard(SectionCard):
    def __init__(self) -> None:
        super().__init__("数据集健康状态")
        self.setMinimumHeight(188)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(14)

        self.ring = HealthRing()
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 10, 0, 0)
        text_layout.setSpacing(10)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(8)
        status_label = QLabel("健康状态：")
        status_label.setObjectName("datasetHealthLabel")
        self.status_value = QLabel("良好")
        self.status_value.setObjectName("datasetHealthValue")
        self.status_value.setProperty("tone", "success")
        status_row.addWidget(status_label)
        status_row.addWidget(self.status_value)
        status_row.addStretch(1)

        self.detail_label = QLabel("数据集结构完整，标签分布合理，可以正常进行实验。")
        self.detail_label.setObjectName("datasetHealthDetail")
        self.detail_label.setWordWrap(True)
        self.detail_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        text_layout.addLayout(status_row)
        text_layout.addWidget(self.detail_label)
        text_layout.addStretch(1)
        row.addWidget(self.ring)
        row.addLayout(text_layout, 1)
        self.body.addLayout(row)

    def set_health(self, score: int, status: str, detail: str) -> None:
        tone = "success" if score >= 80 else "warning" if score >= 60 else "error"
        self.ring.set_score(score, tone)
        self.status_value.setText(status)
        self.status_value.setProperty("tone", tone)
        self.status_value.style().unpolish(self.status_value)
        self.status_value.style().polish(self.status_value)
        self.detail_label.setText(detail)
