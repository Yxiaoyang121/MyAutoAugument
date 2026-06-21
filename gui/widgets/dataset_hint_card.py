from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class SuccessIllustration(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(160, 78)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ECFDF5"))
        painter.drawRoundedRect(QRectF(12, 72, 170, 12), 6, 6)
        painter.setBrush(QColor("#5CC5A6"))
        painter.drawRoundedRect(QRectF(58, 52, 102, 38), 8, 8)
        painter.setBrush(QColor("#34D399"))
        painter.drawPolygon(QPolygon([QPoint(48, 52), QPoint(170, 52), QPoint(148, 32), QPoint(70, 32)]))
        painter.setBrush(QColor("#F8FAFC"))
        painter.drawRoundedRect(QRectF(86, 16, 54, 48), 4, 4)
        painter.setPen(QPen(QColor("#A7F3D0"), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(96, 32, 128, 30)
        painter.drawLine(96, 44, 124, 42)
        painter.setPen(QPen(QColor("#0F766E"), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawEllipse(QRectF(154, 18, 48, 48))
        painter.drawLine(166, 42, 176, 52)
        painter.drawLine(176, 52, 192, 32)

class DatasetHintCard(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("datasetHintCard")
        self.setMinimumHeight(96)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(16)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(5)
        title = QLabel("提示")
        title.setObjectName("hintTitle")
        self.detail = QLabel(
            "• 数据集结构完整，所有必需文件均已找到。\n"
            "• 标签分布较为均衡，可以支持模型训练和验证。\n"
            "• 建议继续进行策略设计和预览，优化数据增强效果。"
        )
        self.detail.setObjectName("hintDetail")
        self.detail.setWordWrap(True)
        text_layout.addWidget(title)
        text_layout.addWidget(self.detail)
        layout.addLayout(text_layout, 1)
        layout.addWidget(SuccessIllustration())

    def set_detail(self, detail: str) -> None:
        self.detail.setText(detail)
