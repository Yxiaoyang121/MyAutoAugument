from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


DEFAULT_STAGES = [
    {"name": "增强", "en": "Augment", "status": "等待中", "progress": 0},
    {"name": "训练", "en": "Train", "status": "等待中", "progress": 0},
    {"name": "验证", "en": "Val", "status": "等待中", "progress": 0},
    {"name": "诊断", "en": "Diag", "status": "等待中", "progress": 0},
    {"name": "策略更新", "en": "Update", "status": "等待中", "progress": 0},
]


class StageNode(QWidget):
    def __init__(self, stage: dict) -> None:
        super().__init__()
        self.stage = stage
        self.setObjectName("stageNode")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.icon = QLabel(self._icon_text())
        self.icon.setObjectName("stageIcon")
        self.icon.setProperty("status", self._tone())

        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        name = QLabel(str(stage["name"]))
        name.setToolTip(str(stage.get("en", "")))
        name.setObjectName("stageName")
        status = QLabel(str(stage["status"]))
        status.setObjectName("stageStatus")
        status.setProperty("status", self._tone())
        progress = QLabel(f"{stage['progress']}%")
        progress.setObjectName("stageProgress")
        text.addWidget(name)
        text.addWidget(status)
        text.addWidget(progress)
        layout.addWidget(self.icon)
        layout.addLayout(text, 1)

    def _tone(self) -> str:
        status = str(self.stage.get("status", "等待中"))
        if status == "已完成":
            return "done"
        if status == "进行中":
            return "active"
        return "wait"

    def _icon_text(self) -> str:
        tone = self._tone()
        if tone == "done":
            return "✓"
        if tone == "active":
            return "•"
        return "○"


class Connector(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedWidth(18)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#94A3B8"), 1.4, Qt.PenStyle.DashLine))
        y = self.height() / 2
        painter.drawLine(QPointF(4, y), QPointF(self.width() - 4, y))


class ExperimentStepper(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.set_stages(DEFAULT_STAGES)

    def set_stages(self, stages: list[dict]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        for index, stage in enumerate(stages):
            self._layout.addWidget(StageNode(stage), 1)
            if index < len(stages) - 1:
                self._layout.addWidget(Connector())
