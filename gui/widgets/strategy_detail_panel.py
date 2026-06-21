from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.widgets.status_badge import StatusBadge


class StrategyDetailPanel(QWidget):
    def __init__(self, policy: dict) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        content.addWidget(self._summary_panel(policy), 1)
        content.addWidget(self._pipeline_panel(policy), 2)
        content.addWidget(self._evaluation_panel(), 1)
        layout.addLayout(content, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for text, obj in [
            ("应用为当前策略", "primaryButton"),
            ("导出策略", "secondaryButton"),
            ("查看完整 JSON", "secondaryButton"),
        ]:
            button = QPushButton(text)
            button.setObjectName(obj)
            buttons.addWidget(button)
        layout.addLayout(buttons)

    def _summary_panel(self, policy: dict) -> QFrame:
        panel = QFrame()
        panel.setObjectName("strategyDetailPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        title = QLabel("最佳策略摘要")
        title.setObjectName("analysisBlockTitle")
        layout.addWidget(title)
        rows = [
            ("策略名称", policy["name"]),
            ("来源 Trial", str(policy["trial"])),
            ("操作数量", str(len(policy["operations"]))),
            ("Score", f"{policy['score']:.3f}"),
            ("mAP50", f"{policy['map50']:.3f}"),
            ("mAP50-95", f"{policy['map5095']:.3f}"),
        ]
        for name, value in rows:
            layout.addLayout(self._kv(name, value))
        layout.addStretch(1)
        return panel

    def _pipeline_panel(self, policy: dict) -> QFrame:
        panel = QFrame()
        panel.setObjectName("strategyDetailPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title = QLabel("策略流水线")
        title.setObjectName("analysisBlockTitle")
        layout.addWidget(title)
        for index, op in enumerate(policy["operations"], 1):
            card = QFrame()
            card.setObjectName("strategyOpCard")
            row = QHBoxLayout(card)
            row.setContentsMargins(10, 8, 10, 8)
            row.setSpacing(10)
            number = QLabel(str(index))
            number.setObjectName("strategyOpIndex")
            name = QLabel(op["name"])
            name.setObjectName("strategyOpName")
            meta = QLabel(f"prob {op['prob']:.2f}  |  strength {op['strength']:.2f}  |  params {op['params']}")
            meta.setObjectName("strategyOpMeta")
            badge = StatusBadge(op["bbox_effect"], "warning" if op["bbox_effect"] == "会改变" else "success")
            row.addWidget(number)
            row.addWidget(name)
            row.addWidget(meta, 1)
            row.addWidget(badge)
            layout.addWidget(card)
        layout.addStretch(1)
        return panel

    def _evaluation_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("strategyDetailPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title = QLabel("策略评价")
        title.setObjectName("analysisBlockTitle")
        layout.addWidget(title)
        for heading, detail in [
            ("优点", "bbox 保留率高、mAP50 提升明显、增强质量稳定。"),
            ("风险", "旋转角度过大可能影响小目标定位。"),
            ("建议", "保持亮度/对比度增强，限制 rotate max_angle 到 10~20 度。"),
        ]:
            label = QLabel(f"{heading}：{detail}")
            label.setObjectName("strategyEvalText")
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch(1)
        return panel

    def _kv(self, key: str, value: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        key_label = QLabel(key)
        key_label.setObjectName("strategySummaryKey")
        value_label = QLabel(value)
        value_label.setObjectName("strategySummaryValue")
        row.addWidget(key_label)
        row.addStretch(1)
        row.addWidget(value_label)
        return row
