from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from gui.widgets.section_card import SectionCard


class OperationCard(QFrame):
    selected = Signal(int)

    def __init__(self, index: int, operation: dict, active: bool = False) -> None:
        super().__init__()
        self.index = index
        self.setObjectName("operationCard")
        self.setProperty("active", active)
        self.setMinimumHeight(56)
        self.setMaximumHeight(62)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 6)
        layout.setSpacing(8)

        number = QLabel(str(index + 1))
        number.setObjectName("operationIndex")
        number.setProperty("active", active)
        layout.addWidget(number)

        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(3)
        name = QLabel(str(operation.get("name", "-")))
        name.setObjectName("operationName")
        meta = QHBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        prob = QLabel(f"prob: {float(operation.get('prob', 0.0)):.2f}")
        prob.setObjectName("operationMeta")
        strength = QLabel(f"strength: {float(operation.get('strength', 0.0)):.2f}")
        strength.setObjectName("operationMeta")
        bbox = QLabel(f"bbox: {operation.get('bbox_effect', '不改变')}")
        bbox.setObjectName("bboxBadge")
        bbox.setProperty("effect", "change" if operation.get("bbox_effect") == "会改变" else "stable")
        meta.addWidget(prob)
        meta.addWidget(strength)
        meta.addStretch(1)
        meta.addWidget(bbox)
        content.addWidget(name)
        content.addLayout(meta)
        layout.addLayout(content, 1)

        handle = QLabel("⋮\n⋮")
        handle.setObjectName("dragHandle")
        layout.addWidget(handle)

    def mousePressEvent(self, event) -> None:
        self.selected.emit(self.index)
        super().mousePressEvent(event)


class PolicyPipeline(SectionCard):
    selection_changed = Signal(int)
    move_requested = Signal(int)
    clear_requested = Signal()

    def __init__(self) -> None:
        super().__init__("")
        self.operations: list[dict] = []
        self.selected_index = 0
        self.header = QHBoxLayout()
        self.header.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("当前策略流水线（0 个操作）")
        self.title.setObjectName("sectionCardTitle")
        clear = QPushButton("清空")
        clear.setObjectName("secondaryButton")
        up = QPushButton("上移")
        up.setObjectName("secondaryButton")
        down = QPushButton("下移")
        down.setObjectName("secondaryButton")
        clear.clicked.connect(self.clear_requested.emit)
        up.clicked.connect(lambda: self.move_requested.emit(-1))
        down.clicked.connect(lambda: self.move_requested.emit(1))
        self.header.addWidget(self.title, 1)
        self.header.addWidget(clear)
        self.header.addWidget(up)
        self.header.addWidget(down)
        self.body.addLayout(self.header)

        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(4)
        self.body.addLayout(self.cards_layout, 1)

        self.drop_hint = QLabel("从左侧拖拽算子到此处插入")
        self.drop_hint.setObjectName("pipelineDropHint")
        self.body.addWidget(self.drop_hint)

    def set_operations(self, operations: list[dict], selected_index: int = 0) -> None:
        self.operations = operations
        self.selected_index = min(max(selected_index, 0), max(len(operations) - 1, 0))
        self._render()

    def _render(self) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self.title.setText(f"当前策略流水线（{len(self.operations)} 个操作）")
        if not self.operations:
            empty = QLabel("尚未配置策略，请从左侧增强算子库添加操作。")
            empty.setObjectName("emptyDetail")
            self.cards_layout.addWidget(empty)
            return
        for index, operation in enumerate(self.operations):
            card = OperationCard(index, operation, active=index == self.selected_index)
            card.selected.connect(self.selection_changed.emit)
            self.cards_layout.addWidget(card)
            if index < len(self.operations) - 1:
                arrow = QLabel("↓")
                arrow.setObjectName("pipelineArrow")
                self.cards_layout.addWidget(arrow)
