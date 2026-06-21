from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from gui.services import ResultLoader
from gui.widgets.mini_charts import MiniDonutChart, MiniLineChart
from gui.widgets.section_card import SectionCard
from gui.widgets.stat_card import StatCard


MAP50_VALUES = [
    0.75,
    0.78,
    0.76,
    0.74,
    0.80,
    0.77,
    0.78,
    0.76,
    0.74,
    0.73,
    0.75,
    0.76,
    0.78,
    0.81,
    0.80,
    0.79,
    0.81,
    0.82,
    0.83,
    0.84,
    0.83,
    0.83,
    0.84,
    0.85,
    0.84,
]
MAP5095_VALUES = [
    0.45,
    0.46,
    0.47,
    0.46,
    0.48,
    0.49,
    0.50,
    0.48,
    0.47,
    0.46,
    0.45,
    0.46,
    0.47,
    0.48,
    0.47,
    0.46,
    0.45,
    0.46,
    0.47,
    0.48,
    0.48,
    0.49,
    0.49,
    0.50,
    0.50,
]


class QuickEntry(QFrame):
    clicked = Signal()

    def __init__(self, icon: str, title: str, detail: str, tone: str = "green") -> None:
        super().__init__()
        self.setObjectName("quickEntry")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(54)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        icon_label = QLabel(icon)
        icon_label.setObjectName("quickIconBox")
        icon_label.setProperty("tone", tone)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)
        title_label = QLabel(title)
        title_label.setObjectName("quickTitle")
        detail_label = QLabel(detail)
        detail_label.setObjectName("quickDetail")
        text_layout.addWidget(title_label)
        text_layout.addWidget(detail_label)

        layout.addWidget(icon_label)
        layout.addLayout(text_layout, 1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class DashboardPage(QWidget):
    navigate_requested = Signal(int)

    def __init__(self, project_root: Path, result_loader: ResultLoader) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.result_loader = result_loader
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("dashboardPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        self._build_stat_cards(layout)
        self._build_visual_cards(layout)
        self._build_quick_entries(layout)
        layout.addStretch(1)

    def _build_stat_cards(self, parent: QVBoxLayout) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        cards = [
            StatCard("试验次数", "24", "⚗", "green"),
            StatCard("最佳 mAP50", "0.934", "◎", "blue"),
            StatCard("最佳 mAP50-95", "0.712", "▥", "purple"),
            StatCard("最佳分数", "0.934", "♕", "orange"),
            StatCard("问题数量", "3", "△", "red"),
        ]
        for column, card in enumerate(cards):
            grid.addWidget(card, 0, column)
            grid.setColumnStretch(column, 1)
        parent.addLayout(grid)

    def _build_visual_cards(self, parent: QVBoxLayout) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        trend_card = SectionCard("mAP 趋势")
        trend_card.setMinimumHeight(198)
        chart = MiniLineChart()
        chart.set_series(
            [
                ("mAP50", "#10B981", MAP50_VALUES),
                ("mAP50-95", "#2563EB", MAP5095_VALUES),
            ]
        )
        trend_card.body.addWidget(chart)

        status_card = SectionCard("试验状态分布")
        status_card.setMinimumHeight(198)
        donut = MiniDonutChart()
        donut.set_segments(
            [
                ("已接受", 18, "#10B981"),
                ("已拒绝", 2, "#EF4444"),
                ("运行中", 2, "#2563EB"),
                ("失败", 2, "#64748B"),
            ]
        )
        status_card.body.addWidget(donut)

        policy_card = self._policy_card()
        policy_card.setMinimumHeight(198)

        grid.addWidget(trend_card, 0, 0)
        grid.addWidget(status_card, 0, 1)
        grid.addWidget(policy_card, 0, 2)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 2)
        parent.addLayout(grid, 1)

    def _policy_card(self) -> SectionCard:
        card = SectionCard("当前策略摘要")
        body = card.body

        rows = [
            ("策略名称", "gui_policy"),
            ("策略操作数", "4"),
            ("策略更新时间", "2026-06-16 14:15"),
            ("策略操作顺序", ""),
        ]
        for label, value in rows:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            label_widget = QLabel(label)
            label_widget.setObjectName("policyMetaLabel")
            value_widget = QLabel(value)
            value_widget.setObjectName("policyMetaValue")
            row.addWidget(label_widget)
            row.addStretch(1)
            if value:
                row.addWidget(value_widget)
            body.addLayout(row)

        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 4, 0, 0)
        badge_row.setSpacing(8)
        for name in ["brightness", "contrast", "horizontal_flip", "rotate"]:
            badge = QLabel(name)
            badge.setObjectName("policyOpBadge")
            badge_row.addWidget(badge)
        badge_row.addStretch(1)
        body.addLayout(badge_row)
        body.addStretch(1)
        return card

    def _build_quick_entries(self, parent: QVBoxLayout) -> None:
        card = SectionCard("快捷入口")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        entries = [
            ("</>", "检查数据集", "检查数据集完整性", "green", 1),
            ("✦", "编辑策略", "设计增强策略", "blue", 2),
            ("▷", "运行预览", "预览增强效果", "orange", 3),
            ("▶", "开始实验", "运行自动搜索", "green", 4),
            ("↗", "查看分析", "查看结果与诊断", "purple", 5),
        ]
        for icon, title, detail, tone, target in entries:
            entry = QuickEntry(icon, title, detail, tone)
            entry.clicked.connect(lambda row_index=target: self.navigate_requested.emit(row_index))
            row.addWidget(entry)
        card.body.addLayout(row)
        parent.addWidget(card)
