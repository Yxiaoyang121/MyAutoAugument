from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
)


class Section(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("section")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 16)
        self.layout.setSpacing(10)
        header = QLabel(title)
        header.setObjectName("sectionTitle")
        self.layout.addWidget(header)


class MetricCard(QFrame):
    def __init__(self, label: str, value: str = "-", accent: str = "#0F766E", hint: str = "") -> None:
        super().__init__()
        self.setObjectName("metricCard")
        self.setMinimumHeight(88)
        self._accent = accent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.label_label = QLabel(label)
        self.label_label.setObjectName("metricLabel")
        self.hint_label = QLabel(hint)
        self.hint_label.setObjectName("metricHint")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.value_label)
        layout.addWidget(self.label_label)
        if hint:
            layout.addWidget(self.hint_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)

    def set_hint(self, value: str) -> None:
        self.hint_label.setText(value)
        self.hint_label.setVisible(bool(value))


class StatusBadge(QLabel):
    def __init__(self, text: str = "-", tone: str = "neutral") -> None:
        super().__init__(text)
        self.setObjectName("statusBadge")
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        self.setProperty("tone", tone)
        self.style().unpolish(self)
        self.style().polish(self)


class EmptyState(QFrame):
    def __init__(self, title: str, detail: str = "") -> None:
        super().__init__()
        self.setObjectName("emptyState")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("emptyTitle")
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("emptyDetail")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.detail_label)

    def set_text(self, title: str, detail: str = "") -> None:
        self.title_label.setText(title)
        self.detail_label.setText(detail)


class MiniBarChart(QFrame):
    def __init__(self, title: str = "") -> None:
        super().__init__()
        self.setObjectName("miniChart")
        self.setMinimumHeight(180)
        self.title = title
        self.values: list[tuple[str, float]] = []

    def set_values(self, values: list[tuple[str, float]]) -> None:
        self.values = values
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(14, 12, -14, -14)
        painter.setPen(QColor("#64748B"))
        if self.title:
            painter.drawText(rect.left(), rect.top() + 14, self.title)
            rect.adjust(0, 24, 0, 0)
        if not self.values:
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "暂无数据")
            return
        max_value = max(value for _, value in self.values) or 1.0
        bar_h = max(14, min(28, int(rect.height() / max(len(self.values), 1)) - 8))
        y = rect.top()
        for label, value in self.values[:12]:
            width = int((rect.width() - 110) * value / max_value)
            painter.setPen(QColor("#64748B"))
            painter.drawText(rect.left(), y + bar_h - 2, str(label)[:12])
            painter.setBrush(QColor("#0F766E"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect.left() + 100, y, max(width, 2), bar_h, 5, 5)
            painter.setPen(QColor("#0F172A"))
            painter.drawText(rect.left() + 106 + max(width, 2), y + bar_h - 2, f"{value:g}")
            y += bar_h + 8


StatCard = MetricCard
SectionCard = Section

def row(*widgets) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    for widget in widgets:
        layout.addWidget(widget)
    return layout


def icon_button(text: str, icon, tooltip: str = "") -> QPushButton:
    button = QPushButton(text)
    if icon is not None:
        button.setIcon(icon)
    if tooltip:
        button.setToolTip(tooltip)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def configure_table(
    table: QTableWidget,
    *,
    stretch_columns: tuple[int, ...] = (),
    resize_to_contents: tuple[int, ...] = (),
    fixed_columns: dict[int, int] | None = None,
    min_column_widths: dict[int, int] | None = None,
    row_height: int = 34,
    word_wrap: bool = True,
) -> None:
    """Apply consistent desktop-friendly table behavior.

    Tables in this GUI often contain long paths, JSON, or Chinese text. The
    default Qt header behavior makes those columns either too narrow or huge, so
    callers specify only the columns that should stretch or stay compact.
    """

    fixed_columns = fixed_columns or {}
    min_column_widths = min_column_widths or {}
    header = table.horizontalHeader()
    vertical = table.verticalHeader()
    vertical.setVisible(False)
    vertical.setDefaultSectionSize(row_height)
    table.setAlternatingRowColors(True)
    table.setWordWrap(word_wrap)
    table.setTextElideMode(Qt.TextElideMode.ElideRight)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    header.setStretchLastSection(False)
    header.setMinimumSectionSize(48)
    header.setDefaultSectionSize(120)
    for column in range(table.columnCount()):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
    for column in resize_to_contents:
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    for column in stretch_columns:
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
    for column, width in fixed_columns.items():
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(column, width)
    for column, width in min_column_widths.items():
        if table.columnWidth(column) < width:
            table.setColumnWidth(column, width)
