from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


class MiniLineChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._series: list[tuple[str, QColor, list[float]]] = []

    def set_series(self, series: list[tuple[str, str, list[float]]]) -> None:
        self._series = [(name, QColor(color), values) for name, color, values in series]
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        chart = QRectF(self.rect().adjusted(42, 30, -22, -40))
        if chart.width() < 80 or chart.height() < 80:
            return

        self._draw_grid(painter, chart)
        for name, color, values in self._series:
            self._draw_series(painter, chart, values, color)
        self._draw_legend(painter, chart)
        self._draw_axis_labels(painter, chart)

    def _draw_grid(self, painter: QPainter, chart: QRectF) -> None:
        painter.setFont(QFont("Microsoft YaHei", 8))
        grid_pen = QPen(QColor("#E2E8F0"), 1)
        label_pen = QColor("#64748B")

        for index, value in enumerate([1.0, 0.5, 0.0]):
            y = chart.top() + chart.height() * index / 2
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))
            painter.setPen(label_pen)
            painter.drawText(
                QRectF(0, y - 10, chart.left() - 8, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{value:g}",
            )

        painter.setPen(QPen(QColor("#CBD5E1"), 1))
        painter.drawLine(chart.bottomLeft(), chart.bottomRight())
        painter.drawLine(chart.bottomLeft(), chart.topLeft())

        painter.setPen(label_pen)
        for tick in [0, 5, 10, 15, 20, 24]:
            x = chart.left() + chart.width() * tick / 24
            painter.drawText(
                QRectF(x - 16, chart.bottom() + 7, 32, 18),
                Qt.AlignmentFlag.AlignCenter,
                str(tick),
            )

    def _draw_series(self, painter: QPainter, chart: QRectF, values: list[float], color: QColor) -> None:
        if not values:
            return
        points = QPolygonF()
        step = chart.width() / max(len(values) - 1, 1)
        for index, raw in enumerate(values):
            value = max(0.0, min(1.0, float(raw)))
            x = chart.left() + index * step
            y = chart.bottom() - value * chart.height()
            points.append(QPointF(x, y))

        painter.setPen(QPen(color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPolyline(points)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        for point in points:
            painter.drawEllipse(point, 3.0, 3.0)

    def _draw_legend(self, painter: QPainter, chart: QRectF) -> None:
        painter.setFont(QFont("Microsoft YaHei", 8))
        x = chart.right() - 150
        y = chart.top() - 22
        for name, color, _ in self._series:
            painter.setPen(QPen(color, 2.2))
            painter.drawLine(QPointF(x, y + 8), QPointF(x + 16, y + 8))
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x + 8, y + 8), 2.6, 2.6)
            painter.setPen(QColor("#334155"))
            painter.drawText(QRectF(x + 22, y, 70, 18), name)
            x += 78

    def _draw_axis_labels(self, painter: QPainter, chart: QRectF) -> None:
        painter.setFont(QFont("Microsoft YaHei", 8))
        painter.setPen(QColor("#64748B"))
        painter.drawText(
            QRectF(chart.left(), self.height() - 24, chart.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            "试验序号",
        )


class MiniDonutChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._segments: list[tuple[str, int, QColor]] = []

    def set_segments(self, segments: list[tuple[str, int, str]]) -> None:
        self._segments = [(name, value, QColor(color)) for name, value, color in segments]
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        if not self._segments:
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "暂无数据")
            return

        total = sum(value for _, value, _ in self._segments) or 1
        size = min(128, max(86, min(self.width() // 2 - 18, self.height() - 36)))
        donut = QRectF(18, 22, size, size)
        start_angle = 90 * 16

        for _, value, color in self._segments:
            span_angle = int(-360 * 16 * value / total)
            painter.setPen(QPen(color, 20, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
            painter.drawArc(donut, start_angle, span_angle)
            start_angle += span_angle

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(donut.adjusted(28, 28, -28, -28))

        painter.setPen(QColor("#0F172A"))
        painter.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        painter.drawText(donut, Qt.AlignmentFlag.AlignCenter, str(total))
        painter.setPen(QColor("#64748B"))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(donut.adjusted(0, 42, 0, 0), Qt.AlignmentFlag.AlignCenter, "总计")

        self._draw_legend(painter, QRectF(donut.right() + 30, 32, self.width() - donut.right() - 36, self.height() - 48), total)

    def _draw_legend(self, painter: QPainter, rect: QRectF, total: int) -> None:
        painter.setFont(QFont("Microsoft YaHei", 9))
        y = rect.top()
        for name, value, color in self._segments:
            percent = int(round(value * 100 / total))
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(rect.left(), y + 4, 10, 10), 2, 2)
            painter.setPen(QColor("#334155"))
            painter.drawText(QRectF(rect.left() + 18, y, 72, 20), name)
            painter.drawText(QRectF(rect.left() + 92, y, 84, 20), f"{value}（{percent}%）")
            y += 28


class MiniBarChart(QWidget):
    def __init__(self, axis_label: str = "类别 ID") -> None:
        super().__init__()
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.axis_label = axis_label
        self.empty_text = "未检测到类别信息"
        self._values: list[tuple[str, int]] = []

    def set_values(self, values: list[tuple[str, int]]) -> None:
        self._values = values
        self.setMinimumWidth(0)
        self.update()

    def set_empty_text(self, text: str) -> None:
        self.empty_text = text
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        chart = QRectF(self.rect().adjusted(44, 18, -18, -46))
        if chart.width() < 80 or chart.height() < 80:
            return
        if not self._values:
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return

        max_value = max(value for _, value in self._values)
        if max_value <= 0:
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return
        rounded_max = self._nice_axis_max(max_value)
        self._draw_grid(painter, chart, rounded_max)
        self._draw_bars(painter, chart, rounded_max)
        self._draw_axis_label(painter, chart)

    def _draw_grid(self, painter: QPainter, chart: QRectF, max_value: int) -> None:
        painter.setFont(QFont("Microsoft YaHei", 8))
        step = max(1, max_value // 5)
        for tick in range(0, max_value + 1, step):
            ratio = tick / max_value if max_value else 0
            y = chart.bottom() - ratio * chart.height()
            painter.setPen(QPen(QColor("#E2E8F0"), 1))
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))
            painter.setPen(QColor("#64748B"))
            label = self._format_axis_label(tick)
            painter.drawText(
                QRectF(0, y - 10, chart.left() - 8, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        painter.setPen(QPen(QColor("#CBD5E1"), 1))
        painter.drawLine(chart.bottomLeft(), chart.bottomRight())
        painter.drawLine(chart.bottomLeft(), chart.topLeft())

    def _draw_bars(self, painter: QPainter, chart: QRectF, max_value: int) -> None:
        count = len(self._values)
        slot = chart.width() / max(count, 1)
        bar_width = max(6.0, min(20.0, slot * 0.44))
        for index, (label, value) in enumerate(self._values):
            x = chart.left() + slot * index + (slot - bar_width) / 2
            height = chart.height() * min(max(value / max_value, 0.0), 1.0)
            bar_rect = QRectF(x, chart.bottom() - height, bar_width, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#80C45B"))
            painter.drawRoundedRect(bar_rect, 4, 4)
            painter.setPen(QColor("#0F172A"))
            painter.setFont(QFont("Microsoft YaHei", 8))
            painter.drawText(
                QRectF(chart.left() + slot * index, chart.bottom() + 6, slot, 34),
                Qt.AlignmentFlag.AlignCenter,
                self._compact_label(str(label)),
            )

    def _draw_axis_label(self, painter: QPainter, chart: QRectF) -> None:
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.setPen(QColor("#334155"))
        painter.drawText(
            QRectF(chart.left(), self.height() - 18, chart.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            self.axis_label,
        )

    def _nice_axis_max(self, max_value: int) -> int:
        if max_value <= 10:
            return 10
        if max_value <= 50:
            return ((max_value + 9) // 10) * 10
        if max_value <= 500:
            return ((max_value + 99) // 100) * 100
        if max_value <= 2000:
            return ((max_value + 199) // 200) * 200
        return ((max_value + 999) // 1000) * 1000

    def _format_axis_label(self, value: int) -> str:
        if value == 0:
            return "0"
        if value >= 1000:
            return f"{value / 1000:g}k"
        return str(value)

    def _compact_label(self, label: str) -> str:
        if len(label) <= 4:
            return label
        return f"{label[:3]}…"
