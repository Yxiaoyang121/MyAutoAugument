from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget


class AnalysisLineChart(QWidget):
    def __init__(self, color: str, series_name: str, y_max: float = 1.0, suffix: str = "") -> None:
        super().__init__()
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.color = QColor(color)
        self.series_name = series_name
        self.y_max = y_max
        self.suffix = suffix
        self.values: list[float] = []
        self.x_labels: list[int] = []

    def set_values(self, x_labels: list[int], values: list[float]) -> None:
        self.x_labels = x_labels
        self.values = values
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        chart = QRectF(self.rect().adjusted(34, 22, -18, -34))
        if chart.width() < 80 or chart.height() < 80:
            return

        self._draw_grid(painter, chart)
        if not self.values:
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "暂无指标曲线")
            return
        self._draw_series(painter, chart)
        self._draw_legend(painter, chart)
        self._draw_best_point(painter, chart)

    def _draw_grid(self, painter: QPainter, chart: QRectF) -> None:
        painter.setFont(QFont("Microsoft YaHei", 8))
        for tick in [self.y_max, self.y_max / 2, 0.0]:
            y = chart.bottom() - (tick / self.y_max if self.y_max else 0) * chart.height()
            painter.setPen(QPen(QColor("#E2E8F0"), 1))
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))
            painter.setPen(QColor("#334155"))
            label = f"{int(tick)}%" if self.suffix == "%" else f"{tick:g}"
            painter.drawText(QRectF(0, y - 10, chart.left() - 8, 20), Qt.AlignmentFlag.AlignRight, label)

        painter.setPen(QPen(QColor("#CBD5E1"), 1))
        painter.drawLine(chart.bottomLeft(), chart.bottomRight())
        painter.drawLine(chart.bottomLeft(), chart.topLeft())
        painter.setPen(QColor("#334155"))
        for tick in [0, 5, 10, 15, 20, 24]:
            x = chart.left() + chart.width() * tick / 24
            painter.drawText(QRectF(x - 14, chart.bottom() + 6, 28, 18), Qt.AlignmentFlag.AlignCenter, str(tick))
        painter.drawText(
            QRectF(chart.left(), self.height() - 20, chart.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            "Trial",
        )

    def _points(self, chart: QRectF) -> list[QPointF]:
        points: list[QPointF] = []
        step = chart.width() / max(len(self.values) - 1, 1)
        for index, raw in enumerate(self.values):
            value = max(0.0, min(float(raw), self.y_max))
            points.append(QPointF(chart.left() + index * step, chart.bottom() - value / self.y_max * chart.height()))
        return points

    def _draw_series(self, painter: QPainter, chart: QRectF) -> None:
        points = self._points(chart)
        if not points:
            return

        fill_path = QPainterPath(points[0])
        for point in points[1:]:
            fill_path.lineTo(point)
        fill_path.lineTo(points[-1].x(), chart.bottom())
        fill_path.lineTo(points[0].x(), chart.bottom())
        fill_path.closeSubpath()

        gradient = QLinearGradient(chart.topLeft(), chart.bottomLeft())
        area_color = QColor(self.color)
        area_color.setAlpha(38)
        transparent = QColor(self.color)
        transparent.setAlpha(0)
        gradient.setColorAt(0, area_color)
        gradient.setColorAt(1, transparent)
        painter.fillPath(fill_path, gradient)

        line_path = QPainterPath(points[0])
        for point in points[1:]:
            line_path.lineTo(point)
        painter.setPen(QPen(self.color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(line_path)
        painter.setBrush(self.color)
        painter.setPen(Qt.PenStyle.NoPen)
        for point in points:
            painter.drawEllipse(point, 2.6, 2.6)

    def _draw_legend(self, painter: QPainter, chart: QRectF) -> None:
        painter.setFont(QFont("Microsoft YaHei", 8))
        x = chart.right() - 82
        y = chart.top() - 20
        painter.setPen(QPen(self.color, 2.2))
        painter.drawLine(QPointF(x, y + 8), QPointF(x + 16, y + 8))
        painter.setBrush(self.color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(x + 8, y + 8), 2.6, 2.6)
        painter.setPen(QColor("#334155"))
        painter.drawText(QRectF(x + 22, y, 70, 18), self.series_name)

    def _draw_best_point(self, painter: QPainter, chart: QRectF) -> None:
        if not self.values:
            return
        best_value = max(self.values)
        best_index = self.values.index(best_value)
        point = self._points(chart)[best_index]

        painter.setPen(QPen(self.color, 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(point.x(), point.y()), QPointF(point.x(), chart.bottom()))
        painter.setPen(QPen(QColor("#FFFFFF"), 2))
        painter.setBrush(self.color)
        painter.drawEllipse(point, 5.6, 5.6)

        label = f"{best_value:.1f}%" if self.suffix == "%" else f"{best_value:.3f}"
        bubble = QRectF(point.x() - 22, max(chart.top() - 4, point.y() - 30), 44, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.color)
        painter.drawRoundedRect(bubble, 10, 10)
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Microsoft YaHei", 8, QFont.Weight.Bold))
        painter.drawText(bubble, Qt.AlignmentFlag.AlignCenter, label)


class AnalysisChartCard(QFrame):
    def __init__(self, title: str, color: str, series_name: str, y_max: float = 1.0, suffix: str = "") -> None:
        super().__init__()
        self.setObjectName("analysisChartCard")
        self.setMinimumHeight(170)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)

        title_label = QLabel(title)
        title_label.setObjectName("analysisChartTitle")
        self.chart = AnalysisLineChart(color, series_name, y_max, suffix)
        layout.addWidget(title_label)
        layout.addWidget(self.chart, 1)

    def set_values(self, x_labels: list[int], values: list[float]) -> None:
        self.chart.set_values(x_labels, values)


class AnalysisChartGrid(QWidget):
    def __init__(
        self,
        trial_ids: list[int],
        map50: list[float],
        map5095: list[float],
        score: list[float],
        bbox_retention: list[float],
    ) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.cards = [
            AnalysisChartCard("mAP50 趋势", "#0F9F7A", "mAP50"),
            AnalysisChartCard("mAP50-95 趋势", "#2563EB", "mAP50-95"),
            AnalysisChartCard("Score 趋势", "#6D4CD7", "Score"),
            AnalysisChartCard("bbox 保留率趋势", "#F59E0B", "bbox 保留率", 100.0, "%"),
        ]
        self.set_data(trial_ids, map50, map5095, score, bbox_retention)
        for card in self.cards:
            layout.addWidget(card, 1)

    def set_data(
        self,
        trial_ids: list[int],
        map50: list[float],
        map5095: list[float],
        score: list[float],
        bbox_retention: list[float],
    ) -> None:
        values = [map50, map5095, score, bbox_retention]
        for card, series_values in zip(self.cards, values, strict=True):
            card.set_values(trial_ids, series_values)
