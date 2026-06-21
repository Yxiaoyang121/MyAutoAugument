from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class MetricChartWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self._metrics: dict[str, list[float | int | None]] = {}

    def set_metrics(self, metrics: dict) -> None:
        self._metrics = {
            "mAP50": self._coerce(metrics.get("mAP50", [])),
            "mAP50-95": self._coerce(
                metrics.get("mAP50_95", metrics.get("mAP50-95", []))
            ),
            "score": self._coerce(metrics.get("score", [])),
            "bbox_retention": self._coerce(metrics.get("bbox_retention", [])),
            "bbox_valid": self._coerce(metrics.get("bbox_valid_rate", [])),
        }
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        chart = self.rect().adjusted(48, 22, -18, -44)
        if chart.width() <= 0 or chart.height() <= 0:
            return

        series = {name: values for name, values in self._metrics.items() if values}
        if not series:
            painter.setPen(QColor("#65717a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "未加载指标")
            return

        values = [value for vals in series.values() for value in vals if value is not None]
        if not values:
            painter.setPen(QColor("#65717a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "未加载指标")
            return

        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            min_value = max(0.0, min_value - 0.05)
            max_value = min(1.0, max_value + 0.05)

        self._draw_grid(painter, QRectF(chart), min_value, max_value)

        colors = {
            "mAP50": QColor("#2563EB"),
            "mAP50-95": QColor("#0F766E"),
            "score": QColor("#F59E0B"),
            "bbox_retention": QColor("#10B981"),
            "bbox_valid": QColor("#EF4444"),
        }
        for name, vals in series.items():
            self._draw_series(
                painter,
                QRectF(chart),
                vals,
                min_value,
                max_value,
                colors.get(name, QColor("#30343b")),
            )
        self._draw_legend(painter, colors)

    def _draw_grid(
        self, painter: QPainter, chart: QRectF, min_value: float, max_value: float
    ) -> None:
        painter.setPen(QPen(QColor("#dfe5e8"), 1))
        for i in range(5):
            y = chart.top() + chart.height() * i / 4
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))
            label_value = max_value - (max_value - min_value) * i / 4
            painter.setPen(QColor("#65717a"))
            painter.drawText(
                QRectF(2, y - 10, chart.left() - 8, 20),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{label_value:.2f}",
            )
            painter.setPen(QPen(QColor("#dfe5e8"), 1))

        painter.setPen(QPen(QColor("#aab5bb"), 1))
        painter.drawLine(chart.bottomLeft(), chart.bottomRight())
        painter.drawLine(chart.bottomLeft(), chart.topLeft())

    def _draw_series(
        self,
        painter: QPainter,
        chart: QRectF,
        values: list[float | None],
        min_value: float,
        max_value: float,
        color: QColor,
    ) -> None:
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            return

        count = len(values)
        x_step = chart.width() / max(count - 1, 1)
        points = QPolygonF()
        for index, value in enumerate(values):
            if value is None:
                continue
            x = chart.left() + index * x_step
            ratio = (value - min_value) / (max_value - min_value)
            y = chart.bottom() - ratio * chart.height()
            points.append(QPointF(x, y))

        painter.setPen(QPen(color, 2.5))
        painter.drawPolyline(points)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        for point in points:
            painter.drawEllipse(point, 3.5, 3.5)

    def _draw_legend(self, painter: QPainter, colors: dict[str, QColor]) -> None:
        painter.setFont(QFont("Microsoft YaHei", 9))
        x = 48
        y = self.height() - 24
        for name, color in colors.items():
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(x, y, 12, 8), 2, 2)
            painter.setPen(QColor("#30343b"))
            painter.drawText(QRectF(x + 18, y - 5, 78, 18), name)
            x += 100

    def _coerce(self, values: Iterable) -> list[float | None]:
        coerced: list[float | None] = []
        for value in values or []:
            if value is None:
                coerced.append(None)
                continue
            try:
                coerced.append(float(value))
            except (TypeError, ValueError):
                coerced.append(None)
        return coerced
