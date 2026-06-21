from __future__ import annotations

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from gui.widgets.section_card import SectionCard


class QualityMetricRow(QFrame):
    def __init__(self, label: str, value: str = "-", tone: str = "neutral") -> None:
        super().__init__()
        self.setObjectName("qualityMetricRow")
        self.setFixedHeight(22)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)
        self.label_widget = QLabel(label)
        self.label_widget.setObjectName("qualityMetricLabel")
        self.value_widget = QLabel(value)
        self.value_widget.setObjectName("qualityMetricValue")
        self.value_widget.setProperty("tone", tone)
        layout.addWidget(self.label_widget)
        layout.addStretch(1)
        layout.addWidget(self.value_widget)

    def set_value(self, value: str, tone: str = "neutral") -> None:
        self.value_widget.setText(value)
        self.value_widget.setProperty("tone", tone)
        self.value_widget.style().unpolish(self.value_widget)
        self.value_widget.style().polish(self.value_widget)


class PreviewQualityPanel(SectionCard):
    export_requested = Signal()

    def __init__(self) -> None:
        super().__init__("增强质量指标")
        self.setMinimumWidth(220)
        self.outer_layout.setContentsMargins(12, 8, 12, 10)
        self.outer_layout.setSpacing(6)
        self.body.setSpacing(5)

        self.original_bbox = QualityMetricRow("原始 bbox 总数")
        self.augmented_bbox = QualityMetricRow("增强后 bbox 总数")
        self.retention = QualityMetricRow("bbox 保留率")
        self.valid_rate = QualityMetricRow("bbox 有效率")
        self.pixel_delta = QualityMetricRow("平均像素变化（%）")
        self.invalid_bbox = QualityMetricRow("无效 bbox 数")
        metrics = [
            self.original_bbox,
            self.augmented_bbox,
            self.retention,
            self.valid_rate,
            self.pixel_delta,
            self.invalid_bbox,
        ]
        for metric in metrics:
            self.body.addWidget(metric)

        self.status = QLabel("◎  增强质量：优秀")
        self.status.setObjectName("previewQualityStatus")
        self.status.setProperty("tone", "success")
        self.status.setFixedHeight(28)
        self.body.addWidget(self.status)
        self.body.addStretch(1)

        self.export_button = QPushButton("☁  导出增强图像")
        self.export_button.setObjectName("secondaryButton")
        self.export_button.setFixedHeight(30)
        self.export_button.clicked.connect(self.export_requested.emit)
        self.body.addWidget(self.export_button)

    def sizeHint(self) -> QSize:
        return QSize(230, 280)

    def minimumSizeHint(self) -> QSize:
        return QSize(210, 250)

    def set_metrics(
        self,
        *,
        original_total: int,
        augmented_total: int,
        retention: float,
        valid_rate: float,
        pixel_delta: str,
        invalid_count: int,
    ) -> None:
        self.original_bbox.set_value(str(original_total))
        self.augmented_bbox.set_value(str(augmented_total))
        self.retention.set_value(f"{retention:.2%}", "success" if retention >= 0.95 else "warning")
        self.valid_rate.set_value(f"{valid_rate:.2%}", "success" if valid_rate >= 0.99 else "warning")
        self.pixel_delta.set_value(pixel_delta, "blue")
        self.invalid_bbox.set_value(str(invalid_count), "success" if invalid_count == 0 else "error")
        ok = invalid_count == 0 and valid_rate >= 0.99
        self.status.setText("◎  增强质量：优秀" if ok else "△  增强质量：需复核")
        self.status.setProperty("tone", "success" if ok else "warning")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
