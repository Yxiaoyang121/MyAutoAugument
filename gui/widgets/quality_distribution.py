from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

from gui.widgets.section_card import SectionCard


class QualityDistributionCard(QFrame):
    def __init__(self, title: str, value: float, minimum: float = 0.0, maximum: float = 100.0) -> None:
        super().__init__()
        self.setObjectName("qualityDistributionCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 6, 9, 6)
        layout.setSpacing(3)

        label = QLabel(title)
        label.setObjectName("qualityDistTitle")
        value_label = QLabel(f"{value:.1f}")
        value_label.setObjectName("qualityDistValue")
        progress = QProgressBar()
        progress.setObjectName("qualityDistProgress")
        progress.setTextVisible(False)
        progress.setRange(0, 1000)
        if maximum == minimum:
            normalized = 0.0
        else:
            normalized = (value - minimum) / (maximum - minimum)
        progress.setValue(int(max(0.0, min(1.0, normalized)) * 1000))

        layout.addWidget(label)
        layout.addWidget(value_label)
        layout.addWidget(progress)

    def _fmt(self, value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.1f}"


class QualityDistributionPanel(SectionCard):
    def __init__(self) -> None:
        super().__init__("增强质量分布（可选）")
        self.outer_layout.setContentsMargins(14, 8, 14, 10)
        self.outer_layout.setSpacing(6)
        self.body.setSpacing(6)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        for title, value, minimum, maximum in [
            ("亮度变化（%）", 18.6, 0, 100),
            ("对比度变化（%）", 12.4, 0, 100),
            ("像素变化（%）", 22.9, 0, 100),
            ("几何变换角度（°）", -12.7, -90, 90),
            ("有效 bbox 比例（%）", 100.0, 0, 100),
        ]:
            row.addWidget(QualityDistributionCard(title, value, minimum, maximum), 1)
        self.body.addLayout(row)
