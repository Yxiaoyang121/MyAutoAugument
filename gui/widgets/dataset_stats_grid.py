from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class DatasetStatsGrid(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._cells: dict[str, tuple[QLabel, QLabel]] = {}
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)
        for index, label in enumerate(["图像总数", "标签总数", "类别数", "空标签数", "缺失标签数", "异常图像数"]):
            cell = QFrame()
            cell.setObjectName("datasetStatsCell")
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(10, 10, 10, 10)
            cell_layout.setSpacing(4)
            title = QLabel(label)
            title.setObjectName("datasetStatsTitle")
            value = QLabel("0")
            value.setObjectName("datasetStatsValue")
            cell_layout.addWidget(title)
            cell_layout.addWidget(value)
            layout.addWidget(cell, index // 3, index % 3)
            self._cells[label] = (title, value)

    def set_stats(self, stats: dict[str, int]) -> None:
        for label, (_, value_label) in self._cells.items():
            value = int(stats.get(label, 0) or 0)
            value_label.setText(f"{value:,}")
