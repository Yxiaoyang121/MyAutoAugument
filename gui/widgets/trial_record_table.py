from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem

from gui.widgets.common import configure_table


class TrialRecordTable(QTableWidget):
    def __init__(self) -> None:
        super().__init__(0, 6)
        self.setObjectName("trialRecordTable")
        self.setHorizontalHeaderLabels(["任务", "Score", "mAP50", "mAP50-95", "状态", "时间"])
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        configure_table(
            self,
            stretch_columns=(5,),
            fixed_columns={0: 90, 1: 92, 2: 92, 3: 112, 4: 110},
            row_height=30,
        )

    def set_records(self, records: list[dict]) -> None:
        self.setRowCount(len(records))
        for row, record in enumerate(records):
            metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            values = [
                record.get("task", record.get("trial", record.get("trial_index", "-"))),
                self._fmt(record.get("score", metrics.get("final_score", metrics.get("proxy_score", "-")))),
                self._fmt(record.get("map50", record.get("mAP50", metrics.get("yolo_map50", "-")))),
                self._fmt(record.get("map5095", record.get("mAP50_95", metrics.get("yolo_map50_95", "-")))),
                self._status_text(record),
                record.get("time", record.get("created_at", "-")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if values[4] == "运行中":
                    item.setBackground(QColor("#ECFDF5"))
                if column == 4:
                    color = {
                        "已接受": "#0F766E",
                        "运行中": "#2563EB",
                        "等待中": "#64748B",
                        "失败": "#EF4444",
                        "拒绝": "#EF4444",
                        "完成": "#0F766E",
                    }.get(str(value), "#334155")
                    item.setForeground(QColor(color))
                self.setItem(row, column, item)

    def _fmt(self, value) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.3f}"
        return str(value)

    def _status_text(self, record: dict) -> str:
        if "status" in record:
            return str(record.get("status") or "-")
        if record.get("accepted") is True:
            return "已接受"
        if record.get("accepted") is False:
            return "拒绝"
        return "-"
