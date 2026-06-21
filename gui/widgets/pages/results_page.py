from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from gui.services import ResultLoader
from gui.widgets.chart_widget import MetricChartWidget
from gui.widgets.common import Section, configure_table


class ResultsPage(QWidget):
    def __init__(self, project_root: Path, result_loader: ResultLoader) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.result_loader = result_loader
        self.data: dict[str, Any] = {}
        self._build_ui()
        self.set_output_dir(str(self.project_root / "outputs"))

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(16)

        title = QLabel("结果")
        title.setObjectName("pageTitle")
        subtitle = QLabel("读取真实输出目录中的 summary、trials、policies、metrics 和 diagnosis JSON。")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = Section("结果来源")
        row = QHBoxLayout()
        self.output_dir = QLineEdit()
        choose = QPushButton("浏览")
        reload_button = QPushButton("重新加载")
        choose.clicked.connect(self._choose_output)
        reload_button.clicked.connect(self.reload)
        row.addWidget(self.output_dir, 1)
        row.addWidget(choose)
        row.addWidget(reload_button)
        controls.layout.addLayout(row)
        layout.addWidget(controls)

        grid = QGridLayout()
        grid.setSpacing(14)

        trial_section = Section("试验")
        self.trial_table = QTableWidget(0, 8)
        self.trial_table.setHorizontalHeaderLabels(["试验", "mAP50", "mAP50-95", "分数", "bbox 保留率", "bbox 有效率", "多样性", "已接受"])
        configure_table(
            self.trial_table,
            stretch_columns=(7,),
            fixed_columns={0: 64, 1: 82, 2: 96, 3: 82, 4: 112, 5: 112, 6: 88},
            min_column_widths={7: 88},
        )
        self.trial_table.itemSelectionChanged.connect(self._show_selected_trial)
        trial_section.layout.addWidget(self.trial_table)

        chart_section = Section("指标曲线")
        self.chart = MetricChartWidget()
        chart_section.layout.addWidget(self.chart)

        detail_section = Section("选中试验的策略 / 指标 / 诊断")
        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        detail_section.layout.addWidget(self.detail_text)

        status_section = Section("加载状态")
        self.status_text = QPlainTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(140)
        status_section.layout.addWidget(self.status_text)

        grid.addWidget(trial_section, 0, 0)
        grid.addWidget(chart_section, 0, 1)
        grid.addWidget(detail_section, 1, 0)
        grid.addWidget(status_section, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)

    def set_output_dir(self, path: str) -> None:
        self.output_dir.setText(path)
        self.reload()

    def reload(self) -> None:
        path = self.output_dir.text().strip()
        self.data = self.result_loader.load(path)
        self._populate_trials()
        self.chart.set_metrics(self.data.get("metrics", {}))
        self._populate_status()
        self._show_selected_trial()

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择结果目录", self.output_dir.text())
        if path:
            self.output_dir.setText(path)
            self.reload()

    def _populate_trials(self) -> None:
        trials = self.data.get("trials", [])
        self.trial_table.setRowCount(len(trials))
        for row, record in enumerate(trials):
            metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            values = [
                record.get("trial_index", record.get("trial", row)),
                self._fmt(metrics.get("yolo_map50", record.get("mAP50"))),
                self._fmt(metrics.get("yolo_map50_95", record.get("mAP50_95"))),
                self._fmt(record.get("score", metrics.get("final_score", metrics.get("proxy_score")))),
                self._fmt(metrics.get("bbox_retention_raw", metrics.get("bbox_retention"))),
                self._fmt(metrics.get("bbox_valid_rate")),
                self._fmt(metrics.get("diversity_score")),
                self._fmt_status(record.get("accepted", record.get("status", "-"))),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.trial_table.setItem(row, column, item)

    def _show_selected_trial(self) -> None:
        trials = self.data.get("trials", [])
        if not trials:
            self.detail_text.setPlainText(json.dumps({key: self.data.get(key) for key in ["summary", "policy", "metrics", "diagnosis"]}, ensure_ascii=False, indent=2))
            return
        row = self.trial_table.currentRow()
        if row < 0:
            row = 0
        if row >= len(trials):
            return
        record = trials[row]
        payload = {
            "trial_dir": record.get("trial_dir"),
            "policy": record.get("policy"),
            "metrics": record.get("metrics"),
            "diagnosis": record.get("diagnosis"),
        }
        self.detail_text.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))

    def _populate_status(self) -> None:
        lines = [
            f"输出目录: {self.data.get('output_dir')}",
            f"已加载试验数: {len(self.data.get('trials', []))}",
            f"缺失文件: {', '.join(self.data.get('missing_files', [])) or '-'}",
        ]
        errors = self.data.get("read_errors", [])
        if errors:
            lines.append("读取错误:")
            lines.extend(f"- {item}" for item in errors)
        best = self.data.get("best_trial") or {}
        if best:
            lines.append(f"最佳试验: {best.get('trial_index', best.get('trial'))}, 分数={best.get('score')}")
        self.status_text.setPlainText("\n".join(lines))

    def _fmt(self, value: Any) -> str:
        try:
            if value is None or value == "":
                return "-"
            if isinstance(value, bool):
                return "是" if value else "否"
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def _fmt_status(self, value: Any) -> str:
        if isinstance(value, bool):
            return "已接受" if value else "未接受"
        return "-" if value is None else str(value)
