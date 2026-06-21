from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from gui.services import ResultLoader
from gui.widgets.common import Section, configure_table


ISSUE_CATEGORIES = {
    "小目标": {"small_object_fn_high"},
    "弱定位": {"weak_localization_gap", "position_fn_gap", "metric_map50_95_gap"},
    "类别问题": {"class_recall_imbalance", "class_precision_imbalance"},
    "过暗": {"exposure_fn_high", "false_negatives_exposure_hint", "false_positives_exposure_hint"},
    "过亮": {"exposure_fn_high", "false_negatives_exposure_hint", "false_positives_exposure_hint"},
    "低对比": {"low_contrast_fn_high", "false_negatives_low_contrast_hint", "false_positives_low_contrast_hint"},
    "bbox 无效": {"bbox_invalid", "bbox_retention_low"},
    "样本不足": {"stable_validation_keep_light_policy"},
}


class DiagnosticsPage(QWidget):
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

        title = QLabel("诊断")
        title.setObjectName("pageTitle")
        subtitle = QLabel("按实际失败模式展示后端 diagnosis.json 和增强建议输出。")
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = Section("诊断来源")
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

        summary = Section("问题分类")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["类别", "问题", "严重程度", "建议方向"])
        configure_table(
            self.table,
            stretch_columns=(1, 3),
            fixed_columns={0: 96, 2: 92},
            min_column_widths={1: 180, 3: 260},
        )
        summary.layout.addWidget(self.table)
        layout.addWidget(summary, 1)

        raw = Section("后端原始诊断")
        self.raw_text = QPlainTextEdit()
        self.raw_text.setReadOnly(True)
        raw.layout.addWidget(self.raw_text)
        layout.addWidget(raw, 1)

    def set_output_dir(self, path: str) -> None:
        self.output_dir.setText(path)
        self.reload()

    def reload(self) -> None:
        self.data = self.result_loader.load(self.output_dir.text().strip())
        diagnosis = self._diagnosis_payload()
        rows = self._category_rows(diagnosis)
        self.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.table.setItem(row, column, item)
        self.raw_text.setPlainText(json.dumps(diagnosis or {"status": "未找到诊断结果"}, ensure_ascii=False, indent=2))

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir.text())
        if path:
            self.output_dir.setText(path)
            self.reload()

    def _diagnosis_payload(self) -> dict[str, Any]:
        diagnoses = self.data.get("diagnoses", [])
        if diagnoses:
            payload = diagnoses[-1].get("diagnosis", {})
            if isinstance(payload, dict):
                return payload
        payload = self.data.get("diagnosis", {})
        return payload if isinstance(payload, dict) else {}

    def _category_rows(self, diagnosis: dict[str, Any]) -> list[list[str]]:
        issues = diagnosis.get("main_issues") or diagnosis.get("issues") or []
        rows: list[list[str]] = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            name = str(issue.get("issue", "unknown"))
            category = self._category_for_issue(name)
            recommendations = ", ".join(str(item) for item in issue.get("recommendations", []))
            rows.append([category, name, str(issue.get("severity", "-")), recommendations or "-"])
        if not rows:
            advisor = diagnosis.get("advisor_search_space", {}) if isinstance(diagnosis, dict) else {}
            comments = advisor.get("comments", []) if isinstance(advisor, dict) else []
            if comments:
                rows.append(["样本不足", "advisor_comments", "-", "; ".join(str(item) for item in comments)])
        return rows

    def _category_for_issue(self, issue: str) -> str:
        for category, names in ISSUE_CATEGORIES.items():
            if issue in names or any(token in issue for token in names):
                return category
        if "contrast" in issue:
            return "低对比"
        if "exposure" in issue:
            return "过暗/过亮"
        if "class" in issue:
            return "类别问题"
        if "localization" in issue or "position" in issue:
            return "弱定位"
        return "其他"
