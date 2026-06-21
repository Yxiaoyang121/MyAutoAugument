from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.controllers import AnalysisController
from gui.services import ResultLoader
from gui.services.app_state import AppState
from gui.widgets.analysis_chart_grid import AnalysisChartGrid
from gui.widgets.analysis_log_panel import AnalysisLogPanel
from gui.widgets.analysis_summary_card import AnalysisSummaryCard
from gui.widgets.analysis_trial_table import AnalysisTrialTable
from gui.widgets.diagnosis_issue_card import DiagnosisIssueCard
from gui.widgets.diagnosis_summary_panel import DiagnosisSummaryPanel
from gui.widgets.json_tab_viewer import JsonTabViewer
from gui.widgets.section_card import SectionCard
from gui.widgets.strategy_detail_panel import StrategyDetailPanel


MOCK_RESULT_SOURCE = r"D:\AutoAugment\outputs\run_20260616_143022"

ANALYSIS_SUMMARY = {
    "total_trials": 24,
    "best_trial": 6,
    "best_score": 0.934,
    "best_map50": 0.934,
    "best_map5095": 0.712,
    "best_policy": "gui_policy",
    "start_time": "2026-06-16 14:15:22",
    "end_time": "2026-06-16 14:30:10",
}

TRIAL_IDS = list(range(1, 25))
MAP50_VALUES = [
    0.712,
    0.689,
    0.845,
    0.812,
    0.901,
    0.934,
    0.756,
    0.887,
    0.832,
    0.818,
    0.821,
    0.799,
    0.842,
    0.861,
    0.874,
    0.868,
    0.883,
    0.891,
    0.879,
    0.894,
    0.901,
    0.912,
    0.904,
    0.918,
]
MAP5095_VALUES = [
    0.515,
    0.498,
    0.638,
    0.612,
    0.691,
    0.712,
    0.548,
    0.668,
    0.621,
    0.606,
    0.615,
    0.594,
    0.629,
    0.642,
    0.655,
    0.649,
    0.662,
    0.671,
    0.658,
    0.676,
    0.682,
    0.695,
    0.687,
    0.701,
]
SCORE_VALUES = MAP50_VALUES
BBOX_RETENTION_VALUES = [
    95.4,
    94.2,
    98.7,
    98.3,
    100.0,
    100.0,
    96.1,
    100.0,
    98.8,
    97.9,
    99.1,
    98.6,
    99.0,
    99.3,
    98.9,
    99.5,
    100.0,
    99.4,
    99.6,
    99.1,
    99.7,
    100.0,
    99.2,
    99.8,
]

TRIAL_ROWS = [
    {"trial": 6, "score": 0.934, "map50": 0.934, "map5095": 0.712, "bbox_retention": "100.0%", "status": "最优", "time": "06-16 14:22:48"},
    {"trial": 5, "score": 0.901, "map50": 0.901, "map5095": 0.691, "bbox_retention": "100.0%", "status": "已接受", "time": "06-16 14:21:35"},
    {"trial": 8, "score": 0.887, "map50": 0.887, "map5095": 0.668, "bbox_retention": "100.0%", "status": "已接受", "time": "06-16 14:24:01"},
    {"trial": 3, "score": 0.845, "map50": 0.845, "map5095": 0.638, "bbox_retention": "98.7%", "status": "已接受", "time": "06-16 14:19:12"},
    {"trial": 4, "score": 0.812, "map50": 0.812, "map5095": 0.612, "bbox_retention": "98.3%", "status": "已接受", "time": "06-16 14:20:22"},
    {"trial": 7, "score": 0.756, "map50": 0.756, "map5095": 0.548, "bbox_retention": "96.1%", "status": "拒绝", "time": "06-16 14:23:29"},
    {"trial": 1, "score": 0.712, "map50": 0.712, "map5095": 0.515, "bbox_retention": "95.4%", "status": "拒绝", "time": "06-16 14:16:45"},
    {"trial": 2, "score": 0.689, "map50": 0.689, "map5095": 0.498, "bbox_retention": "94.2%", "status": "拒绝", "time": "06-16 14:17:58"},
]

DIAGNOSIS_COUNTS = {"严重问题": 2, "中等问题": 3, "轻微问题": 4, "提示": 5}

DIAGNOSIS_ISSUES = [
    {
        "category": "小目标问题",
        "title": "小目标召回不足",
        "severity": "严重",
        "description": "小目标在增强后 FN 增加，导致 Recall 降低。",
        "evidence": "小目标 Recall: 0.62（低于阈值 0.75）",
        "reason": "旋转 / 缩放导致小目标 bbox 太小或被过滤。",
        "suggestion": "增加 crop、scale、brightness 扰动，减弱强旋转角度。",
        "adjustment": "rotate.max_angle: 15 -> 10；crop.prob: 0.5 -> 0.2；brightness.prob: 0.5 -> 0.6",
    },
    {
        "category": "增强过强",
        "title": "过度平滑导致特征丢失",
        "severity": "中等",
        "description": "Blur 操作强度过高，模糊样本比例 32%。",
        "evidence": "Blur 增强后边缘响应下降。",
        "reason": "模糊算子强度或概率偏高。",
        "suggestion": "降低 blur 强度，或降低应用概率。",
        "adjustment": "blur.strength: 0.6 -> 0.3；blur.prob: 0.5 -> 0.2",
    },
    {
        "category": "类别不均衡",
        "title": "部分类别样本分布不均衡",
        "severity": "中等",
        "description": "类别 7 样本较少，容易导致过拟合。",
        "evidence": "类别 7 验证样本数偏低。",
        "reason": "原始数据采集分布不均，少数类增强不足。",
        "suggestion": "对少数类增加采样或增强。",
        "adjustment": "minor_class_sampling: 1.0 -> 1.5；brightness.prob: 0.5 -> 0.6",
    },
]

BEST_POLICY = {
    "name": "gui_policy",
    "trial": 6,
    "score": 0.934,
    "map50": 0.934,
    "map5095": 0.712,
    "operations": [
        {"name": "brightness", "prob": 0.5, "strength": 0.5, "bbox_effect": "不改变", "params": {"max_delta": 0.25}},
        {"name": "contrast", "prob": 0.5, "strength": 0.5, "bbox_effect": "不改变", "params": {"max_delta": 0.5}},
        {"name": "horizontal_flip", "prob": 0.5, "strength": 0.5, "bbox_effect": "不改变", "params": {}},
        {"name": "rotate", "prob": 0.5, "strength": 0.5, "bbox_effect": "会改变", "params": {"max_angle": 15.0}},
    ],
}

STDOUT_LOG = "\n".join(
    [
        "[14:28:10] > Experiment started.",
        r"[14:28:10] > Dataset: D:\AutoAugment\dataset",
        "[14:28:10] > Loaded 24 trial records.",
        "[14:28:11] > Best trial detected: #6.",
        "[14:28:12] > Diagnosis completed. Severe=2, Medium=3.",
    ]
)
STDERR_LOG = "[14:30:10] > No fatal error. Warnings were converted to diagnosis cards."


class AnalysisPage(QWidget):
    def __init__(self, project_root: Path, result_loader: ResultLoader, app_state: AppState | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.result_loader = result_loader
        self.controller = AnalysisController(self.project_root, app_state)
        self._using_mock = True
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("analysisPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        self._build_source_card(layout)
        self._build_summary_cards(layout)
        self.chart_grid = AnalysisChartGrid(TRIAL_IDS, MAP50_VALUES, MAP5095_VALUES, SCORE_VALUES, BBOX_RETENTION_VALUES)
        layout.addWidget(self.chart_grid)
        self._build_tabs(layout)
        layout.addStretch(1)

    def _build_source_card(self, parent: QVBoxLayout) -> None:
        card = SectionCard("")
        card.setObjectName("analysisSourceCard")
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)

        label = QLabel("结果来源")
        label.setObjectName("analysisSourceLabel")
        self.output_dir = QLineEdit(MOCK_RESULT_SOURCE)
        self.output_dir.setObjectName("analysisSourceInput")
        self.output_dir.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        browse = QPushButton("浏览")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(self._choose_output)
        reload_button = QPushButton("重新加载")
        reload_button.setObjectName("primaryButton")
        reload_button.clicked.connect(self.reload)

        row.addWidget(label)
        row.addWidget(self.output_dir, 1)
        row.addWidget(browse)
        row.addWidget(reload_button)
        card.body.addLayout(row)
        parent.addWidget(card)

    def _build_summary_cards(self, parent: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.summary_cards = {
            "total_trials": AnalysisSummaryCard("Total Trials", str(ANALYSIS_SUMMARY["total_trials"]), "⚗", "green"),
            "best_trial": AnalysisSummaryCard("Best Trial", str(ANALYSIS_SUMMARY["best_trial"]), "♕", "blue"),
            "best_score": AnalysisSummaryCard("Best Score", f"{ANALYSIS_SUMMARY['best_score']:.3f}", "☆", "purple"),
            "best_map50": AnalysisSummaryCard("Best mAP50", f"{ANALYSIS_SUMMARY['best_map50']:.3f}", "◎", "green"),
            "best_map5095": AnalysisSummaryCard("Best mAP50-95", f"{ANALYSIS_SUMMARY['best_map5095']:.3f}", "▥", "blue"),
            "best_policy": AnalysisSummaryCard("当前策略", ANALYSIS_SUMMARY["best_policy"], "</>", "orange"),
            "time": AnalysisSummaryCard(
                "实验时间",
                f"{ANALYSIS_SUMMARY['start_time']}\n~ {ANALYSIS_SUMMARY['end_time']}",
                "◷",
                "neutral",
            ),
        }
        for card in self.summary_cards.values():
            row.addWidget(card, 1)
        parent.addLayout(row)

    def _build_tabs(self, parent: QVBoxLayout) -> None:
        card = SectionCard("")
        card.setObjectName("analysisTabsCard")
        self.tabs = QTabWidget()
        self.tabs.setObjectName("analysisTabs")
        self._set_tabs(TRIAL_ROWS, DIAGNOSIS_COUNTS, DIAGNOSIS_ISSUES, BEST_POLICY, self._json_payloads())
        card.body.addWidget(self.tabs, 1)
        parent.addWidget(card, 1)

    def _set_tabs(
        self,
        trial_rows: list[dict],
        diagnosis_counts: dict[str, int],
        diagnosis_issues: list[dict],
        best_policy: dict,
        json_payloads: dict[str, object],
    ) -> None:
        current = self.tabs.currentIndex() if hasattr(self, "tabs") else 0
        while self.tabs.count():
            widget = self.tabs.widget(0)
            self.tabs.removeTab(0)
            widget.deleteLater()
        self.tabs.addTab(self._trial_tab(trial_rows, diagnosis_counts, diagnosis_issues), "试验表格")
        self.tabs.addTab(StrategyDetailPanel(best_policy), "策略详情")
        self.tabs.addTab(self._diagnosis_tab(diagnosis_issues), "诊断建议")
        self.tabs.addTab(AnalysisLogPanel(json_payloads, STDOUT_LOG, STDERR_LOG), "日志文件")
        self.tabs.addTab(JsonTabViewer(json_payloads), "JSON")
        self.tabs.setCurrentIndex(max(0, min(current, self.tabs.count() - 1)))

    def _trial_tab(self, trial_rows: list[dict], diagnosis_counts: dict[str, int], diagnosis_issues: list[dict]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(AnalysisTrialTable(trial_rows), 3)
        layout.addWidget(DiagnosisSummaryPanel(diagnosis_counts, diagnosis_issues), 2)
        return widget

    def _diagnosis_tab(self, issues: list[dict]) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        categories = QFrame()
        categories.setObjectName("diagnosisCategoryList")
        cat_layout = QVBoxLayout(categories)
        cat_layout.setContentsMargins(10, 10, 10, 10)
        cat_layout.setSpacing(6)
        title = QLabel("问题分类")
        title.setObjectName("analysisBlockTitle")
        cat_layout.addWidget(title)
        for index, name in enumerate(["定位问题", "小目标问题", "类别不均衡", "增强过强", "bbox 异常", "训练不稳定"]):
            item = QLabel(name)
            item.setObjectName("diagnosisCategoryItem")
            item.setProperty("active", index == 1)
            item.setAlignment(Qt.AlignmentFlag.AlignVCenter)
            cat_layout.addWidget(item)
        cat_layout.addStretch(1)

        detail = QVBoxLayout()
        detail.setContentsMargins(0, 0, 0, 0)
        detail.setSpacing(8)
        detail_title = QLabel("问题详情")
        detail_title.setObjectName("analysisBlockTitle")
        detail.addWidget(detail_title)
        for issue in issues:
            card = DiagnosisIssueCard(
                f"{issue['category']} / {issue['title']}",
                issue["severity"],
                issue["description"],
                issue["evidence"],
                f"{issue['suggestion']}  推荐策略调整：{issue['adjustment']}",
            )
            detail.addWidget(card)
        detail.addStretch(1)

        layout.addWidget(categories, 1)
        layout.addLayout(detail, 3)
        return widget

    def set_output_dir(self, path: str) -> None:
        self.output_dir.setText(path or MOCK_RESULT_SOURCE)
        if path and self.controller.has_results(path):
            self.reload()

    def reload(self) -> None:
        path = self.output_dir.text().strip()
        if not path or not self.controller.has_results(path):
            self._using_mock = True
            self.output_dir.setToolTip("未找到真实结果文件，当前显示内置演示数据。")
            self._populate_mock()
            return
        data = self.controller.load(path)
        self._using_mock = False
        self.output_dir.setToolTip("已加载真实实验结果。")
        self._populate_from_result(data)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择结果目录", self.output_dir.text())
        if path:
            self.set_output_dir(path)

    def _json_payloads(self) -> dict[str, object]:
        return {
            "summary.json": ANALYSIS_SUMMARY,
            "trial_record.json": TRIAL_ROWS,
            "diagnosis.json": {"counts": DIAGNOSIS_COUNTS, "issues": DIAGNOSIS_ISSUES},
            "policy.json": BEST_POLICY,
            "metrics.json": {
                "trial_ids": TRIAL_IDS,
                "map50": MAP50_VALUES,
                "map5095": MAP5095_VALUES,
                "score": SCORE_VALUES,
                "bbox_retention": BBOX_RETENTION_VALUES,
            },
        }

    def _populate_mock(self) -> None:
        self.summary_cards["total_trials"].set_value(str(ANALYSIS_SUMMARY["total_trials"]))
        self.summary_cards["best_trial"].set_value(str(ANALYSIS_SUMMARY["best_trial"]))
        self.summary_cards["best_score"].set_value(f"{ANALYSIS_SUMMARY['best_score']:.3f}")
        self.summary_cards["best_map50"].set_value(f"{ANALYSIS_SUMMARY['best_map50']:.3f}")
        self.summary_cards["best_map5095"].set_value(f"{ANALYSIS_SUMMARY['best_map5095']:.3f}")
        self.summary_cards["best_policy"].set_value(str(ANALYSIS_SUMMARY["best_policy"]))
        self.summary_cards["time"].set_value(f"{ANALYSIS_SUMMARY['start_time']}\n~ {ANALYSIS_SUMMARY['end_time']}")
        self.chart_grid.set_data(TRIAL_IDS, MAP50_VALUES, MAP5095_VALUES, SCORE_VALUES, BBOX_RETENTION_VALUES)
        self._set_tabs(TRIAL_ROWS, DIAGNOSIS_COUNTS, DIAGNOSIS_ISSUES, BEST_POLICY, self._json_payloads())

    def _populate_from_result(self, data: dict) -> None:
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        trials = data.get("trials") if isinstance(data.get("trials"), list) else []
        best = data.get("best_trial") if isinstance(data.get("best_trial"), dict) else {}
        policy = data.get("policy") if isinstance(data.get("policy"), dict) else {}
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}

        total = summary.get("total_trials", len(trials))
        best_trial = summary.get("best_trial", best.get("trial_index", best.get("trial", "-")))
        best_score = summary.get("best_score", best.get("score"))
        best_metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
        best_map50 = summary.get("best_map50", best_metrics.get("yolo_map50", best.get("mAP50")))
        best_map5095 = summary.get("best_map5095", best_metrics.get("yolo_map50_95", best.get("mAP50_95")))
        self.summary_cards["total_trials"].set_value(str(total))
        self.summary_cards["best_trial"].set_value(str(best_trial))
        self.summary_cards["best_score"].set_value(self._fmt(best_score))
        self.summary_cards["best_map50"].set_value(self._fmt(best_map50))
        self.summary_cards["best_map5095"].set_value(self._fmt(best_map5095))
        self.summary_cards["best_policy"].set_value(str(policy.get("name", summary.get("best_policy", "-"))))
        self.summary_cards["time"].set_value(str(summary.get("time_range", "-")))

        trial_ids = self._numeric_list(metrics.get("trial_index")) or list(range(1, len(trials) + 1))
        map50 = self._float_list(metrics.get("mAP50")) or self._trial_metric(trials, "yolo_map50", "mAP50")
        map5095 = self._float_list(metrics.get("mAP50_95")) or self._trial_metric(trials, "yolo_map50_95", "mAP50_95")
        score = self._float_list(metrics.get("score")) or [self._safe_float(row.get("score")) for row in trials]
        bbox = self._float_list(metrics.get("bbox_retention")) or self._trial_metric(trials, "bbox_retention_raw", "bbox_retention")
        bbox = [value * 100 if value is not None and value <= 1.5 else value for value in bbox]
        self.chart_grid.set_data(trial_ids, map50, map5095, score, bbox)

        trial_rows = self._trial_rows(trials)
        diagnosis_issues = self._diagnosis_issues(data)
        diagnosis_counts = self._diagnosis_counts(diagnosis_issues)
        best_policy = self._best_policy(policy, best, best_trial, best_score, best_map50, best_map5095)
        self._set_tabs(trial_rows, diagnosis_counts, diagnosis_issues or DIAGNOSIS_ISSUES, best_policy, self._result_payloads(data))

    def _trial_rows(self, trials: list[dict]) -> list[dict]:
        rows: list[dict] = []
        for row in trials:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            bbox = metrics.get("bbox_retention_raw", metrics.get("bbox_retention", row.get("bbox_retention", "-")))
            if isinstance(bbox, (int, float)):
                bbox_text = f"{bbox * 100:.1f}%" if bbox <= 1.5 else f"{bbox:.1f}%"
            else:
                bbox_text = str(bbox)
            status = "已接受" if row.get("accepted") is True else "拒绝" if row.get("accepted") is False else str(row.get("status", "-"))
            rows.append(
                {
                    "trial": row.get("trial_index", row.get("trial", "-")),
                    "score": self._safe_float(row.get("score")),
                    "map50": self._safe_float(metrics.get("yolo_map50", row.get("mAP50"))),
                    "map5095": self._safe_float(metrics.get("yolo_map50_95", row.get("mAP50_95"))),
                    "bbox_retention": bbox_text,
                    "status": status,
                    "time": str(row.get("time", row.get("created_at", "-"))),
                }
            )
        return rows

    def _diagnosis_issues(self, data: dict) -> list[dict]:
        diagnosis = data.get("diagnosis") if isinstance(data.get("diagnosis"), dict) else {}
        raw_issues = diagnosis.get("main_issues") or diagnosis.get("issues") or []
        issues: list[dict] = []
        for index, item in enumerate(raw_issues):
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "中等"))
            if severity.lower() in {"high", "error"}:
                severity = "严重"
            elif severity.lower() in {"medium", "warning"}:
                severity = "中等"
            elif severity.lower() in {"low", "info"}:
                severity = "轻微"
            issues.append(
                {
                    "category": str(item.get("category", "诊断问题")),
                    "title": str(item.get("title", item.get("issue", item.get("name", f"问题 {index + 1}")))),
                    "severity": severity,
                    "description": str(item.get("description", item.get("desc", "后端诊断返回的问题。"))),
                    "evidence": self._stringify(item.get("evidence", item.get("metric", "-"))),
                    "reason": self._stringify(item.get("reason", item.get("possible_reason", "-"))),
                    "suggestion": self._stringify(item.get("suggestion", item.get("recommendations", item.get("advice", "-")))),
                    "adjustment": self._stringify(item.get("adjustment", item.get("policy_adjustment", "-"))),
                }
            )
        return issues

    def _diagnosis_counts(self, issues: list[dict]) -> dict[str, int]:
        counts = {"严重问题": 0, "中等问题": 0, "轻微问题": 0, "提示": 0}
        for issue in issues:
            severity = issue.get("severity")
            if severity == "严重":
                counts["严重问题"] += 1
            elif severity == "中等":
                counts["中等问题"] += 1
            elif severity == "轻微":
                counts["轻微问题"] += 1
            else:
                counts["提示"] += 1
        return counts

    def _best_policy(self, policy: dict, best: dict, best_trial, best_score, best_map50, best_map5095) -> dict:
        operations = policy.get("operations") if isinstance(policy.get("operations"), list) else []
        return {
            "name": policy.get("name", "gui_policy"),
            "trial": best_trial,
            "score": self._safe_float(best_score),
            "map50": self._safe_float(best_map50),
            "map5095": self._safe_float(best_map5095),
            "operations": [
                {
                    "name": operation.get("name", "unknown"),
                    "prob": float(operation.get("prob", 1.0) or 1.0),
                    "strength": float(operation.get("strength", 1.0) or 1.0),
                    "bbox_effect": "会改变" if operation.get("name") in {"rotate", "scale", "translate", "shear", "vertical_flip"} else "不改变",
                    "params": dict(operation.get("params", {}) or {}),
                }
                for operation in operations
                if isinstance(operation, dict)
            ],
        }

    def _result_payloads(self, data: dict) -> dict[str, object]:
        return {
            "summary.json": data.get("summary", {}),
            "trial_record.json": {"trials": data.get("trials", [])},
            "diagnosis.json": data.get("diagnosis", {}),
            "policy.json": data.get("policy", {}),
            "metrics.json": data.get("metrics", {}),
        }

    def _trial_metric(self, trials: list[dict], metric_key: str, fallback_key: str) -> list[float]:
        values: list[float] = []
        for row in trials:
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            values.append(self._safe_float(metrics.get(metric_key, row.get(fallback_key))))
        return values

    def _float_list(self, values) -> list[float]:
        if not isinstance(values, list):
            return []
        return [self._safe_float(value) for value in values if value is not None]

    def _numeric_list(self, values) -> list[int]:
        if not isinstance(values, list):
            return []
        result = []
        for value in values:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                pass
        return result

    def _safe_float(self, value) -> float:
        try:
            if value in {None, ""}:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _fmt(self, value) -> str:
        try:
            if value in {None, ""}:
                return "-"
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    def _stringify(self, value) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
