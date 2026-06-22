from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QProgressBar, QVBoxLayout

from gui.widgets.experiment_stepper import ExperimentStepper
from gui.widgets.log_viewer import LogViewer
from gui.widgets.section_card import SectionCard
from gui.widgets.trial_record_table import TrialRecordTable


class ExperimentStatusCard(QFrame):
    def __init__(self, title: str, value: str, tone: str = "neutral") -> None:
        super().__init__()
        self.setObjectName("experimentStatusCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setObjectName("experimentStatusTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("experimentStatusValue")
        self.value_label.setProperty("tone", tone)
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str, tone: str = "neutral") -> None:
        self.value_label.setText(value)
        self.value_label.setProperty("tone", tone)
        self.value_label.style().unpolish(self.value_label)
        self.value_label.style().polish(self.value_label)


class ExperimentStatusPanel(SectionCard):
    def __init__(self) -> None:
        super().__init__("实验运行状态")
        self._build_summary()
        self._build_progress()
        self._build_stepper()
        self._build_records()
        self._build_logs()

    def append_log(self, line: str) -> None:
        self.log_viewer.append(line)

    def clear_log(self) -> None:
        self.log_viewer.clear()

    def set_waiting(self) -> None:
        self.state_card.set_value("等待开始", "neutral")
        self.task_card.set_value("-")
        self.stage_card.set_value("-")
        self.set_progress(0)
        self.stepper.set_stages(self._stage_rows("等待中", 0))

    def set_stopped(self) -> None:
        self.state_card.set_value("已停止", "warning")
        self.stage_card.set_value("用户停止")

    def set_process_running(self, total_tasks: int = 1) -> None:
        self.state_card.set_value("运行中", "success")
        self.task_card.set_value(f"0 / {max(1, total_tasks)}")
        self.stage_card.set_value("启动后端训练进程")
        self.set_progress(1)
        self.stepper.set_stages(
            [
                {"name": "准备", "en": "Prepare", "status": "进行中", "progress": 10},
                {"name": "训练", "en": "Train", "status": "等待中", "progress": 0},
                {"name": "验证", "en": "Val", "status": "等待中", "progress": 0},
                {"name": "反馈", "en": "Feedback", "status": "等待中", "progress": 0},
                {"name": "完成", "en": "Done", "status": "等待中", "progress": 0},
            ]
        )

    def set_progress(self, value: int) -> None:
        value = max(0, min(100, int(value)))
        self.progress.setValue(value)
        self.progress_label.setText(f"{value}%")
        self.progress_card.set_value(f"{value}%")

    def set_finished(self, success: bool) -> None:
        if success:
            self.state_card.set_value("已完成", "success")
            self.task_card.set_value("1 / 1")
            self.stage_card.set_value("训练任务完成")
            self.set_progress(100)
            self.stepper.set_stages(self._stage_rows("已完成", 100))
        else:
            self.state_card.set_value("失败", "error")
            self.stage_card.set_value("后端返回错误")

    def set_trial_status(self, current: int, total: int, stage: str = "") -> None:
        self.task_card.set_value(f"{current} / {max(1, total)}")
        if stage:
            self.stage_card.set_value(stage)
        if total > 0:
            self.set_progress(int(current * 100 / total))

    def set_trial_records(self, records: list[dict]) -> None:
        self.record_table.set_records(records)

    def _build_summary(self) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        self.state_card = ExperimentStatusCard("当前状态", "等待开始", "neutral")
        self.task_card = ExperimentStatusCard("训练任务", "-")
        self.stage_card = ExperimentStatusCard("当前阶段", "-")
        self.progress_card = ExperimentStatusCard("总进度", "0%")
        for index, card in enumerate([self.state_card, self.task_card, self.stage_card, self.progress_card]):
            grid.addWidget(card, 0, index)
            grid.setColumnStretch(index, 1)
        self.body.addLayout(grid)

    def _build_progress(self) -> None:
        row = QGridLayout()
        row.setContentsMargins(0, 6, 0, 4)
        self.progress = QProgressBar()
        self.progress.setObjectName("experimentProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress_label = QLabel("0%")
        self.progress_label.setObjectName("experimentProgressLabel")
        row.addWidget(self.progress, 0, 0)
        row.addWidget(self.progress_label, 0, 1)
        row.setColumnStretch(0, 1)
        self.body.addLayout(row)

    def _build_stepper(self) -> None:
        card = QFrame()
        card.setObjectName("stageProgressCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)
        title = QLabel("阶段进度")
        title.setObjectName("stageProgressTitle")
        self.stepper = ExperimentStepper()
        self.stepper.set_stages(self._stage_rows("等待中", 0))
        layout.addWidget(title)
        layout.addWidget(self.stepper)
        self.body.addWidget(card)

    def _build_records(self) -> None:
        title = QLabel("训练记录")
        title.setObjectName("experimentBlockTitle")
        self.body.addWidget(title)
        self.record_table = TrialRecordTable()
        self.record_table.setMaximumHeight(152)
        self.record_table.set_records([])
        self.body.addWidget(self.record_table)

    def _build_logs(self) -> None:
        title = QLabel("实时日志（stdout / stderr）")
        title.setObjectName("experimentBlockTitle")
        self.body.addWidget(title)
        self.log_viewer = LogViewer()
        self.log_viewer.setMaximumHeight(166)
        self.log_viewer.clear()
        self.body.addWidget(self.log_viewer, 1)

    def _stage_rows(self, status: str, progress: int) -> list[dict]:
        return [
            {"name": "准备", "en": "Prepare", "status": status, "progress": progress},
            {"name": "训练", "en": "Train", "status": status, "progress": progress},
            {"name": "验证", "en": "Val", "status": status, "progress": progress},
            {"name": "反馈", "en": "Feedback", "status": status, "progress": progress},
            {"name": "完成", "en": "Done", "status": status, "progress": progress},
        ]
