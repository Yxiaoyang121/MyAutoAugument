from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

from gui.models.experiment_config import (
    SINGLE_TRIAL_MODES,
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_FIXED_CATF,
    TRAINING_MODE_INTELLIGENT,
    TRAINING_MODE_LEGACY_SEARCH,
    TRAINING_MODE_NORMAL,
)
from gui.widgets.section_card import SectionCard


MOCK_EXPERIMENT_CONFIG = {
    "dataset_path": r"D:\AutoAugment\dataset",
    "output_dir": r"D:\AutoAugment\outputs",
    "trials": 24,
    "samples_per_trial": 500,
    "epochs": 50,
    "imgsz": 640,
    "workers": 4,
    "batch": 16,
    "seed": 42,
    "model": "yolov8n.pt",
    "run_mode": TRAINING_MODE_INTELLIGENT,
    "metric": "mAP50",
}

MAIN_TRAINING_MODES = [
    ("普通训练", TRAINING_MODE_NORMAL),
    ("自定义增强训练", TRAINING_MODE_CUSTOM),
    ("智能自动增强训练", TRAINING_MODE_INTELLIGENT),
]

ADVANCED_TRAINING_MODES = [
    ("fixed CATF-v2 对照实验", TRAINING_MODE_FIXED_CATF),
    ("Legacy AutoAugment Search", TRAINING_MODE_LEGACY_SEARCH),
]

START_BUTTON_TEXT = {
    TRAINING_MODE_NORMAL: "开始普通训练",
    TRAINING_MODE_CUSTOM: "开始自定义增强训练",
    TRAINING_MODE_INTELLIGENT: "开始智能自动增强",
    TRAINING_MODE_FIXED_CATF: "运行 fixed CATF-v2 对照",
    TRAINING_MODE_LEGACY_SEARCH: "运行 Legacy 搜索",
}


class ExperimentConfigPanel(SectionCard):
    start_requested = Signal()
    stop_requested = Signal()
    open_output_requested = Signal()
    output_dir_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__("实验配置")
        self._build_form()
        self._build_advanced()
        self._build_buttons()

    def values(self) -> dict:
        return {
            "dataset_path": self.dataset_path.text().strip(),
            "output_dir": self.output_dir.text().strip(),
            "trials": self.trials.value(),
            "samples_per_trial": self.samples.value(),
            "epochs": self.epochs.value(),
            "imgsz": self.imgsz.value(),
            "workers": self.workers.value(),
            "batch": self.batch.value(),
            "seed": self.seed.value(),
            "model": self.model.currentText(),
            "run_mode": self.run_mode.currentData() or TRAINING_MODE_INTELLIGENT,
            "metric": self.metric_combo.currentText(),
            "proxy_filter": self.proxy_filter.isChecked(),
            "real_yolo_train": self.real_yolo_train.isChecked(),
            "adaptive_policy_update": self.adaptive_policy_update.isChecked(),
            "per_trial_diagnosis": self.per_trial_diagnosis.isChecked(),
        }

    def set_dataset_path(self, path: str) -> None:
        if path:
            self.dataset_path.setText(path)

    def set_output_dir(self, path: str) -> None:
        if path:
            self.output_dir.setText(path)

    def set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _build_form(self) -> None:
        form = QGridLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)

        self.dataset_path = QLineEdit(MOCK_EXPERIMENT_CONFIG["dataset_path"])
        self.dataset_path.setObjectName("experimentInput")
        self.dataset_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.output_dir = QLineEdit(MOCK_EXPERIMENT_CONFIG["output_dir"])
        self.output_dir.setObjectName("experimentInput")
        self.output_dir.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.output_dir.textChanged.connect(self.output_dir_changed.emit)

        self.trials = self._spin(1, 1000, MOCK_EXPERIMENT_CONFIG["trials"])
        self.samples = self._spin(1, 100000, MOCK_EXPERIMENT_CONFIG["samples_per_trial"])
        self.epochs = self._spin(1, 1000, MOCK_EXPERIMENT_CONFIG["epochs"])
        self.imgsz = self._spin(64, 4096, MOCK_EXPERIMENT_CONFIG["imgsz"], 32)
        self.workers = self._spin(0, 64, MOCK_EXPERIMENT_CONFIG["workers"])
        self.batch = self._spin(1, 1024, MOCK_EXPERIMENT_CONFIG["batch"])
        self.seed = self._spin(0, 2_147_483_647, MOCK_EXPERIMENT_CONFIG["seed"])

        self.model = QComboBox()
        self.model.setObjectName("experimentCombo")
        self.model.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])

        self.run_mode = QComboBox()
        self.run_mode.setObjectName("experimentCombo")
        self.run_mode.currentIndexChanged.connect(self._sync_mode_ui)
        self._populate_run_modes(TRAINING_MODE_INTELLIGENT)

        self.metric_combo = QComboBox()
        self.metric_combo.setObjectName("experimentCombo")
        self.metric_combo.addItems(["mAP50", "mAP50-95", "Score"])

        full_rows = [
            ("训练模式", self.run_mode),
            ("数据集路径", self._path_row(self.dataset_path, self._choose_dataset)),
            ("输出目录", self._path_row(self.output_dir, self._choose_output)),
        ]
        for row, (label, widget) in enumerate(full_rows):
            label_widget = QLabel(label)
            label_widget.setObjectName("experimentFormLabel")
            form.addWidget(label_widget, row, 0)
            form.addWidget(widget, row, 1, 1, 3)

        compact_rows = [
            (("试验次数", self.trials), ("样本数", self.samples)),
            (("epochs", self.epochs), ("imgsz", self.imgsz)),
            (("workers", self.workers), ("batch", self.batch)),
            (("seed", self.seed), ("YOLO", self.model)),
            (("评价指标", self.metric_combo), (None, None)),
        ]
        for offset, pair in enumerate(compact_rows, start=len(full_rows)):
            for group, (label, widget) in enumerate(pair):
                if label is None or widget is None:
                    continue
                label_widget = QLabel(label)
                label_widget.setObjectName("experimentFormLabel")
                form.addWidget(label_widget, offset, group * 2)
                form.addWidget(widget, offset, group * 2 + 1)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        self.body.addLayout(form)

    def _build_advanced(self) -> None:
        box = QFrame()
        box.setObjectName("advancedOptionsBox")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)

        title = QLabel("高级选项")
        title.setObjectName("advancedOptionsTitle")
        layout.addWidget(title)

        self.show_ablation_modes = QCheckBox("显示实验对照模式")
        self.show_ablation_modes.setObjectName("experimentCheck")
        self.show_ablation_modes.setChecked(False)
        self.show_ablation_modes.toggled.connect(self._on_ablation_modes_toggled)
        layout.addWidget(self.show_ablation_modes)

        self.proxy_filter = QCheckBox("启用代理筛选")
        self.real_yolo_train = QCheckBox("真实 YOLO 训练/验证")
        self.adaptive_policy_update = QCheckBox("自适应策略更新")
        self.per_trial_diagnosis = QCheckBox("逐轮诊断")
        self.real_yolo_train.setChecked(True)
        self.adaptive_policy_update.setChecked(True)
        self.per_trial_diagnosis.setChecked(True)

        checks = [
            self.proxy_filter,
            self.real_yolo_train,
            self.adaptive_policy_update,
            self.per_trial_diagnosis,
        ]
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        for index, check in enumerate(checks):
            check.setObjectName("experimentCheck")
            grid.addWidget(check, index // 2, index % 2)
        layout.addLayout(grid)
        self.body.addWidget(box)
        self._sync_mode_ui()

    def _build_buttons(self) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(8)
        self.start_button = QPushButton()
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QPushButton("停止实验")
        self.stop_button.setObjectName("dangerButton")
        self.open_output_button = QPushButton("打开输出目录")
        self.open_output_button.setObjectName("secondaryButton")
        self.start_button.clicked.connect(self.start_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.open_output_button.clicked.connect(self.open_output_requested.emit)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        row.addWidget(self.open_output_button)
        self.body.addLayout(row)
        self._sync_mode_ui()

    def _populate_run_modes(self, selected_mode: str | None = None) -> None:
        selected_mode = selected_mode or (self.run_mode.currentData() if hasattr(self, "run_mode") else TRAINING_MODE_INTELLIGENT)
        show_advanced = bool(getattr(self, "show_ablation_modes", None) and self.show_ablation_modes.isChecked())
        modes = list(MAIN_TRAINING_MODES)
        if show_advanced:
            modes.extend(ADVANCED_TRAINING_MODES)

        self.run_mode.blockSignals(True)
        self.run_mode.clear()
        for label, mode in modes:
            self.run_mode.addItem(label, mode)
        index = self.run_mode.findData(selected_mode)
        if index < 0:
            index = self.run_mode.findData(TRAINING_MODE_INTELLIGENT)
        self.run_mode.setCurrentIndex(max(index, 0))
        self.run_mode.blockSignals(False)
        self._sync_mode_ui()

    def _on_ablation_modes_toggled(self, enabled: bool) -> None:
        current = self.run_mode.currentData()
        if not enabled and current in {TRAINING_MODE_FIXED_CATF, TRAINING_MODE_LEGACY_SEARCH}:
            current = TRAINING_MODE_INTELLIGENT
        self._populate_run_modes(str(current or TRAINING_MODE_INTELLIGENT))

    def _path_row(self, line: QLineEdit, callback) -> QFrame:
        row = QFrame()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse = QPushButton("浏览")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(callback)
        layout.addWidget(line, 1)
        layout.addWidget(browse)
        return row

    def _choose_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集路径", self.dataset_path.text())
        if path:
            self.dataset_path.setText(path)

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir.text())
        if path:
            self.output_dir.setText(path)
            self.output_dir_changed.emit(path)

    def _spin(self, low: int, high: int, value: int, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setObjectName("experimentSpin")
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _sync_mode_ui(self) -> None:
        if not hasattr(self, "run_mode") or not hasattr(self, "trials"):
            return
        mode = self.run_mode.currentData() or TRAINING_MODE_INTELLIGENT
        single_trial = mode in SINGLE_TRIAL_MODES
        self.trials.setEnabled(not single_trial)
        if single_trial:
            self.trials.setValue(1)
        elif self.trials.value() <= 1:
            self.trials.setValue(MOCK_EXPERIMENT_CONFIG["trials"])

        catf_visible = mode in {TRAINING_MODE_INTELLIGENT, TRAINING_MODE_FIXED_CATF}
        if hasattr(self, "proxy_filter"):
            self.proxy_filter.setEnabled(False)
            self.proxy_filter.setChecked(mode == TRAINING_MODE_INTELLIGENT)
        if hasattr(self, "adaptive_policy_update"):
            self.adaptive_policy_update.setEnabled(mode in {TRAINING_MODE_INTELLIGENT, TRAINING_MODE_LEGACY_SEARCH})
            self.adaptive_policy_update.setChecked(mode == TRAINING_MODE_INTELLIGENT)
        if hasattr(self, "per_trial_diagnosis"):
            self.per_trial_diagnosis.setEnabled(mode != TRAINING_MODE_NORMAL)
        if hasattr(self, "start_button"):
            self.start_button.setText(START_BUTTON_TEXT.get(mode, "开始实验"))

        self.run_mode.setToolTip(
            "当前模式使用 Preserve-Weak Image-only CATF；sampler_only=false，weighted_index_list=false，sampled_distribution_changed=false。"
            if catf_visible and mode == TRAINING_MODE_INTELLIGENT
            else ""
        )
