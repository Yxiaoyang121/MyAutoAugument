from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
    PRESERVE_WEAK_ALGORITHM_LOCK,
    PRESERVE_WEAK_DEFAULTS,
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_PRESERVE_WEAK,
    TRAINING_MODE_YOLO_DEFAULT,
    YOLO_AUG_DEFAULTS,
)
from gui.widgets.section_card import SectionCard


TRAINING_MODE_ITEMS = [
    ("自定义增强策略训练", TRAINING_MODE_CUSTOM),
    ("YOLO 默认训练", TRAINING_MODE_YOLO_DEFAULT),
    ("Preserve-Weak Image-only CATF", TRAINING_MODE_PRESERVE_WEAK),
]

MODE_DETAILS = {
    TRAINING_MODE_CUSTOM: (
        "当前模式：自定义增强策略训练",
        "用户手动配置 YOLO 增强参数后训练；不启用 CATF，不调用智能策略选择，采样分布不变。",
        ["YOLO 增强参数", "用户配置", "单次训练", "采样分布不变"],
    ),
    TRAINING_MODE_YOLO_DEFAULT: (
        "当前模式：YOLO 默认训练",
        "普通 YOLO baseline 训练；不启用 CATF、不启用自动增强、不显示增强参数。",
        ["YOLO baseline", "默认训练", "单次训练", "采样分布不变"],
    ),
    TRAINING_MODE_PRESERVE_WEAK: (
        "当前模式：Preserve-Weak Image-only CATF",
        "低风险保留原增强，中风险弱化增强，高风险跳过增强；全程不改变训练采样分布。",
        ["CATF-v2", "Image-only", "Preserve-Weak", "采样分布不变"],
    ),
}


class ExperimentConfigPanel(SectionCard):
    start_requested = Signal()
    stop_requested = Signal()
    open_output_requested = Signal()
    output_dir_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__("实验配置")
        self._build_form()
        self._build_mode_summary()
        self._build_custom_aug_box()
        self._build_algorithm_lock_box()
        self._build_buttons()
        self._sync_mode_ui()

    def values(self) -> dict:
        return {
            "dataset_path": self.dataset_path.text().strip(),
            "output_dir": self.output_dir.text().strip(),
            "epochs": self.epochs.value(),
            "imgsz": self.imgsz.value(),
            "workers": self.workers.value(),
            "batch": self.batch.value(),
            "seed": self.seed.value(),
            "device": self.device.text().strip() or "0",
            "model": self.model.currentText().strip(),
            "run_mode": self.run_mode.currentData() or TRAINING_MODE_PRESERVE_WEAK,
            "metric": "map50",
            "yolo_aug_params": self._custom_aug_values(),
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
        form.setVerticalSpacing(8)

        self.run_mode = QComboBox()
        self.run_mode.setObjectName("experimentCombo")
        for label, mode in TRAINING_MODE_ITEMS:
            self.run_mode.addItem(label, mode)
        self.run_mode.setCurrentIndex(self.run_mode.findData(TRAINING_MODE_PRESERVE_WEAK))
        self.run_mode.currentIndexChanged.connect(self._sync_mode_ui)

        self.dataset_path = QLineEdit(PRESERVE_WEAK_DEFAULTS["dataset_path"])
        self.dataset_path.setObjectName("experimentInput")
        self.dataset_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        self.output_dir = QLineEdit(r"D:\AutoAugment\outputs")
        self.output_dir.setObjectName("experimentInput")
        self.output_dir.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.output_dir.textChanged.connect(self.output_dir_changed.emit)

        self.model = QComboBox()
        self.model.setObjectName("experimentCombo")
        self.model.addItems(["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt"])

        self.epochs = self._spin(1, 1000, int(PRESERVE_WEAK_DEFAULTS["epochs"]))
        self.imgsz = self._spin(64, 4096, int(PRESERVE_WEAK_DEFAULTS["imgsz"]), 32)
        self.batch = self._spin(1, 1024, int(PRESERVE_WEAK_DEFAULTS["batch"]))
        self.workers = self._spin(0, 64, int(PRESERVE_WEAK_DEFAULTS["workers"]))
        self.device = QLineEdit(str(PRESERVE_WEAK_DEFAULTS["device"]))
        self.device.setObjectName("experimentInput")
        self.seed = self._spin(0, 2, int(PRESERVE_WEAK_DEFAULTS["seed"]))

        rows = [
            ("训练模式", self.run_mode),
            ("data.yaml 路径", self._path_row(self.dataset_path, self._choose_data_yaml)),
            ("输出目录", self._path_row(self.output_dir, self._choose_output)),
            ("model", self.model),
        ]
        for row, (label, widget) in enumerate(rows):
            form.addWidget(self._label(label), row, 0)
            form.addWidget(widget, row, 1, 1, 3)

        compact_rows = [
            (("epochs", self.epochs), ("imgsz", self.imgsz)),
            (("batch", self.batch), ("workers", self.workers)),
            (("device", self.device), ("seed", self.seed)),
        ]
        for offset, pair in enumerate(compact_rows, start=len(rows)):
            for group, (label, widget) in enumerate(pair):
                form.addWidget(self._label(label), offset, group * 2)
                form.addWidget(widget, offset, group * 2 + 1)

        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        self.body.addLayout(form)

    def _build_mode_summary(self) -> None:
        self.mode_summary = QFrame()
        self.mode_summary.setObjectName("modeSummaryCard")
        layout = QVBoxLayout(self.mode_summary)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self.mode_title = QLabel()
        self.mode_title.setObjectName("modeSummaryTitle")
        self.mode_description = QLabel()
        self.mode_description.setWordWrap(True)
        self.mode_description.setObjectName("modeSummaryDescription")
        self.mode_badges = QHBoxLayout()
        self.mode_badges.setContentsMargins(0, 0, 0, 0)
        self.mode_badges.setSpacing(6)
        self.mode_badge_labels: list[QLabel] = []
        for text in ["", "", "", ""]:
            badge = QLabel(text)
            badge.setObjectName("modeSummaryBadge")
            self.mode_badge_labels.append(badge)
            self.mode_badges.addWidget(badge)
        self.mode_badges.addStretch(1)

        layout.addWidget(self.mode_title)
        layout.addWidget(self.mode_description)
        layout.addLayout(self.mode_badges)
        self.body.addWidget(self.mode_summary)

    def _build_custom_aug_box(self) -> None:
        self.custom_aug_box = QFrame()
        self.custom_aug_box.setObjectName("advancedOptionsBox")
        layout = QVBoxLayout(self.custom_aug_box)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        title = QLabel("用户自定义增强参数")
        title.setObjectName("advancedOptionsTitle")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.aug_fields: dict[str, QDoubleSpinBox] = {}
        for index, (name, value) in enumerate(YOLO_AUG_DEFAULTS.items()):
            spin = QDoubleSpinBox()
            spin.setObjectName("experimentSpin")
            spin.setDecimals(3 if name != "close_mosaic" else 0)
            spin.setRange(-180.0 if name in {"degrees", "shear"} else 0.0, 180.0 if name in {"degrees", "shear"} else 100.0)
            if name == "perspective":
                spin.setRange(0.0, 0.01)
                spin.setSingleStep(0.0005)
                spin.setDecimals(4)
            elif name == "close_mosaic":
                spin.setRange(0.0, 1000.0)
                spin.setSingleStep(1.0)
            else:
                spin.setSingleStep(0.05)
            spin.setValue(float(value))
            self.aug_fields[name] = spin
            row = index // 2
            col = (index % 2) * 2
            grid.addWidget(self._label(name), row, col)
            grid.addWidget(spin, row, col + 1)
        layout.addLayout(grid)
        self.body.addWidget(self.custom_aug_box)

    def _build_algorithm_lock_box(self) -> None:
        self.algorithm_lock_box = QFrame()
        self.algorithm_lock_box.setObjectName("advancedOptionsBox")
        layout = QVBoxLayout(self.algorithm_lock_box)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        title = QLabel("算法锁状态（只读）")
        title.setObjectName("advancedOptionsTitle")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        for index, (name, value) in enumerate(PRESERVE_WEAK_ALGORITHM_LOCK.items()):
            row = index // 2
            col = (index % 2) * 2
            key = QLabel(name)
            key.setObjectName("lockKey")
            val = QLabel(str(value).lower())
            val.setObjectName("lockValue")
            grid.addWidget(key, row, col)
            grid.addWidget(val, row, col + 1)
        layout.addLayout(grid)
        self.body.addWidget(self.algorithm_lock_box)

    def _build_buttons(self) -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.setSpacing(8)
        self.start_button = QPushButton("启动训练任务")
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QPushButton("停止训练任务")
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

    def _choose_data_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 data.yaml", self.dataset_path.text(), "YAML (*.yaml *.yml)")
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

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("experimentFormLabel")
        return label

    def _custom_aug_values(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {}
        for name, spin in self.aug_fields.items():
            values[name] = int(spin.value()) if name == "close_mosaic" else float(spin.value())
        return values

    def _sync_mode_ui(self) -> None:
        mode = self.run_mode.currentData() or TRAINING_MODE_PRESERVE_WEAK
        if mode == TRAINING_MODE_PRESERVE_WEAK:
            self.dataset_path.setText(PRESERVE_WEAK_DEFAULTS["dataset_path"])
            self.model.setCurrentText(str(PRESERVE_WEAK_DEFAULTS["model"]))
            self.epochs.setValue(int(PRESERVE_WEAK_DEFAULTS["epochs"]))
            self.imgsz.setValue(int(PRESERVE_WEAK_DEFAULTS["imgsz"]))
            self.batch.setValue(int(PRESERVE_WEAK_DEFAULTS["batch"]))
            self.workers.setValue(int(PRESERVE_WEAK_DEFAULTS["workers"]))
            self.device.setText(str(PRESERVE_WEAK_DEFAULTS["device"]))
            self.seed.setRange(0, 2)
            self.seed.setValue(int(PRESERVE_WEAK_DEFAULTS["seed"]))
        else:
            self.seed.setRange(0, 2_147_483_647)
            if self.model.currentText() == "":
                self.model.setCurrentText("yolo11n.pt")

        self.custom_aug_box.setVisible(mode == TRAINING_MODE_CUSTOM)
        self.algorithm_lock_box.setVisible(mode == TRAINING_MODE_PRESERVE_WEAK)
        title, description, badges = MODE_DETAILS[mode]
        self.mode_title.setText(title)
        self.mode_description.setText(description)
        for badge, text in zip(self.mode_badge_labels, badges):
            badge.setText(text)
