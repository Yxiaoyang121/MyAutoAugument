from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from gui.controllers import DatasetController
from gui.widgets.dataset_health_card import DatasetHealthCard
from gui.widgets.dataset_hint_card import DatasetHintCard
from gui.widgets.dataset_stats_grid import DatasetStatsGrid
from gui.widgets.mini_charts import MiniBarChart, MiniDonutChart
from gui.widgets.sample_preview_grid import SampleGalleryDialog, SamplePreviewGrid
from gui.widgets.section_card import SectionCard
from gui.widgets.structure_check_table import StructureCheckTable


MOCK_DATASET_PATH = r"D:\AutoAugment\dataset"
MOCK_LAST_CHECKED = "2026-06-16 14:28:10"
MOCK_STATS = {
    "图像总数": 12581,
    "标签总数": 45832,
    "类别数": 11,
    "空标签数": 23,
    "缺失标签数": 0,
    "异常图像数": 2,
}
MOCK_STRUCTURE = [
    ("images/train", "正常"),
    ("images/val", "正常"),
    ("labels/train", "正常"),
    ("labels/val", "正常"),
    ("data.yaml", "正常"),
]
MOCK_CLASS_COUNTS = [9000, 5600, 5500, 6000, 5200, 3900, 5000, 4300, 3200, 1200, 800]
MOCK_SPLIT = {"Train": 10057, "Val": 2524}


class DatasetInspectWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, project_root: Path, dataset_path: str) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.dataset_path = dataset_path

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(5, "准备检查数据集目录...")
            info = DatasetController(self.project_root).inspect(self.dataset_path, progress_callback=self.progress.emit)
            self.progress.emit(72, "正在统计类别分布...")
            class_count = int(info.get("class_count", 0) or 0)
            info["_class_counts"] = self._class_distribution(info, class_count)

            self.progress.emit(84, "正在生成样本预览索引...")
            root = Path(info.get("root", self.dataset_path))
            train = int(info.get("train_image_count", 0) or 0)
            val = int(info.get("val_image_count", 0) or 0)
            samples = self._sample_previews(root, limit=48) if info.get("exists") and (train or val) else []
            info["_sample_previews"] = samples[:4]
            info["_gallery_samples"] = samples

            self.progress.emit(96, "正在更新界面数据...")
            self.finished.emit(info)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _class_distribution(self, info: dict, class_count: int) -> list[tuple[int, int]]:
        counts = {index: 0 for index in range(class_count)}
        root = Path(info.get("root", ""))
        for label_root in [root / "labels" / "train", root / "labels" / "val"]:
            if not label_root.exists():
                continue
            for label_path in label_root.rglob("*.txt"):
                for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    counts[class_id] = counts.get(class_id, 0) + 1
        return sorted(counts.items())

    def _sample_previews(self, root: Path, limit: int = 4) -> list[tuple[str, str | None]]:
        samples: list[tuple[int, Path, Path | None]] = []
        image_roots = [
            root / "images" / "train",
            root / "images" / "val",
            root / "images",
            root,
        ]
        seen: set[Path] = set()
        scan_limit = max(limit * 8, limit)
        for image_root in image_roots:
            if not image_root.exists():
                continue
            for image_path in sorted(
                path
                for path in image_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ):
                if image_path in seen:
                    continue
                seen.add(image_path)
                label_path = self._label_for_image(root, image_root, image_path)
                resolved_label = label_path if label_path.exists() else None
                samples.append((self._sample_preview_score(resolved_label), image_path, resolved_label))
                if len(samples) >= scan_limit:
                    break
            if len(samples) >= scan_limit:
                break
        samples.sort(key=lambda item: (-item[0], item[1].name))
        return [(str(image_path), str(label_path) if label_path else None) for _, image_path, label_path in samples[:limit]]

    def _sample_preview_score(self, label_path: Path | None) -> int:
        if not label_path or not label_path.exists():
            return 0
        small_boxes = 0
        medium_boxes = 0
        large_boxes = 0
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                area = max(0.0, float(parts[3])) * max(0.0, float(parts[4]))
            except ValueError:
                continue
            if area <= 0.03:
                small_boxes += 1
            elif area <= 0.18:
                medium_boxes += 1
            else:
                large_boxes += 1
        return small_boxes * 8 + medium_boxes * 4 - large_boxes

    def _label_for_image(self, dataset_root: Path, image_root: Path, image_path: Path) -> Path:
        try:
            relative = image_path.relative_to(dataset_root / "images")
            return dataset_root / "labels" / relative.with_suffix(".txt")
        except ValueError:
            pass
        try:
            relative = image_path.relative_to(image_root)
            if image_root.name in {"train", "val"} and image_root.parent.name == "images":
                return dataset_root / "labels" / image_root.name / relative.with_suffix(".txt")
            return dataset_root / "labels" / relative.with_suffix(".txt")
        except ValueError:
            return image_path.with_suffix(".txt")


class DatasetPage(QWidget):
    dataset_changed = Signal(str, dict)

    def __init__(self, project_root: Path, *, auto_inspect: bool = True) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.auto_inspect = auto_inspect
        self.state_path = self.project_root / "outputs" / "gui_state.json"
        self.controller = DatasetController(self.project_root)
        self.inspection: dict = self._mock_inspection()
        self.current_samples: list[tuple[Path, Path | None]] = []
        self.current_gallery_samples: list[tuple[Path, Path | None]] = []
        self._inspect_thread: QThread | None = None
        self._inspect_worker: DatasetInspectWorker | None = None
        self._build_ui()
        default_path = self._default_dataset_path()
        self.dataset_root.setText(default_path)
        if Path(default_path).exists() and self.auto_inspect:
            self._populate_loading(default_path)
            QTimer.singleShot(120, self.inspect)
        elif Path(default_path).exists():
            self._populate_loading(default_path)
        else:
            self._populate_mock()

    def _build_ui(self) -> None:
        self.setObjectName("datasetPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 12, 18, 12)
        layout.setSpacing(10)

        self._build_path_card(layout)
        self._build_health_row(layout)
        self._build_visual_row(layout)
        self.hint_card = DatasetHintCard()
        layout.addWidget(self.hint_card)
        layout.addStretch(1)

    def _build_path_card(self, parent: QVBoxLayout) -> None:
        card = SectionCard("")
        card.setObjectName("datasetPathCard")
        card.setMinimumHeight(72)
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(10)

        label = QLabel("数据集路径")
        label.setObjectName("pathCardLabel")
        self.dataset_root = QLineEdit()
        self.dataset_root.setObjectName("datasetPathInput")
        self.dataset_root.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.dataset_root.setText(MOCK_DATASET_PATH)

        self.browse_button = QPushButton("浏览")
        self.browse_button.setObjectName("secondaryButton")
        self.check_button = QPushButton("检查数据集")
        self.check_button.setObjectName("primaryButton")
        self.last_checked_label = QLabel(f"最后检查：{MOCK_LAST_CHECKED}")
        self.last_checked_label.setObjectName("lastChecked")

        self.browse_button.clicked.connect(self._choose_dataset_root)
        self.check_button.clicked.connect(self.inspect)

        row.addWidget(label)
        row.addWidget(self.dataset_root, 1)
        row.addWidget(self.browse_button)
        row.addWidget(self.check_button)
        row.addStretch(1)
        row.addWidget(self.last_checked_label)
        card.body.addLayout(row)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(12)
        spacer = QLabel("")
        spacer.setObjectName("pathCardLabel")
        self.dataset_progress_label = QLabel("就绪")
        self.dataset_progress_label.setObjectName("datasetProgressLabel")
        self.dataset_progress = QProgressBar()
        self.dataset_progress.setObjectName("datasetProgress")
        self.dataset_progress.setRange(0, 100)
        self.dataset_progress.setValue(0)
        self.dataset_progress.setTextVisible(False)
        self.dataset_progress.setVisible(False)
        self.dataset_progress_label.setVisible(False)
        progress_row.addWidget(spacer)
        progress_row.addWidget(self.dataset_progress, 1)
        progress_row.addWidget(self.dataset_progress_label)
        card.body.addLayout(progress_row)
        parent.addWidget(card)

    def _build_health_row(self, parent: QVBoxLayout) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.health_card = DatasetHealthCard()

        stats_card = SectionCard("数据集统计")
        stats_card.setMinimumHeight(188)
        self.stats_grid = DatasetStatsGrid()
        stats_card.body.addWidget(self.stats_grid, 1)

        structure_card = SectionCard("结构检查结果")
        structure_card.setMinimumHeight(188)
        self.structure_table = StructureCheckTable()
        structure_card.body.addWidget(self.structure_table, 1)

        grid.addWidget(self.health_card, 0, 0)
        grid.addWidget(stats_card, 0, 1)
        grid.addWidget(structure_card, 0, 2)
        grid.setColumnStretch(0, 12)
        grid.setColumnStretch(1, 14)
        grid.setColumnStretch(2, 11)
        parent.addLayout(grid)

    def _build_visual_row(self, parent: QVBoxLayout) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        class_card = SectionCard("类别分布（按标签数量）")
        class_card.setMinimumHeight(205)
        self.class_chart = MiniBarChart("类别名称")
        class_card.body.addWidget(self.class_chart, 1)

        split_card = SectionCard("数据集划分")
        split_card.setMinimumHeight(205)
        self.split_chart = MiniDonutChart()
        split_card.body.addWidget(self.split_chart, 1)

        sample_card = SectionCard("")
        sample_card.setMinimumHeight(205)
        sample_header = QHBoxLayout()
        sample_header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("样本预览")
        title.setObjectName("sectionCardTitle")
        self.more_samples_button = QPushButton("更多 >")
        self.more_samples_button.setObjectName("secondaryButton")
        self.more_samples_button.clicked.connect(self._show_more_samples)
        sample_header.addWidget(title)
        sample_header.addStretch(1)
        sample_header.addWidget(self.more_samples_button)
        self.sample_grid = SamplePreviewGrid()
        sample_card.body.addLayout(sample_header)
        sample_card.body.addWidget(self.sample_grid, 1)

        grid.addWidget(class_card, 0, 0)
        grid.addWidget(split_card, 0, 1)
        grid.addWidget(sample_card, 0, 2)
        grid.setColumnStretch(0, 13)
        grid.setColumnStretch(1, 9)
        grid.setColumnStretch(2, 17)
        parent.addLayout(grid, 1)

    def current_dataset_path(self) -> str:
        return self.dataset_root.text().strip()

    def inspect(self) -> None:
        if self._inspect_thread and self._inspect_thread.isRunning():
            return
        dataset_path = self.dataset_root.text().strip()
        self._set_inspection_busy(True)
        self._on_inspection_progress(3, "正在启动后台检查...")

        self._inspect_thread = QThread(self)
        self._inspect_worker = DatasetInspectWorker(self.project_root, dataset_path)
        self._inspect_worker.moveToThread(self._inspect_thread)
        self._inspect_thread.started.connect(self._inspect_worker.run)
        self._inspect_worker.progress.connect(self._on_inspection_progress)
        self._inspect_worker.finished.connect(self._on_inspection_finished)
        self._inspect_worker.failed.connect(self._on_inspection_failed)
        self._inspect_worker.finished.connect(self._inspect_thread.quit)
        self._inspect_worker.failed.connect(self._inspect_thread.quit)
        self._inspect_thread.finished.connect(self._inspect_worker.deleteLater)
        self._inspect_thread.finished.connect(self._inspect_thread.deleteLater)
        self._inspect_thread.finished.connect(self._on_inspection_thread_finished)
        self._inspect_thread.start()

    def _choose_dataset_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 YOLO 数据集根目录", self.dataset_root.text())
        if path:
            self.dataset_root.setText(path)
            self._save_recent_dataset(path)
            self.inspect()

    def _set_inspection_busy(self, busy: bool) -> None:
        self.browse_button.setEnabled(not busy)
        self.check_button.setEnabled(not busy)
        self.dataset_root.setEnabled(not busy)
        self.dataset_progress.setVisible(True)
        self.dataset_progress_label.setVisible(True)
        if busy:
            self.check_button.setText("检查中...")
        else:
            self.check_button.setText("检查数据集")

    @Slot(int, str)
    def _on_inspection_progress(self, value: int, text: str) -> None:
        self.dataset_progress.setRange(0, 100)
        self.dataset_progress.setValue(max(0, min(100, value)))
        self.dataset_progress_label.setText(text)

    @Slot(dict)
    def _on_inspection_finished(self, info: dict) -> None:
        self.inspection = info
        self._populate_from_inspection(info)
        checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_checked_label.setText(f"最后检查：{checked_at}")
        self._on_inspection_progress(100, "检查完成")
        self._set_inspection_busy(False)
        self._save_recent_dataset(str(info.get("root", self.dataset_root.text().strip())))
        self.dataset_changed.emit(info["root"], info)

    @Slot(str)
    def _on_inspection_failed(self, message: str) -> None:
        self._on_inspection_progress(0, f"检查失败：{message}")
        self._set_inspection_busy(False)

    @Slot()
    def _on_inspection_thread_finished(self) -> None:
        self._inspect_thread = None
        self._inspect_worker = None

    def shutdown(self) -> None:
        if self._inspect_thread and self._inspect_thread.isRunning():
            self._inspect_thread.quit()
            self._inspect_thread.wait(8000)

    def _populate_mock(self) -> None:
        self.dataset_root.setText(MOCK_DATASET_PATH)
        self.health_card.set_health(92, "良好", "数据集结构完整，标签分布合理，可以正常进行实验。")
        self.stats_grid.set_stats(MOCK_STATS)
        self.structure_table.set_checks(MOCK_STRUCTURE)
        self.class_chart.set_values([(str(index), count) for index, count in enumerate(MOCK_CLASS_COUNTS)])
        self.split_chart.set_segments(
            [
                ("Train", MOCK_SPLIT["Train"], "#3B82F6"),
                ("Val", MOCK_SPLIT["Val"], "#10B981"),
            ]
        )
        self.current_samples = []
        self.current_gallery_samples = []
        self.sample_grid.set_mock_samples()

    def _populate_loading(self, dataset_path: str) -> None:
        self.health_card.set_health(0, "待检查", f"正在后台检查数据集：{dataset_path}")
        self.stats_grid.set_stats({"图像总数": 0, "标签总数": 0, "类别数": 0, "空标签数": 0, "缺失标签数": 0, "异常图像数": 0})
        self.structure_table.set_checks([("images/train", "警告"), ("images/val", "警告"), ("labels/train", "警告"), ("labels/val", "警告"), ("data.yaml", "警告")])
        self.class_chart.set_values([])
        self.split_chart.set_segments([])
        self.current_samples = []
        self.current_gallery_samples = []
        self.sample_grid.set_empty("正在加载样本预览")
        self.hint_card.set_detail("• 正在后台检查数据集，不会阻塞界面。\n• 检查完成后会自动刷新类别分布和样本预览。\n• 若路径不正确，请点击浏览重新选择。")

    def _default_dataset_path(self) -> str:
        recent = self._load_recent_dataset()
        if recent and Path(recent).exists():
            return recent
        candidates = [
            Path(r"E:\TJGY\DataSet2"),
            self.project_root / "dataset",
            self.project_root.parent / "dataset",
            self.project_root.parent.parent / "DataSet2",
            Path(MOCK_DATASET_PATH),
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return MOCK_DATASET_PATH

    def _load_recent_dataset(self) -> str:
        try:
            if not self.state_path.exists():
                return ""
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return str(data.get("dataset_path", "")).strip()
        except (OSError, json.JSONDecodeError, TypeError):
            return ""

    def _save_recent_dataset(self, path: str) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if self.state_path.exists():
                try:
                    data = json.loads(self.state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, TypeError):
                    data = {}
            data["dataset_path"] = path
            self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            return

    def _populate_from_inspection(self, info: dict) -> None:
        exists = bool(info.get("exists"))
        errors = info.get("errors", []) or []
        warnings = info.get("warnings", []) or []
        missing = info.get("missing_labels", []) or []
        empty = info.get("empty_labels", []) or []
        bad = info.get("bad_images", []) or []
        score = self._health_score(info)
        if not exists:
            self.health_card.set_health(0, "路径不存在", "数据集路径不存在，请选择正确的 YOLO 数据集根目录。")
            self.hint_card.set_detail("• 数据集路径不存在。\n• 请先选择包含 images、labels 和 data.yaml 的目录。\n• 修复路径后再运行实验。")
        elif errors:
            self.health_card.set_health(score, "需修复", "关键目录缺失，暂不建议继续实验。")
            self.hint_card.set_detail("• 检测到关键目录缺失。\n• 请补齐 images/train、images/val、labels/train、labels/val。\n• 修复后再启动实验。")
        elif warnings or missing or bad or empty:
            self.health_card.set_health(score, "有警告", "数据集可用于预览，但建议修复缺失标签、空标签或异常图像。")
            self.hint_card.set_detail("• 数据集可以继续检查和预览。\n• 建议清理缺失标签、空标签或异常图像。\n• 修复后训练结果会更稳定。")
        else:
            self.health_card.set_health(score, "良好", "数据集结构完整，标签分布合理，可以正常进行实验。")
            self.hint_card.set_detail("• 数据集结构完整，所有必需文件均已找到。\n• 标签分布较为均衡，可以支持模型训练和验证。\n• 建议继续进行策略设计和预览，优化数据增强效果。")

        stats = {
            "图像总数": int(info.get("train_image_count", 0) or 0) + int(info.get("val_image_count", 0) or 0),
            "标签总数": int(info.get("train_label_count", 0) or 0) + int(info.get("val_label_count", 0) or 0),
            "类别数": int(info.get("class_count", 0) or 0),
            "空标签数": len(empty),
            "缺失标签数": len(missing),
            "异常图像数": len(bad),
        }
        self.stats_grid.set_stats(stats)
        self.structure_table.set_checks(self._structure_checks(info))

        class_counts = self._normalize_class_counts(info.get("_class_counts"))
        if class_counts is None:
            class_counts = self._class_distribution(info, int(info.get("class_count", 0) or 0))
        class_names = [str(name) for name in (info.get("class_names", []) or [])]
        self.class_chart.set_values(
            [
                (class_names[index] if index < len(class_names) and class_names[index] else str(index), count)
                for index, count in class_counts
            ]
            if class_counts
            else []
        )
        train = int(info.get("train_image_count", 0) or 0)
        val = int(info.get("val_image_count", 0) or 0)
        self.split_chart.set_segments([("Train", train, "#3B82F6"), ("Val", val, "#10B981")] if train or val else [])
        samples = self._normalize_samples(info.get("_sample_previews"))
        if not samples:
            samples = self._sample_previews(Path(info.get("root", "")), limit=4) if exists and (train or val) else []
        gallery_samples = self._normalize_samples(info.get("_gallery_samples"))
        self.current_gallery_samples = gallery_samples or samples
        self.current_samples = samples
        self.sample_grid.set_samples(samples) if samples else self.sample_grid.set_empty("暂无样本预览")

    def _normalize_class_counts(self, raw: object) -> list[tuple[int, int]] | None:
        if not isinstance(raw, list):
            return None
        counts: list[tuple[int, int]] = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                counts.append((int(item[0]), int(item[1])))
            except (TypeError, ValueError):
                continue
        return counts

    def _normalize_samples(self, raw: object) -> list[tuple[Path, Path | None]]:
        if not isinstance(raw, list):
            return []
        samples: list[tuple[Path, Path | None]] = []
        for item in raw:
            if not isinstance(item, (list, tuple)) or len(item) < 1:
                continue
            image_path = Path(str(item[0]))
            label_path = Path(str(item[1])) if len(item) > 1 and item[1] else None
            samples.append((image_path, label_path))
        return samples

    def _mock_inspection(self) -> dict:
        return {
            "root": MOCK_DATASET_PATH,
            "exists": True,
            "has_images_train": True,
            "has_images_val": True,
            "has_labels_train": True,
            "has_labels_val": True,
            "has_data_yaml": True,
            "train_image_count": MOCK_SPLIT["Train"],
            "val_image_count": MOCK_SPLIT["Val"],
            "train_label_count": 36666,
            "val_label_count": 9166,
            "class_count": 11,
            "class_names": [f"class{index}" for index in range(11)],
            "empty_labels": ["mock"] * MOCK_STATS["空标签数"],
            "missing_labels": [],
            "bad_images": ["mock"] * MOCK_STATS["异常图像数"],
            "warnings": [],
            "errors": [],
        }

    def _health_score(self, info: dict) -> int:
        if not info.get("exists"):
            return 0
        score = 100
        score -= 12 * len(info.get("errors", []) or [])
        score -= 4 * len(info.get("warnings", []) or [])
        score -= min(18, len(info.get("missing_labels", []) or []))
        score -= min(10, len(info.get("bad_images", []) or []))
        score -= min(8, len(info.get("empty_labels", []) or []) // 10)
        return max(0, min(100, score))

    def _structure_checks(self, info: dict) -> list[tuple[str, str]]:
        return [
            ("images/train", "正常" if info.get("has_images_train") else "缺失"),
            ("images/val", "正常" if info.get("has_images_val") else "缺失"),
            ("labels/train", "正常" if info.get("has_labels_train") else "缺失"),
            ("labels/val", "正常" if info.get("has_labels_val") else "缺失"),
            ("data.yaml", "正常" if info.get("has_data_yaml") else "缺失"),
        ]

    def _class_distribution(self, info: dict, class_count: int) -> list[tuple[int, int]]:
        counts = {index: 0 for index in range(class_count)}
        root = Path(info.get("root", ""))
        for label_root in [root / "labels" / "train", root / "labels" / "val"]:
            if not label_root.exists():
                continue
            for label_path in label_root.rglob("*.txt"):
                for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    counts[class_id] = counts.get(class_id, 0) + 1
        return sorted(counts.items())

    def _sample_previews(self, root: Path, limit: int = 4) -> list[tuple[Path, Path | None]]:
        samples: list[tuple[int, Path, Path | None]] = []
        image_roots = [
            root / "images" / "train",
            root / "images" / "val",
            root / "images",
            root,
        ]
        seen: set[Path] = set()
        for image_root in image_roots:
            if not image_root.exists():
                continue
            for image_path in sorted(
                path
                for path in image_root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ):
                if image_path in seen:
                    continue
                seen.add(image_path)
                label_path = self._label_for_image(root, image_root, image_path)
                resolved_label = label_path if label_path.exists() else None
                samples.append((self._sample_preview_score(resolved_label), image_path, resolved_label))
                if len(samples) >= max(limit * 8, limit):
                    break
            if len(samples) >= max(limit * 8, limit):
                break
        samples.sort(key=lambda item: (-item[0], item[1].name))
        return [(image_path, label_path) for _, image_path, label_path in samples[:limit]]

    def _sample_preview_score(self, label_path: Path | None) -> int:
        if not label_path or not label_path.exists():
            return 0
        small_boxes = 0
        medium_boxes = 0
        large_boxes = 0
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                area = max(0.0, float(parts[3])) * max(0.0, float(parts[4]))
            except ValueError:
                continue
            if area <= 0.03:
                small_boxes += 1
            elif area <= 0.18:
                medium_boxes += 1
            else:
                large_boxes += 1
        return small_boxes * 8 + medium_boxes * 4 - large_boxes

    def _label_for_image(self, dataset_root: Path, image_root: Path, image_path: Path) -> Path:
        try:
            relative = image_path.relative_to(dataset_root / "images")
            return dataset_root / "labels" / relative.with_suffix(".txt")
        except ValueError:
            pass
        try:
            relative = image_path.relative_to(image_root)
            if image_root.name in {"train", "val"} and image_root.parent.name == "images":
                return dataset_root / "labels" / image_root.name / relative.with_suffix(".txt")
            return dataset_root / "labels" / relative.with_suffix(".txt")
        except ValueError:
            return image_path.with_suffix(".txt")

    def _show_more_samples(self) -> None:
        info = self.inspection or {}
        root = Path(info.get("root", self.dataset_root.text().strip()))
        samples = self.current_gallery_samples or self.current_samples
        if not samples and root.exists():
            samples = self._sample_previews(root, limit=48)
        dialog = SampleGalleryDialog(samples, self)
        dialog.exec()
