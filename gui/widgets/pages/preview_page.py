from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.adapters import AugmentationAdapter, PolicyAdapter
from gui.controllers import PreviewController
from gui.widgets.preview_compare import PreviewImageDialog, PreviewImageLabel
from gui.widgets.preview_quality_panel import PreviewQualityPanel
from gui.widgets.quality_distribution import QualityDistributionPanel
from gui.widgets.section_card import SectionCard


MOCK_IMAGE_PATH = r"D:\AutoAugment\dataset\images\train\img_0001.png"
MOCK_LABEL_PATH = r"D:\AutoAugment\dataset\labels\train\img_0001.txt"


class PreviewPage(QWidget):
    policy_changed = Signal(dict)
    policy_editor_requested = Signal()

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = Path(project_root)
        self.dataset_path = self.project_root / "dataset"
        self.policy: dict = {
            "name": "gui_policy",
            "operations": [
                {"name": "brightness", "prob": 0.5, "strength": 0.5, "params": {"max_delta": 0.25}},
                {"name": "contrast", "prob": 0.5, "strength": 0.5, "params": {"max_delta": 0.5}},
                {"name": "horizontal_flip", "prob": 0.5, "strength": 0.5, "params": {}},
                {"name": "rotate", "prob": 0.5, "strength": 0.5, "params": {"max_angle": 15.0}},
            ],
        }
        self.adapter = AugmentationAdapter()
        self.controller = PreviewController()
        self.policy_adapter = PolicyAdapter()
        self._rendering = False
        self._build_ui()
        self._load_default_paths()
        self._render()

    def _build_ui(self) -> None:
        self.setObjectName("previewPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(8)

        self._build_input_card(layout)
        self._build_preview_area(layout)
        self._build_policy_card(layout)
        self._build_distribution_card(layout)
        self._build_parameter_strip(layout)

    def _build_input_card(self, parent: QVBoxLayout) -> None:
        card = SectionCard("")
        card.setObjectName("previewInputCard")
        card.setFixedHeight(58)
        row = QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)

        self.image_path = QLineEdit()
        self.image_path.setObjectName("previewPathInput")
        self.image_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.image_path.setText(MOCK_IMAGE_PATH)
        self.label_path = QLineEdit()
        self.label_path.setObjectName("previewPathInput")
        self.label_path.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.label_path.setText(MOCK_LABEL_PATH)
        image_button = QPushButton("浏览")
        image_button.setObjectName("secondaryButton")
        label_button = QPushButton("浏览")
        label_button.setObjectName("secondaryButton")
        run_button = QPushButton("运行预览")
        run_button.setObjectName("primaryButton")

        image_button.clicked.connect(self._choose_image)
        label_button.clicked.connect(self._choose_label)
        run_button.clicked.connect(self._render)

        row.addWidget(self._label("图像路径"))
        row.addWidget(self.image_path, 3)
        row.addWidget(image_button)
        row.addWidget(self._label("标签路径"))
        row.addWidget(self.label_path, 3)
        row.addWidget(label_button)
        row.addWidget(run_button)
        card.body.addLayout(row)
        parent.addWidget(card)

    def _build_preview_area(self, parent: QVBoxLayout) -> None:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        original = SectionCard("原图预览（with bbox）")
        augmented = SectionCard("增强后预览（with bbox）")
        original.setMinimumHeight(240)
        augmented.setMinimumHeight(240)
        original.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        augmented.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.original_image = PreviewImageLabel("原图预览")
        self.augmented_image = PreviewImageLabel("增强后预览")
        self.original_image.clicked.connect(lambda image: self._show_image_dialog("?????with bbox?", image))
        self.augmented_image.clicked.connect(lambda image: self._show_image_dialog("??????with bbox?", image))
        original.body.addWidget(self.original_image, 1)
        augmented.body.addWidget(self.augmented_image, 1)

        self.quality_panel = PreviewQualityPanel()
        self.quality_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.quality_panel.export_requested.connect(self._export_augmented_image)
        grid.addWidget(original, 0, 0)
        grid.addWidget(augmented, 0, 1)
        grid.addWidget(self.quality_panel, 0, 2)
        grid.setColumnStretch(0, 9)
        grid.setColumnStretch(1, 9)
        grid.setColumnStretch(2, 5)
        parent.addLayout(grid, 1)

    def _build_policy_card(self, parent: QVBoxLayout) -> None:
        card = SectionCard("当前增强策略")
        card.outer_layout.setContentsMargins(14, 8, 14, 10)
        card.outer_layout.setSpacing(6)
        card.body.setSpacing(6)
        card.setFixedHeight(80)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.policy_badges = QHBoxLayout()
        self.policy_badges.setContentsMargins(0, 0, 0, 0)
        self.policy_badges.setSpacing(8)
        row.addLayout(self.policy_badges, 1)

        load_button = QPushButton("加载策略")
        load_button.setObjectName("secondaryButton")
        edit_button = QPushButton("去策略编辑")
        edit_button.setObjectName("secondaryButton")
        rerun_button = QPushButton("重新应用")
        rerun_button.setObjectName("primaryButton")
        load_button.clicked.connect(self._load_policy_file)
        edit_button.clicked.connect(self.policy_editor_requested.emit)
        rerun_button.clicked.connect(self._render)
        row.addWidget(load_button)
        row.addWidget(edit_button)
        row.addWidget(rerun_button)
        card.body.addLayout(row)

        self.preview_status = QLabel("策略变更会立即用于增强预览，无需先保存 JSON。")
        self.preview_status.setObjectName("previewStatus")
        parent.addWidget(card)

    def _build_distribution_card(self, parent: QVBoxLayout) -> None:
        self.distribution_panel = QualityDistributionPanel()
        self.distribution_panel.setFixedHeight(104)
        parent.addWidget(self.distribution_panel)

    def _build_parameter_strip(self, parent: QVBoxLayout) -> None:
        strip = SectionCard("")
        strip.setObjectName("previewParamStrip")
        strip.setFixedHeight(38)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        self.parameter_labels: dict[str, QLabel] = {}
        items = [
            ("image_size", "?????-"),
            ("bbox_draw", "bbox ?????"),
            ("bbox_threshold", "bbox ?????0.25"),
            ("interpolation", "?????bilinear"),
            ("fill_mode", "?????border"),
        ]
        for index, (key, text) in enumerate(items):
            if index:
                sep = QLabel("|")
                sep.setObjectName("previewParamSep")
                row.addWidget(sep)
            label = QLabel(text)
            label.setObjectName("previewParamItem")
            self.parameter_labels[key] = label
            row.addWidget(label)
        row.addStretch(1)
        strip.body.addLayout(row)
        parent.addWidget(strip)

    def set_policy(self, policy: dict) -> None:
        self.policy = policy or {"name": "gui_policy", "operations": []}
        self._refresh_policy_badges()
        self._render()

    def set_dataset_path(self, path: str) -> None:
        if path:
            self.dataset_path = Path(path)
            if not self.image_path.text().strip():
                self._load_default_paths()
            self._render()

    def current_policy_dict(self) -> dict:
        return self.policy

    def _label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("previewFieldLabel")
        return label

    def _show_image_dialog(self, title: str, image: np.ndarray) -> None:
        dialog = PreviewImageDialog(title, image, self)
        dialog.exec()

    def _load_default_paths(self) -> None:
        if self.image_path.text().strip():
            return
        candidates = self._image_candidates()
        if not candidates:
            return
        image = candidates[0]
        label = self._guess_label_path(image)
        self.image_path.setText(str(image))
        self.label_path.setText(str(label) if label.exists() else "")

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", str(self.dataset_path), "图像 (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.image_path.setText(path)
            label = self._guess_label_path(Path(path))
            if label.exists():
                self.label_path.setText(str(label))
            self._render()

    def _choose_label(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 标签", str(self.dataset_path), "YOLO 标签 (*.txt)")
        if path:
            self.label_path.setText(path)
            self._render()

    def _random_sample(self) -> None:
        candidates = self._image_candidates()
        if not candidates:
            self.image_path.setText("")
            self.label_path.setText("")
            self.preview_status.setText(f"未在数据集目录找到图像，已显示内置 mock 样本：{self.dataset_path}")
            self._render()
            return
        image = random.choice(candidates)
        label = self._guess_label_path(image)
        self.image_path.setText(str(image))
        self.label_path.setText(str(label) if label.exists() else "")
        self._render()

    def _load_policy_file(self) -> None:
        default_dir = self.project_root / "outputs" / "gui_policies"
        start_dir = default_dir if default_dir.exists() else self.project_root
        path, _ = QFileDialog.getOpenFileName(self, "加载增强策略", str(start_dir), "策略文件 (*.json *.yaml *.yml)")
        if not path:
            return
        try:
            policy = self.policy_adapter.load_policy_dict(path)
            self.set_policy(policy)
            self.policy_changed.emit(policy)
            self.preview_status.setText(f"已加载并应用策略：{Path(path).name}")
        except Exception as exc:
            QMessageBox.warning(self, "加载策略失败", str(exc))

    def _render(self) -> None:
        if self._rendering or not hasattr(self, "original_image"):
            return
        self._rendering = True
        try:
            sample = self._load_sample_or_mock()
            is_mock = bool(sample.get("mock"))
            if is_mock:
                augmented = self._mock_augmented_sample(sample)
            else:
                preview = self.controller.run_preview(
                    self.image_path.text().strip(),
                    self.label_path.text().strip() or None,
                    self.policy,
                    deterministic_preview=True,
                )
                sample = preview["sample"]
                augmented = preview["augmented"]
            original_vis = self.adapter.draw_bboxes(sample["image"], sample["labels"], sample["bboxes"])
            augmented_vis = self.adapter.draw_bboxes(augmented["image"], augmented["labels"], augmented["bboxes"])
            if is_mock:
                original_vis = self._draw_mock_bboxes(sample["image"], sample["bboxes"], [0.96, 0.93])
                augmented_vis = self._draw_mock_bboxes(augmented["image"], augmented["bboxes"], [0.95, 0.92])
            self.original_image.set_image(original_vis)
            self.augmented_image.set_image(augmented_vis)
            self._set_mock_quality() if is_mock else self._set_quality(sample, augmented)
            self._update_parameter_strip(sample, augmented)
            self._refresh_policy_badges()
        except Exception as exc:
            self.original_image.set_image(None, "图像加载失败")
            self.augmented_image.set_image(None, "增强预览失败")
            self.preview_status.setText(str(exc))
        finally:
            self._rendering = False

    def _load_sample_or_mock(self) -> dict:
        image_text = self.image_path.text().strip()
        image_path = Path(image_text) if image_text else None
        if image_path and image_path.exists():
            return self.adapter.load_yolo_sample(image_path, self.label_path.text().strip() or None)
        return self._mock_sample()

    def _mock_sample(self) -> dict:
        height, width = 330, 760
        image = np.full((height, width, 3), 64, dtype=np.uint8)
        cv2.rectangle(image, (22, 30), (width - 22, height - 34), (118, 118, 116), 18)
        cv2.rectangle(image, (56, 66), (width - 58, height - 78), (44, 46, 48), -1)
        cv2.rectangle(image, (314, 120), (408, 212), (210, 214, 210), -1)
        cv2.rectangle(image, (586, 108), (675, 230), (214, 214, 210), -1)
        cv2.circle(image, (150, 170), 36, (92, 96, 94), -1)
        cv2.circle(image, (690, 270), 12, (216, 216, 212), -1)
        cv2.circle(image, (95, 82), 16, (210, 210, 206), -1)
        for x in range(86, width - 90, 72):
            cv2.circle(image, (x, 86), 5, (150, 156, 150), -1)
            cv2.circle(image, (x, 270), 5, (150, 156, 150), -1)
        for x in range(80, width - 80, 46):
            cv2.rectangle(image, (x, 118), (x + 12, 228), (84, 88, 86), -1)
        noise = np.random.default_rng(3).normal(0, 7, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        labels = np.arange(11, dtype=np.int64)
        bboxes = np.array(
            [
                [128, 128, 248, 246],
                [590, 130, 686, 256],
                [312, 116, 410, 218],
                [84, 68, 116, 100],
                [452, 82, 492, 112],
                [80, 246, 118, 282],
                [510, 246, 548, 280],
                [690, 260, 724, 294],
                [268, 78, 302, 106],
                [432, 220, 462, 248],
                [178, 82, 210, 110],
            ],
            dtype=np.float32,
        )
        return {"image": image, "labels": labels, "bboxes": bboxes, "image_path": "mock://industrial-sample", "mock": True}

    def _mock_augmented_sample(self, sample: dict) -> dict:
        image = cv2.convertScaleAbs(sample["image"], alpha=1.08, beta=8)
        center = (image.shape[1] / 2, image.shape[0] / 2)
        matrix = cv2.getRotationMatrix2D(center, -3.5, 0.98)
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        bboxes = np.array(
            [
                [138, 142, 252, 262],
                [596, 142, 686, 266],
                [318, 122, 410, 220],
                [90, 74, 120, 105],
                [456, 90, 494, 118],
                [84, 252, 120, 286],
                [512, 250, 548, 284],
                [690, 262, 724, 296],
                [274, 84, 306, 112],
                [436, 224, 466, 252],
                [182, 88, 214, 116],
            ],
            dtype=np.float32,
        )
        return {**sample, "image": rotated, "bboxes": bboxes}

    def _draw_mock_bboxes(self, image: np.ndarray, bboxes: np.ndarray, scores: list[float]) -> np.ndarray:
        out = image.copy()
        for index, box in enumerate(np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box]
            cv2.rectangle(out, (x1, y1), (x2, y2), (34, 197, 94), 2)
            if index < 2:
                text = f"class {scores[index]:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), (22, 163, 74), -1)
                cv2.putText(out, text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _set_mock_quality(self) -> None:
        self.quality_panel.set_metrics(
            original_total=11,
            augmented_total=11,
            retention=1.0,
            valid_rate=1.0,
            pixel_delta="22.93",
            invalid_count=0,
        )
        self.preview_status.setText("当前策略：gui_policy / 操作数=4 / bbox 保留率=100.0%")

    def _set_quality(self, sample: dict, augmented: dict) -> None:
        before = self.adapter.bbox_validity(sample["image"], sample["bboxes"])
        after = self.adapter.bbox_validity(augmented["image"], augmented["bboxes"])
        before_total = int(before.get("total", 0))
        after_total = int(after.get("total", 0))
        valid_after = int(after.get("valid_count", 0))
        invalid_after = int(after.get("invalid_count", 0))
        retention = after_total / before_total if before_total else 1.0
        valid_rate = valid_after / after_total if after_total else 1.0
        if sample["image"].shape == augmented["image"].shape:
            pixel_delta = f"{cv2.absdiff(sample['image'], augmented['image']).mean():.2f}"
        else:
            pixel_delta = "尺寸变化"
        self.quality_panel.set_metrics(
            original_total=before_total,
            augmented_total=after_total,
            retention=retention,
            valid_rate=valid_rate,
            pixel_delta=pixel_delta,
            invalid_count=invalid_after,
        )
        self.preview_status.setText(
            f"当前策略：{self.policy.get('name', 'gui_policy')} / 操作数={len(self.policy.get('operations', []))} / bbox 保留率={retention:.2%}"
        )


    def _update_parameter_strip(self, sample: dict, augmented: dict) -> None:
        if not hasattr(self, "parameter_labels"):
            return
        src_h, src_w = sample["image"].shape[:2]
        aug_h, aug_w = augmented["image"].shape[:2]
        if (src_w, src_h) == (aug_w, aug_h):
            size_text = f"?????{src_w} x {src_h}"
        else:
            size_text = f"??????? {src_w} x {src_h} / ?? {aug_w} x {aug_h}"
        self.parameter_labels["image_size"].setText(size_text)
        self.parameter_labels["bbox_draw"].setText("bbox ?????")
        self.parameter_labels["bbox_threshold"].setText("bbox ?????0.25")

    def _refresh_policy_badges(self) -> None:
        if not hasattr(self, "policy_badges"):
            return
        while self.policy_badges.count():
            item = self.policy_badges.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        operations = self.policy.get("operations", []) or []
        if not operations:
            badge = QLabel("空策略")
            badge.setObjectName("previewPolicyBadge")
            self.policy_badges.addWidget(badge)
            return
        for operation in operations:
            name = str(operation.get("name", "unknown"))
            prob = operation.get("prob", operation.get("probability", 1.0))
            badge = QLabel(f"{name} ({self._format_number(prob)})")
            badge.setObjectName("previewPolicyBadge")
            self.policy_badges.addWidget(badge)
        self.policy_badges.addStretch(1)

    def _export_augmented_image(self) -> None:
        self.preview_status.setText("导出增强图像功能已预留：后续会保存增强图像与对应 YOLO 标签。")

    def _image_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        for root in [self.dataset_path / "images" / "train", self.dataset_path / "images" / "val", self.dataset_path / "images", self.dataset_path]:
            if root.exists():
                candidates.extend(path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        return sorted(candidates)

    def _guess_label_path(self, image: Path) -> Path:
        try:
            relative = image.relative_to(self.dataset_path / "images")
            return self.dataset_path / "labels" / relative.with_suffix(".txt")
        except ValueError:
            return image.with_suffix(".txt")

    def _format_number(self, value) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)
