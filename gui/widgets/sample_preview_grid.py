from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


YoloBox = tuple[int, float, float, float, float]


class SampleThumb(QWidget):
    def __init__(self, variant: int = 0, image_path: str | Path | None = None, label_path: str | Path | None = None) -> None:
        super().__init__()
        self.variant = variant
        self.image_path = Path(image_path) if image_path else None
        self.label_path = Path(label_path) if label_path else None
        self.pixmap = QPixmap(str(self.image_path)) if self.image_path and self.image_path.exists() else QPixmap()
        self.bboxes = self._read_yolo_bboxes()
        self.setCursor(Qt.CursorShape.PointingHandCursor if not self.pixmap.isNull() else Qt.CursorShape.ArrowCursor)
        self.setMinimumSize(120, 112)
        self.setMaximumHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#F8FAFC"))
        painter.drawRoundedRect(rect, 8, 8)

        if not self.pixmap.isNull():
            source_rect = self._thumbnail_source_rect(rect, self.pixmap.width(), self.pixmap.height())
            painter.save()
            painter.setClipRect(rect)
            painter.drawPixmap(rect, self.pixmap, source_rect)
            self._draw_bboxes_for_source(painter, rect, source_rect, thumbnail=True)
            painter.restore()
            return

        self._draw_placeholder(painter, rect)

    def _draw_placeholder(self, painter: QPainter, rect: QRectF) -> None:
        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0, QColor("#6B7280"))
        gradient.setColorAt(0.5, QColor("#2F343A"))
        gradient.setColorAt(1, QColor("#A3A3A3"))
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)

        painter.setBrush(QColor(230, 230, 230, 70))
        painter.drawRoundedRect(rect.adjusted(18, 22, -18, -24), 6, 6)
        painter.setBrush(QColor(245, 245, 245, 140))
        object_rect = rect.adjusted(46, 66, -26, -22) if self.variant % 2 else rect.adjusted(44, 38, -38, -38)
        painter.drawRoundedRect(object_rect, 5, 5)

        painter.setPen(QPen(QColor("#22C55E"), 1.4))
        painter.drawRect(object_rect.adjusted(-8, -8, 8, 8))
        if self.variant in {1, 3}:
            painter.drawRect(rect.adjusted(20, 26, -56, -60))

    def _fit_rect(self, outer: QRectF, image_width: int, image_height: int) -> QRectF:
        if image_width <= 0 or image_height <= 0:
            return outer
        image_ratio = image_width / image_height
        outer_ratio = outer.width() / max(1.0, outer.height())
        if image_ratio >= outer_ratio:
            width = outer.width()
            height = width / image_ratio
        else:
            height = outer.height()
            width = height * image_ratio
        left = outer.left() + (outer.width() - width) / 2
        top = outer.top() + (outer.height() - height) / 2
        return QRectF(left, top, width, height)

    def _cover_source_rect(self, target: QRectF, image_width: int, image_height: int) -> QRectF:
        if image_width <= 0 or image_height <= 0 or target.width() <= 0 or target.height() <= 0:
            return QRectF(0, 0, max(1, image_width), max(1, image_height))
        image_ratio = image_width / image_height
        target_ratio = target.width() / target.height()
        if image_ratio > target_ratio:
            source_height = image_height
            source_width = source_height * target_ratio
            left = (image_width - source_width) / 2
            return QRectF(left, 0, source_width, source_height)
        source_width = image_width
        source_height = source_width / target_ratio
        top = (image_height - source_height) / 2
        return QRectF(0, top, source_width, source_height)

    def _thumbnail_source_rect(self, target: QRectF, image_width: int, image_height: int) -> QRectF:
        focus_boxes = [box for box in self.bboxes if self._box_area(box) <= 0.18]
        if not focus_boxes:
            return self._cover_source_rect(target, image_width, image_height)

        left = min((x - w / 2) * image_width for _, x, y, w, h in focus_boxes)
        top = min((y - h / 2) * image_height for _, x, y, w, h in focus_boxes)
        right = max((x + w / 2) * image_width for _, x, y, w, h in focus_boxes)
        bottom = max((y + h / 2) * image_height for _, x, y, w, h in focus_boxes)

        margin_x = max(140.0, (right - left) * 0.35)
        margin_y = max(100.0, (bottom - top) * 0.45)
        left = max(0.0, left - margin_x)
        top = max(0.0, top - margin_y)
        right = min(float(image_width), right + margin_x)
        bottom = min(float(image_height), bottom + margin_y)

        return self._expand_to_target_ratio(QRectF(left, top, max(1.0, right - left), max(1.0, bottom - top)), target, image_width, image_height)

    def _expand_to_target_ratio(self, source: QRectF, target: QRectF, image_width: int, image_height: int) -> QRectF:
        target_ratio = target.width() / max(1.0, target.height())
        source_ratio = source.width() / max(1.0, source.height())
        left = source.left()
        top = source.top()
        width = source.width()
        height = source.height()
        if source_ratio > target_ratio:
            new_height = width / target_ratio
            top -= (new_height - height) / 2
            height = new_height
        else:
            new_width = height * target_ratio
            left -= (new_width - width) / 2
            width = new_width

        min_width = min(float(image_width), image_width * 0.42)
        min_height = min(float(image_height), image_height * 0.34)
        if width < min_width:
            left -= (min_width - width) / 2
            width = min_width
        if height < min_height:
            top -= (min_height - height) / 2
            height = min_height

        if left < 0:
            left = 0.0
        if top < 0:
            top = 0.0
        if left + width > image_width:
            left = max(0.0, image_width - width)
        if top + height > image_height:
            top = max(0.0, image_height - height)
        width = min(width, float(image_width))
        height = min(height, float(image_height))
        return QRectF(left, top, width, height)

    def _read_yolo_bboxes(self) -> list[YoloBox]:
        if not self.label_path or not self.label_path.exists():
            return []
        boxes: list[YoloBox] = []
        for line in self.label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                x, y, w, h = [float(value) for value in parts[1:5]]
            except ValueError:
                continue
            boxes.append((class_id, x, y, w, h))
        return boxes

    def _draw_bboxes(self, painter: QPainter, image_rect: QRectF) -> None:
        if not self.bboxes:
            return
        painter.setPen(QPen(QColor("#22C55E"), 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for class_id, x, y, w, h in self.bboxes[:12]:
            left = image_rect.left() + (x - w / 2) * image_rect.width()
            top = image_rect.top() + (y - h / 2) * image_rect.height()
            width = w * image_rect.width()
            height = h * image_rect.height()
            painter.drawRect(QRectF(left, top, width, height))

    def _draw_bboxes_for_source(self, painter: QPainter, target_rect: QRectF, source_rect: QRectF, thumbnail: bool = False) -> None:
        if not self.bboxes or self.pixmap.isNull() or source_rect.width() <= 0 or source_rect.height() <= 0:
            return
        painter.setPen(QPen(QColor("#22C55E"), 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        image_width = self.pixmap.width()
        image_height = self.pixmap.height()
        boxes = [box for box in self.bboxes if self._box_area(box) <= 0.18] if thumbnail else self.bboxes
        if not boxes:
            boxes = self.bboxes
        for class_id, x, y, w, h in boxes[:24]:
            box_left = (x - w / 2) * image_width
            box_top = (y - h / 2) * image_height
            box_width = w * image_width
            box_height = h * image_height

            left = target_rect.left() + (box_left - source_rect.left()) / source_rect.width() * target_rect.width()
            top = target_rect.top() + (box_top - source_rect.top()) / source_rect.height() * target_rect.height()
            width = box_width / source_rect.width() * target_rect.width()
            height = box_height / source_rect.height() * target_rect.height()
            painter.drawRect(QRectF(left, top, width, height))

    def _box_area(self, box: YoloBox) -> float:
        return max(0.0, box[3]) * max(0.0, box[4])

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.pixmap.isNull():
            SamplePreviewDialog(self.image_path, self.label_path, self).exec()
            return
        super().mousePressEvent(event)


class LargeSampleView(QWidget):
    def __init__(self, image_path: Path, label_path: Path | None = None) -> None:
        super().__init__()
        self.image_path = Path(image_path)
        self.label_path = Path(label_path) if label_path else None
        self.pixmap = QPixmap(str(self.image_path))
        self.bboxes = self._read_yolo_bboxes()
        self.setMinimumSize(840, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#F8FAFC"))
        if self.pixmap.isNull():
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "图像加载失败")
            return
        outer = QRectF(self.rect()).adjusted(16, 16, -16, -16)
        image_rect = self._fit_rect(outer, self.pixmap.width(), self.pixmap.height())
        painter.drawPixmap(image_rect.toRect(), self.pixmap)
        self._draw_bboxes(painter, image_rect)

    def _fit_rect(self, outer: QRectF, image_width: int, image_height: int) -> QRectF:
        if image_width <= 0 or image_height <= 0:
            return outer
        image_ratio = image_width / image_height
        outer_ratio = outer.width() / max(1.0, outer.height())
        if image_ratio >= outer_ratio:
            width = outer.width()
            height = width / image_ratio
        else:
            height = outer.height()
            width = height * image_ratio
        left = outer.left() + (outer.width() - width) / 2
        top = outer.top() + (outer.height() - height) / 2
        return QRectF(left, top, width, height)

    def _read_yolo_bboxes(self) -> list[YoloBox]:
        if not self.label_path or not self.label_path.exists():
            return []
        boxes: list[YoloBox] = []
        for line in self.label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
                x, y, w, h = [float(value) for value in parts[1:5]]
            except ValueError:
                continue
            boxes.append((class_id, x, y, w, h))
        return boxes

    def _draw_bboxes(self, painter: QPainter, image_rect: QRectF) -> None:
        painter.setPen(QPen(QColor("#22C55E"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for class_id, x, y, w, h in self.bboxes[:60]:
            left = image_rect.left() + (x - w / 2) * image_rect.width()
            top = image_rect.top() + (y - h / 2) * image_rect.height()
            width = w * image_rect.width()
            height = h * image_rect.height()
            painter.drawRect(QRectF(left, top, width, height))


class SamplePreviewDialog(QDialog):
    def __init__(self, image_path: Path | None, label_path: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(Path(image_path).name if image_path else "样本预览")
        self.resize(980, 680)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        if image_path:
            view = LargeSampleView(Path(image_path), label_path)
            layout.addWidget(view, 1)
        else:
            label = QLabel("暂无样本预览")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, 1)

        close = QPushButton("关闭")
        close.setObjectName("secondaryButton")
        close.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close)
        layout.addLayout(footer)


class SampleGalleryDialog(QDialog):
    def __init__(self, samples: list[tuple[Path, Path | None]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"样本预览（{len(samples)} 张）")
        self.resize(1120, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel("点击任意缩略图可放大查看 bbox")
        header.setObjectName("emptyDetail")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        if samples:
            for index, (image_path, label_path) in enumerate(samples):
                thumb = SampleThumb(index, image_path, label_path)
                thumb.setMinimumSize(190, 145)
                grid.addWidget(thumb, index // 5, index % 5)
        else:
            empty = QLabel("暂无样本预览")
            empty.setObjectName("emptyDetail")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(empty, 0, 0)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        close = QPushButton("关闭")
        close.setObjectName("secondaryButton")
        close.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch(1)
        footer.addWidget(close)
        layout.addLayout(footer)


class SamplePreviewGrid(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(12)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.set_mock_samples()

    def set_mock_samples(self) -> None:
        self._clear()
        for index in range(4):
            self._layout.addWidget(SampleThumb(index), 1, Qt.AlignmentFlag.AlignVCenter)

    def set_samples(self, samples: list[tuple[Path, Path | None]]) -> None:
        self._clear()
        if not samples:
            self.set_empty("暂无样本预览")
            return
        for index, (image_path, label_path) in enumerate(samples[:4]):
            self._layout.addWidget(SampleThumb(index, image_path, label_path), 1, Qt.AlignmentFlag.AlignVCenter)

    def set_empty(self, text: str = "暂无样本预览") -> None:
        self._clear()
        box = QVBoxLayout()
        label = QLabel(text)
        label.setObjectName("emptyDetail")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(label)
        self._layout.addLayout(box)

    def _clear(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().setParent(None)
