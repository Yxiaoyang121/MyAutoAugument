from __future__ import annotations

import cv2
import numpy as np

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QSizePolicy, QVBoxLayout


class PreviewImageLabel(QLabel):
    clicked = Signal(object)

    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self.placeholder = placeholder
        self.setObjectName("previewImage")
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.setMinimumHeight(170)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self._image_bgr: np.ndarray | None = None
        self.setToolTip("??????")

    def sizeHint(self) -> QSize:
        return QSize(420, 170)

    def minimumSizeHint(self) -> QSize:
        return QSize(180, 140)

    def set_image(self, image_bgr: np.ndarray | None, placeholder: str | None = None) -> None:
        self._image_bgr = image_bgr.copy() if image_bgr is not None else None
        if self._image_bgr is None:
            self.setPixmap(QPixmap())
            self.setText(placeholder or "????")
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        self.setText("")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._image_bgr is not None:
            self.clicked.emit(self._image_bgr.copy())
        super().mousePressEvent(event)

    def _update_pixmap(self) -> None:
        if self._image_bgr is None or self.width() <= 4 or self.height() <= 4:
            return
        rgb = cv2.cvtColor(self._image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        bytes_per_line = 3 * width
        qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class PreviewImageDialog(QDialog):
    def __init__(self, title: str, image_bgr: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(900, 560)
        self.resize(1180, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.image = PreviewImageLabel("????")
        self.image.setMinimumHeight(460)
        self.image.set_image(image_bgr)
        layout.addWidget(self.image, 1)

        close_button = QPushButton("??")
        close_button.setObjectName("secondaryButton")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
