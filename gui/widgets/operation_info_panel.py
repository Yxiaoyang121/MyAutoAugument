from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QTransform
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLayout, QVBoxLayout, QWidget

from gui.widgets.section_card import SectionCard


OPERATION_INFO: dict[str, dict] = {
    "brightness": {
        "intro": "随机调整图像亮度，模拟光照强弱变化。",
        "scenes": ["曝光鲁棒性", "低照度", "目标检测"],
        "bbox": "不改变",
        "range": "max_delta 0.10 ~ 0.50",
        "cost": "★☆☆☆☆  很低",
        "kind": "brightness",
    },
    "contrast": {
        "intro": "围绕图像均值调整对比度，增强纹理和边缘差异。",
        "scenes": ["低对比", "纹理增强", "目标检测"],
        "bbox": "不改变",
        "range": "max_delta 0.20 ~ 0.80",
        "cost": "★☆☆☆☆  很低",
        "kind": "contrast",
    },
    "gamma": {
        "intro": "进行非线性亮度调整，改善过暗或过亮样本。",
        "scenes": ["曝光补偿", "低照度", "泛化增强"],
        "bbox": "不改变",
        "range": "gamma 0.7 ~ 1.5",
        "cost": "★☆☆☆☆  很低",
        "kind": "brightness",
    },
    "hue": {
        "intro": "调整色相分布，用于彩色数据的颜色扰动。",
        "scenes": ["颜色鲁棒性", "光照变化", "分类/检测"],
        "bbox": "不改变",
        "range": "轻微扰动",
        "cost": "★☆☆☆☆  很低",
        "kind": "contrast",
    },
    "saturation": {
        "intro": "调整饱和度，模拟不同成像和照明条件。",
        "scenes": ["颜色鲁棒性", "工业成像", "目标检测"],
        "bbox": "不改变",
        "range": "轻微到中等",
        "cost": "★☆☆☆☆  很低",
        "kind": "contrast",
    },
    "horizontal_flip": {
        "intro": "水平翻转图像，并同步更新 bbox 坐标。",
        "scenes": ["方向鲁棒性", "几何多样性", "目标检测"],
        "bbox": "会改变",
        "range": "prob 0.30 ~ 0.70",
        "cost": "★☆☆☆☆  很低",
        "kind": "flip",
    },
    "vertical_flip": {
        "intro": "垂直翻转图像，并同步更新 bbox 坐标。",
        "scenes": ["方向鲁棒性", "几何多样性", "目标检测"],
        "bbox": "会改变",
        "range": "按任务谨慎使用",
        "cost": "★☆☆☆☆  很低",
        "kind": "flip",
    },
    "rotate": {
        "intro": "随机旋转图像，增加方向多样性。",
        "scenes": ["方向鲁棒性", "几何多样性", "目标检测"],
        "bbox": "会改变",
        "range": "5° ~ 30°",
        "cost": "★★★☆☆  中等",
        "kind": "rotate",
    },
    "scale": {
        "intro": "围绕图像中心缩放，模拟目标尺度变化。",
        "scenes": ["尺度鲁棒性", "小目标", "目标检测"],
        "bbox": "会改变",
        "range": "max_delta 0.05 ~ 0.30",
        "cost": "★★☆☆☆  较低",
        "kind": "scale",
    },
    "translate": {
        "intro": "平移图像内容，模拟目标位置偏移。",
        "scenes": ["位置鲁棒性", "边缘目标", "目标检测"],
        "bbox": "会改变",
        "range": "max_translate 0.03 ~ 0.15",
        "cost": "★★☆☆☆  较低",
        "kind": "translate",
    },
    "shear": {
        "intro": "错切图像，模拟轻微视角和机械偏差。",
        "scenes": ["几何扰动", "工业偏差", "目标检测"],
        "bbox": "会改变",
        "range": "小角度使用",
        "cost": "★★★☆☆  中等",
        "kind": "rotate",
    },
    "gaussian_noise": {
        "intro": "加入高斯噪声，模拟传感器噪声和成像干扰。",
        "scenes": ["噪声鲁棒性", "工业相机", "低信噪比"],
        "bbox": "不改变",
        "range": "max_std 0.02 ~ 0.12",
        "cost": "★☆☆☆☆  很低",
        "kind": "noise",
    },
    "blur": {
        "intro": "轻微模糊图像，模拟离焦或成像细节损失。",
        "scenes": ["成像模糊", "鲁棒性", "工业检测"],
        "bbox": "不改变",
        "range": "kernel 3 ~ 9",
        "cost": "★★☆☆☆  较低",
        "kind": "blur",
    },
    "motion_blur": {
        "intro": "模拟运动模糊，用于传送带或相机抖动场景。",
        "scenes": ["运动模糊", "工业产线", "鲁棒性"],
        "bbox": "不改变",
        "range": "kernel 3 ~ 11",
        "cost": "★★★☆☆  中等",
        "kind": "blur",
    },
}


class OperationPreview(QWidget):
    def __init__(self, kind: str = "default") -> None:
        super().__init__()
        self.kind = kind
        self.setMinimumHeight(82)

    def set_kind(self, kind: str) -> None:
        self.kind = kind
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = 62
        top = 10
        left = QRectF(10, top, size, size)
        right = QRectF(self.width() - size - 10, top, size, size)
        self._draw_tile(painter, left, "before")
        self._draw_arrow(painter)
        self._draw_tile(painter, right, self.kind)

    def _draw_tile(self, painter: QPainter, rect: QRectF, mode: str) -> None:
        painter.setPen(QPen(QColor("#E2E8F0"), 1))
        painter.setBrush(QColor("#F8FAFC"))
        painter.drawRoundedRect(rect, 6, 6)
        for i in range(1, 5):
            x = rect.left() + rect.width() * i / 5
            y = rect.top() + rect.height() * i / 5
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        box = QRectF(rect.center().x() - 16, rect.center().y() - 16, 32, 32)
        if mode == "brightness":
            painter.setBrush(QColor("#E0F2FE"))
            painter.setPen(QPen(QColor("#0F766E"), 1.6))
            painter.drawRect(box)
            painter.setBrush(QColor(255, 255, 255, 90))
            painter.drawEllipse(box.adjusted(10, 10, -10, -10))
            return
        if mode == "contrast":
            painter.setBrush(QColor("#CBD5E1"))
            painter.setPen(QPen(QColor("#0F766E"), 1.6))
            painter.drawRect(box)
            painter.setBrush(QColor("#334155"))
            painter.drawRect(box.adjusted(10, 10, -22, -8))
            return
        if mode == "noise":
            painter.setBrush(QColor("#E2E8F0"))
            painter.setPen(QPen(QColor("#0F766E"), 1.6))
            painter.drawRect(box)
            painter.setPen(QPen(QColor("#64748B"), 1.3))
            for offset in [(8, 8), (22, 12), (34, 28), (16, 36), (30, 18)]:
                painter.drawPoint(QPointF(box.left() + offset[0], box.top() + offset[1]))
            return
        if mode == "blur":
            painter.setBrush(QColor("#E2E8F0"))
            painter.setPen(QPen(QColor("#0F766E"), 1.2, Qt.PenStyle.DashLine))
            painter.drawRect(box.adjusted(-2, -2, 2, 2))
            painter.drawRect(box.adjusted(2, 2, -2, -2))
            return
        if mode == "rotate":
            painter.save()
            transform = QTransform()
            transform.translate(rect.center().x(), rect.center().y())
            transform.rotate(-18)
            transform.translate(-rect.center().x(), -rect.center().y())
            painter.setTransform(transform, True)
            painter.setPen(QPen(QColor("#0F766E"), 1.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)
            painter.restore()
            painter.setPen(QPen(QColor("#0F766E"), 1, Qt.PenStyle.DashLine))
            painter.drawRect(box.adjusted(-8, -8, 8, 8))
            return
        if mode in {"flip", "scale", "translate"}:
            shift = 12 if mode == "translate" else 0
            scale = 1.22 if mode == "scale" else 1.0
            target = QRectF(
                box.center().x() - box.width() * scale / 2 + shift,
                box.center().y() - box.height() * scale / 2,
                box.width() * scale,
                box.height() * scale,
            )
            painter.setPen(QPen(QColor("#0F766E"), 1.6))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(target)
            return

        painter.setPen(QPen(QColor("#0F766E"), 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(box)

    def _draw_arrow(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("#0F172A"), 2))
        center_x = self.width() / 2
        y = 42
        painter.drawLine(QPointF(center_x - 12, y), QPointF(center_x + 12, y))
        painter.drawLine(QPointF(center_x + 12, y), QPointF(center_x + 5, y - 7))
        painter.drawLine(QPointF(center_x + 12, y), QPointF(center_x + 5, y + 7))


class OperationInfoPanel(SectionCard):
    def __init__(self) -> None:
        super().__init__("操作信息")
        self.set_operation(None)

    def set_operation(self, operation: dict | None) -> None:
        self._clear_layout(self.body)
        if not operation:
            empty = QLabel("请选择一个增强操作以查看说明。")
            empty.setObjectName("emptyDetail")
            empty.setWordWrap(True)
            self.body.addWidget(empty)
            self.body.addStretch(1)
            return

        name = str(operation.get("name", ""))
        info_name = {"gaussian_blur": "blur", "median_blur": "blur", "affine": "shear"}.get(name, name)
        fallback_bbox = str(operation.get("bbox_effect", "不改变"))
        info = OPERATION_INFO.get(
            info_name,
            {
                "intro": f"{name} 增强操作。",
                "scenes": ["数据增强", "目标检测"],
                "bbox": fallback_bbox,
                "range": "按后端参数配置",
                "cost": "★★☆☆☆  较低",
                "kind": "default",
            },
        )
        bbox = fallback_bbox if fallback_bbox else str(info.get("bbox", "不改变"))
        tone = "orange" if bbox == "会改变" else "green"

        self._add_text_block("简介", str(info["intro"]))
        self._add_badge_block("适用场景", list(info["scenes"]), "green")
        self._add_badge_block("对 bbox 的影响", [bbox], tone)
        self._add_text_block("建议强度范围", str(info["range"]))
        self._add_text_block("计算开销", str(info["cost"]))

        label = QLabel("可视化预览")
        label.setObjectName("infoBlockTitle")
        self.body.addWidget(label)
        self.body.addWidget(OperationPreview(str(info.get("kind", "default"))))
        self.body.addStretch(1)

    def _add_text_block(self, title: str, detail: str) -> None:
        label = QLabel(title)
        label.setObjectName("infoBlockTitle")
        self.body.addWidget(label)
        detail_label = QLabel(detail)
        detail_label.setObjectName("infoBlockText")
        detail_label.setWordWrap(True)
        self.body.addWidget(detail_label)

    def _add_badge_block(self, title: str, values: list[str], tone: str) -> None:
        label = QLabel(title)
        label.setObjectName("infoBlockTitle")
        self.body.addWidget(label)
        self.body.addLayout(self._badges(values, tone))

    def _badges(self, values: list[str], tone: str) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for value in values:
            badge = QLabel(value)
            badge.setObjectName("infoBadge")
            badge.setProperty("tone", tone)
            layout.addWidget(badge)
        layout.addStretch(1)
        return layout

    def _clear_layout(self, layout: QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                self._clear_layout(item.layout())
