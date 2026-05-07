from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np


def load_voc_xml(xml_path: str | Path, class_to_id: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """读取 VOC XML 标注并返回 labels 与 xyxy bbox。"""
    root = ElementTree.parse(Path(xml_path)).getroot()
    labels: list[int] = []
    boxes: list[list[float]] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bbox_node = obj.find("bndbox")
        if name_node is None or bbox_node is None:
            continue
        class_name = name_node.text or ""
        if class_name not in class_to_id:
            continue
        labels.append(class_to_id[class_name])
        boxes.append(
            [
                float(bbox_node.findtext("xmin", "0")),
                float(bbox_node.findtext("ymin", "0")),
                float(bbox_node.findtext("xmax", "0")),
                float(bbox_node.findtext("ymax", "0")),
            ]
        )
    return np.asarray(labels, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
