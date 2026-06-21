from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2

from AutoAugment.utils import IMAGE_EXTENSIONS

ProgressCallback = Callable[[int, str], None]


@dataclass
class DatasetInspection:
    root: str
    format: str = "YOLO"
    exists: bool = False
    has_images_train: bool = False
    has_images_val: bool = False
    has_labels_train: bool = False
    has_labels_val: bool = False
    has_data_yaml: bool = False
    train_image_count: int = 0
    val_image_count: int = 0
    train_label_count: int = 0
    val_label_count: int = 0
    class_count: int = 0
    class_names: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    bad_images: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAdapter:
    def inspect_yolo_dataset(self, root: str | Path, progress_callback: ProgressCallback | None = None) -> DatasetInspection:
        dataset = Path(root)
        info = DatasetInspection(root=str(dataset))
        self._emit_progress(progress_callback, 8, "检查数据集根目录...")
        info.exists = dataset.exists()
        if not dataset.exists():
            info.errors.append(f"数据集根目录不存在: {dataset}")
            return info

        self._emit_progress(progress_callback, 14, "检查 YOLO 目录结构...")
        images_train = dataset / "images" / "train"
        images_val = dataset / "images" / "val"
        labels_train = dataset / "labels" / "train"
        labels_val = dataset / "labels" / "val"
        data_yaml = dataset / "data.yaml"

        info.has_images_train = images_train.exists()
        info.has_images_val = images_val.exists()
        info.has_labels_train = labels_train.exists()
        info.has_labels_val = labels_val.exists()
        info.has_data_yaml = data_yaml.exists()

        self._emit_progress(progress_callback, 22, "扫描图像和标签文件...")
        train_images = self._image_files(images_train)
        val_images = self._image_files(images_val)
        info.train_image_count = len(train_images)
        info.val_image_count = len(val_images)
        info.train_label_count = len(list(labels_train.rglob("*.txt"))) if labels_train.exists() else 0
        info.val_label_count = len(list(labels_val.rglob("*.txt"))) if labels_val.exists() else 0

        if not info.has_images_train:
            info.errors.append("缺少 images/train")
        if not info.has_images_val:
            info.errors.append("缺少 images/val")
        if not info.has_labels_train:
            info.errors.append("缺少 labels/train")
        if not info.has_labels_val:
            info.errors.append("缺少 labels/val")
        if not info.has_data_yaml:
            info.warnings.append("缺少 data.yaml")

        for split, image_root, label_root, images in [
            ("train", images_train, labels_train, train_images),
            ("val", images_val, labels_val, val_images),
        ]:
            start = 28 if split == "train" else 45
            end = 45 if split == "train" else 58
            self._inspect_split(split, image_root, label_root, images, info, progress_callback, start, end)

        self._emit_progress(progress_callback, 62, "读取 data.yaml 和类别信息...")
        names = self._read_class_names(data_yaml) if data_yaml.exists() else []
        self._emit_progress(progress_callback, 66, "检测标签类别 ID...")
        detected_count = self._detect_class_count([labels_train, labels_val])
        if names:
            info.class_names = names
            info.class_count = max(len(names), detected_count)
        else:
            info.class_count = detected_count
            info.class_names = [f"class{index}" for index in range(detected_count)]
        self._emit_progress(progress_callback, 70, "基础检查完成")
        return info

    def _inspect_split(
        self,
        split: str,
        image_root: Path,
        label_root: Path,
        images: list[Path],
        info: DatasetInspection,
        progress_callback: ProgressCallback | None = None,
        progress_start: int = 0,
        progress_end: int = 0,
    ) -> None:
        total = len(images)
        for index, image_path in enumerate(images, start=1):
            if not self._is_readable_image(image_path):
                info.bad_images.append(str(image_path))
            try:
                relative = image_path.relative_to(image_root)
            except ValueError:
                relative = Path(image_path.name)
            label_path = label_root / relative.with_suffix(".txt")
            if not label_path.exists():
                info.missing_labels.append(str(label_path))
                continue
            text = label_path.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                info.empty_labels.append(str(label_path))
            if progress_callback and total and (index == total or index % 20 == 0):
                progress = progress_start + int((progress_end - progress_start) * index / total)
                self._emit_progress(progress_callback, progress, f"检查 {split} 样本 {index}/{total}...")

    def _image_files(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        extensions = {ext.lower() for ext in IMAGE_EXTENSIONS}
        return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)

    def _read_class_names(self, data_yaml: Path) -> list[str]:
        lines = data_yaml.read_text(encoding="utf-8", errors="replace").splitlines()
        names: list[str] = []
        in_names = False
        for raw in lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("names:"):
                inline = stripped[len("names:") :].strip()
                if inline.startswith("[") and inline.endswith("]"):
                    return [item.strip().strip("'\"") for item in inline[1:-1].split(",") if item.strip()]
                in_names = True
                continue
            if in_names:
                if not raw.startswith((" ", "\t")):
                    break
                if ":" in stripped:
                    _, value = stripped.split(":", 1)
                    names.append(value.strip().strip("'\""))
                else:
                    names.append(stripped.strip("- ").strip("'\""))
        return names

    def _detect_class_count(self, label_roots: list[Path]) -> int:
        max_class_id = -1
        for root in label_roots:
            if not root.exists():
                continue
            for label_path in root.rglob("*.txt"):
                for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        max_class_id = max(max_class_id, int(float(parts[0])))
                    except ValueError:
                        continue
        return max_class_id + 1 if max_class_id >= 0 else 0

    def _is_readable_image(self, image_path: Path) -> bool:
        try:
            return image_path.stat().st_size > 0 and cv2.haveImageReader(str(image_path))
        except OSError:
            return False

    def _emit_progress(self, progress_callback: ProgressCallback | None, value: int, text: str) -> None:
        if progress_callback:
            progress_callback(value, text)
