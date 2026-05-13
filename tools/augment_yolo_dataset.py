from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.datasets import YoloDetectionDataset
from AutoAugment.formats.yolo import save_yolo_labels
from AutoAugment.pipelines import Compose
from AutoAugment.transforms.geometric import HorizontalFlip, RandomCrop, Rotate, Scale, Translate, VerticalFlip
from AutoAugment.visualize import draw_bboxes


def resolve_project_path(path: str | Path) -> Path:
    """将相对路径解析为项目根目录下的路径。"""
    path = Path(path)
    if path.is_absolute():
        raise ValueError("路径必须是相对项目根目录的相对路径")
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """解析 YOLO 批量增强参数。"""
    parser = argparse.ArgumentParser(description="批量增强 YOLO 目标检测数据集")
    parser.add_argument("--images-dir", type=str, default=None, help="YOLO 图像目录相对项目根目录路径")
    parser.add_argument("--labels-dir", type=str, default=None, help="YOLO 标签目录相对项目根目录路径")
    parser.add_argument("--output-dir", type=str, default="outputs/yolo_augmented", help="增强结果输出目录")
    parser.add_argument("--repeat", type=int, default=1, help="每张图像增强次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--visualize", action="store_true", help="是否额外保存 bbox 可视化图")
    return parser.parse_args()


def build_pipeline(seed: int) -> Compose:
    """构建 YOLO 批量增强流水线。"""
    return Compose(
        [
            HorizontalFlip(probability=0.5),
            VerticalFlip(probability=0.05),
            Rotate(angle_range=(-6.0, 6.0), probability=0.5),
            Translate(dx_range=(-0.05, 0.05), dy_range=(-0.05, 0.05), probability=0.5),
            Scale(scale_range=(0.9, 1.12), probability=0.4),
            RandomCrop(min_visibility=0.25, probability=0.25),
        ],
        seed=seed,
    )


def create_demo_yolo_dataset(root: Path) -> tuple[str, str]:
    """创建可直接运行的最小 YOLO 检测数据集。"""
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        image = np.full((256, 256, 3), 35 + index * 20, dtype=np.uint8)
        x1 = 50 + index * 20
        y1 = 70
        x2 = 155 + index * 15
        y2 = 165
        cv2.rectangle(image, (x1, y1), (x2, y2), (170, 180, 190), -1)
        cv2.circle(image, (185, 95 + index * 15), 18, (80, 140, 220), -1)
        image_path = images_dir / f"demo_{index:03d}.jpg"
        label_path = labels_dir / f"demo_{index:03d}.txt"
        cv2.imwrite(str(image_path), image)
        bboxes = np.asarray([[x1, y1, x2, y2], [167, 77 + index * 15, 203, 113 + index * 15]], dtype=np.float32)
        labels = np.asarray([0, 1], dtype=np.int64)
        save_yolo_labels(label_path, labels, bboxes, image_width=256, image_height=256)
    return "outputs/demo_yolo_source/images", "outputs/demo_yolo_source/labels"


def prepare_dataset_args(args: argparse.Namespace) -> tuple[str, str]:
    """准备真实或内置 YOLO 数据集路径。"""
    if args.images_dir is not None:
        labels_dir = args.labels_dir if args.labels_dir is not None else args.images_dir
        return args.images_dir, labels_dir
    return create_demo_yolo_dataset(resolve_project_path("outputs/demo_yolo_source"))


def save_augmented_sample(sample: dict, output_dir: Path, stem: str, visualize: bool) -> None:
    """保存增强后的图像、YOLO 标签和可选可视化结果。"""
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    vis_dir = output_dir / "visualize"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / f"{stem}.jpg"
    label_path = labels_dir / f"{stem}.txt"
    cv2.imwrite(str(image_path), sample["image"])
    height, width = sample["image"].shape[:2]
    save_yolo_labels(label_path, sample["labels"], sample["bboxes"], width, height)
    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(vis_dir / f"{stem}.jpg"), draw_bboxes(sample))


def main() -> None:
    """运行 YOLO 目标检测数据集批量增强。"""
    args = parse_args()
    images_dir, labels_dir = prepare_dataset_args(args)
    pipeline = build_pipeline(args.seed)
    dataset = YoloDetectionDataset(images_dir=images_dir, labels_dir=labels_dir, transform=None)
    output_dir = resolve_project_path(args.output_dir)
    total = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        image_stem = Path(sample["image_path"]).stem
        for repeat_index in range(max(1, args.repeat)):
            augmented = pipeline(sample)
            output_stem = f"{image_stem}_aug_{repeat_index:03d}"
            save_augmented_sample(augmented, output_dir, output_stem, args.visualize)
            total += 1
    print(f"已增强样本数: {total}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
