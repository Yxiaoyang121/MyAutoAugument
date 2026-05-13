from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AutoAugment.formats.yolo import load_yolo_labels, save_yolo_labels
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
    """解析单图增强调试参数。"""
    parser = argparse.ArgumentParser(description="单张目标检测图像增强调试")
    parser.add_argument("--image", type=str, default=None, help="输入图像相对项目根目录路径")
    parser.add_argument("--label", type=str, default=None, help="YOLO 标签 txt 相对项目根目录路径")
    parser.add_argument("--output-image", type=str, default="outputs/demo/augmented.jpg", help="增强图保存路径")
    parser.add_argument("--output-label", type=str, default="outputs/demo/augmented.txt", help="增强标签保存路径")
    parser.add_argument("--output-vis", type=str, default="outputs/demo/augmented_vis.jpg", help="bbox 可视化保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--show", action="store_true", help="是否弹窗展示可视化结果")
    return parser.parse_args()


def build_pipeline(seed: int) -> Compose:
    """构建用于单图调试的 Compose 增强流水线。"""
    return Compose(
        [
            HorizontalFlip(probability=0.5),
            VerticalFlip(probability=0.1),
            Rotate(angle_range=(-8.0, 8.0), probability=0.7),
            Translate(dx_range=(-0.06, 0.06), dy_range=(-0.06, 0.06), probability=0.7),
            Scale(scale_range=(0.9, 1.1), probability=0.5),
            RandomCrop(width_range=(180, 256), height_range=(180, 256), min_visibility=0.25, probability=0.3),
        ],
        seed=seed,
    )


def make_demo_sample() -> dict:
    """创建无需外部文件的目标检测调试样本。"""
    image = np.full((256, 256, 3), 38, dtype=np.uint8)
    cv2.rectangle(image, (70, 80), (170, 165), (180, 180, 170), -1)
    cv2.circle(image, (185, 85), 20, (80, 140, 220), -1)
    bboxes = np.asarray([[70, 80, 170, 165], [165, 65, 205, 105]], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    return {"image": image, "bboxes": bboxes, "labels": labels}


def load_sample(image_path: str | None, label_path: str | None) -> dict:
    """读取单图和 YOLO 标签，未传路径时返回内置样本。"""
    if image_path is None:
        return make_demo_sample()
    resolved_image = resolve_project_path(image_path)
    image = cv2.imread(str(resolved_image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {resolved_image}")
    height, width = image.shape[:2]
    if label_path is None:
        labels = np.zeros((0,), dtype=np.int64)
        bboxes = np.zeros((0, 4), dtype=np.float32)
    else:
        labels, bboxes = load_yolo_labels(resolve_project_path(label_path), width, height)
    return {"image": image, "bboxes": bboxes, "labels": labels}


def save_outputs(sample: dict, output_image: str, output_label: str, output_vis: str) -> None:
    """保存增强图像、YOLO 标签和 bbox 可视化图像。"""
    image_path = resolve_project_path(output_image)
    label_path = resolve_project_path(output_label)
    vis_path = resolve_project_path(output_vis)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), sample["image"])
    height, width = sample["image"].shape[:2]
    save_yolo_labels(label_path, sample["labels"], sample["bboxes"], width, height)
    cv2.imwrite(str(vis_path), draw_bboxes(sample))


def main() -> None:
    """运行单张图像目标检测增强调试。"""
    args = parse_args()
    sample = load_sample(args.image, args.label)
    augmented = build_pipeline(args.seed)(sample)
    save_outputs(augmented, args.output_image, args.output_label, args.output_vis)
    print(f"增强图像已保存: {resolve_project_path(args.output_image)}")
    print(f"增强标签已保存: {resolve_project_path(args.output_label)}")
    print(f"可视化结果已保存: {resolve_project_path(args.output_vis)}")
    if args.show:
        cv2.imshow("augmented", draw_bboxes(augmented))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
