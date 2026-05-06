from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.AugumentMethods import apply_mechanical_deviation
from src.utils.demo_data import create_demo_image
from src.utils.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    """解析亚像素位移测试参数。"""
    parser = argparse.ArgumentParser(description="亚像素位移增强测试")
    parser.add_argument("--image", type=str, default=None, help="项目根目录下的输入图片相对路径")
    parser.add_argument("--output", type=str, default=None, help="项目根目录下的输出图片相对路径")
    parser.add_argument("--dx", type=float, default=0.5, help="水平方向位移")
    parser.add_argument("--dy", type=float, default=0.5, help="垂直方向位移")
    parser.add_argument("--angle", type=float, default=0.0, help="旋转角度")
    parser.add_argument("--seed", type=int, default=42, help="合成示例图随机种子")
    return parser.parse_args()


def main() -> None:
    """运行亚像素位移增强测试。"""
    args = parse_args()
    image = cv2.imread(str(resolve_project_path(args.image)), cv2.IMREAD_COLOR) if args.image else create_demo_image(seed=args.seed)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {args.image}")
    augmented = apply_mechanical_deviation(image, args.dx, args.dy, args.angle)
    difference = cv2.absdiff(image, augmented)
    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), augmented)
    print(f"增强完成，平均灰度差异: {float(np.mean(difference)):.4f}")


if __name__ == "__main__":
    main()
