from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.AugumentMethods import apply_reflectivity_variation
from src.utils.demo_data import create_demo_image
from src.utils.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    """解析反射率差异测试参数。"""
    parser = argparse.ArgumentParser(description="反射率差异增强测试")
    parser.add_argument("--image", type=str, default=None, help="项目根目录下的输入图片相对路径")
    parser.add_argument("--output", type=str, default=None, help="项目根目录下的输出图片相对路径")
    parser.add_argument("--alpha", type=float, default=0.8, help="对比度缩放因子")
    parser.add_argument("--seed", type=int, default=42, help="合成示例图随机种子")
    return parser.parse_args()


def main() -> None:
    """运行反射率差异增强测试。"""
    args = parse_args()
    image = cv2.imread(str(resolve_project_path(args.image)), cv2.IMREAD_COLOR) if args.image else create_demo_image(seed=args.seed)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {args.image}")
    augmented = apply_reflectivity_variation(image, args.alpha)
    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), augmented)
    print(f"增强完成，输出尺寸: {augmented.shape}")


if __name__ == "__main__":
    main()
