#!/usr/bin/env python3
# 示例脚本：展示如何使用 src.AugumentMethods 中的方法
import cv2
from src.AugumentMethods import apply_mechanical_deviation


def main():
    img_path = 'assets/Luntai1.jpg'
    img = cv2.imread(img_path)
    if img is None:
        print(f'无法读取图像: {img_path}')
        return

    out = apply_mechanical_deviation(img, 0.2, -0.1, 2)

    cv2.imwrite('out_augmented.jpg', out)
    print('增强图像已保存为 out_augmented.jpg')


if __name__ == '__main__':
    main()
