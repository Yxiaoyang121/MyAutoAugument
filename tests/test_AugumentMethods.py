import cv2
import numpy as np
from src.AugumentMethods import apply_mechanical_deviation


def test_apply_mechanical_deviation_load_and_shape():
    # 尝试读取示例图像
    img_path = 'assets/Luntai1.jpg'
    img = cv2.imread(img_path)
    assert img is not None, f"测试图像未找到，请确保路径存在: {img_path}"

    # 应用变换
    out = apply_mechanical_deviation(img, 0.2, -0.1, 2.0)
    # 输出应与输入具有相同的尺寸
    assert out.shape == img.shape


if __name__ == '__main__':
    print('运行本地测试...')
    test_apply_mechanical_deviation_load_and_shape()
    print('测试通过')
