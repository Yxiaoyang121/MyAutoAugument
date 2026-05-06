# main.py
from importlib.resources import Resource

import cv2
import numpy as np


# 从 augment_utils.py 中导入特定的方法
from AugumentMethods import apply_mechanical_deviation


def main():
    # 1. 读取图像
    Resource_img = cv2.imread('ImageSourceTest/Luntai1.jpg')  # 替换为你的工业工件图像路径
    if Resource_img is None:
        print("错误：无法读取图像，请检查路径。")
        return

    # 2. 设置亚像素位移参数
    # 模拟 0.2 像素的水平位移和 -0.1 像素的垂直位移
    dx, dy = 0.2, -0.1

    # 3. 调用增强方法
    augmented_img = apply_mechanical_deviation(Resource_img, dx, dy,2)

    # 4. 显示或保存结果

    win_name = "Industrial Vision - Subpixel Shift"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 600)
    cv2.imshow(win_name, augmented_img)
    print(f"已成功应用位移: dx={dx}, dy={dy}")

    Resour_name = "原图"
    cv2.namedWindow(Resour_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(Resour_name, 800, 600)
    cv2.imshow(Resour_name, Resource_img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()