import cv2
import matplotlib.pyplot as plt
import numpy as np

from AugumentMethods import apply_mechanical_deviation

# 1. 加载图像并转换颜色空间
# OpenCV 默认 BGR，Matplotlib 绘图需要 RGB
img_bgr = cv2.imread('../ImageSourceTest/boli.bmp')
if img_bgr is None:
    print("未找到图片！！！！！")
else:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. 应用你的亚像素位移方法
    # 模拟一个微小的位移，比如 0.4 像素
    # 这样计算 absdiff 时就不会有颜色空间冲突
    augmented_bgr = apply_mechanical_deviation(img_bgr, 0.5, 0.5,0)

    # 2. 为了 Matplotlib 显示，转换两个 RGB 副本
    img_show_original = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_show_augmented = cv2.cvtColor(augmented_bgr, cv2.COLOR_BGR2RGB)

    # 3. 使用 Matplotlib 创建对比窗口
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 左侧：原始图
    axes[0].imshow(img_show_original)
    axes[0].set_title("Original (Integer Grid)")
    axes[0].axis('on') # 显示坐标轴，方便看像素位置

    # 右侧：增强图
    axes[1].imshow(img_show_augmented)
    axes[1].set_title("Sub-pixel Shifted (0.4 px)")
    axes[1].axis('on')

    # 启用交互功能
    plt.tight_layout()
    print("提示：点击窗口工具栏的『放大镜』图标，然后在图中拉框，可以查看边缘的插值灰度过渡。")
    plt.show()

    diff = cv2.absdiff(img_bgr, augmented_bgr)

    mean_diff = np.mean(diff)
    print(f"位移像素时，平均灰度改变了: {mean_diff:.4f}")

    # 为了让肉眼看清，把差异放大 10 倍
    cv2.namedWindow("Difference (x10 Enhanced)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Difference (x10 Enhanced)", 800, 300)
    diff_enhanced = cv2.multiply(diff, 10)
    # cv2.imshow("Difference (x10 Enhanced)", diff_enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存增强后的图像
    # 第一个参数是文件名，第二个参数是 BGR 格式的图像矩阵
    # cv2.imwrite('original_save.bmp', img_bgr)
    cv2.imwrite('augmented_save.bmp', augmented_bgr)

    # 如果想保存差异图（查看增强效果）
    diff = cv2.absdiff(img_bgr, augmented_bgr)
    # cv2.imwrite('difference_save.bmp', cv2.multiply(diff, 10))  # 放大10倍保存

    print("图像已通过 OpenCV 成功保存。")
