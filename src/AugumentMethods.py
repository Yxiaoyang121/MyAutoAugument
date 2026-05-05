import cv2
import numpy as np

# 机械位姿（亚像素平移/旋转）：解决“位置”的不确定性。
# 光源波动（亮度线性调节）：解决“环境”的不确定性。
# 反射率差异（对比度非线性调节）：解决“材质”的不确定性。
# 粉尘干扰（高斯噪声叠加）：解决“环境洁净度”的不确定性。


# 模拟机械位姿偏差 (亚像素平移 + 微小旋转)
def apply_mechanical_deviation(image, dx, dy, angle_deg):
    """
    模拟机械位姿偏差 (亚像素平移 + 微小旋转)
    :param image: 输入图像
    :param dx: x方向偏移量 (建议 -0.5 到 0.5 之间)
    :param dy: y方向偏移量 (建议 -0.5 到 0.5 之间)
    :param angle_deg: 旋转角度 (模拟吸盘误差，建议 -2.0 到 2.0 度之间)
    """
    rows, cols = image.shape[:2]

    # 1. 设定旋转中心 (通常为图像中心点)
    center = (cols / 2.0, rows / 2.0)

    # 2. 让 OpenCV 帮我们算出包含旋转的 2x3 仿射矩阵
    # 参数：旋转中心, 旋转角度, 缩放比例(1.0表示不缩放)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 3. 关键一步：在计算好的矩阵的“平移项”（第三列）上，叠加我们的亚像素微震动
    M[0, 2] += dx  # 累加 x 方向的亚像素位移
    M[1, 2] += dy  # 累加 y 方向的亚像素位移

    # 4. 一次性完成旋转与亚像素平移，坚持使用 INTER_CUBIC 保护边缘
    augmented_image = cv2.warpAffine(image, M, (cols, rows),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)  # 边缘复制，防止旋转出现黑边

    # INTER_NEAREST（最邻近）：取最近的一个点。速度最快，但位移不连续，会有锯齿。
    # INTER_LINEAR（双线性）：取周围 2 x 2（4个点）。速度均衡，但边缘会变得比较模糊（类似蒙了一层雾）。
    # INTER_CUBIC：取周围 4 x 4（16个点）。计算量大，但边缘保留最好，能产生最真实的亚像素灰度过渡。

    return augmented_image

# 光源波动模拟
def apply_light_fluctuation(image, alpha):
    """
    光源波动模拟
    :param image: 输入图像 (BGR 或 Gray)
    :param alpha: 亮度调节因子 (建议 0.95 到 1.05)
    """
    # 使用 cv2.convertScaleAbs 进行线性变换：Result = image * alpha + beta
    # 这里 beta=0 表示只改变对比度/亮度，不增加偏置
    # 它会自动处理数据截断（超过255设为255，小于0设为0）
    brightened_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    return brightened_image

# 反射率差异模拟 (对比度调节)
def apply_reflectivity_variation(image, alpha):
    """
    反射率差异模拟 (对比度调节)
    :param image: 输入图像
    :param alpha: 对比度因子 (建议 0.9 到 1.1)
    """
    # 计算图像均值，以均值为中心进行缩放，可以保持图像整体亮度基本不变，只改变对比度
    mean = np.mean(image)

    # 公式: Output = alpha * (Input - Mean) + Mean
    # 这比直接乘法更能模拟“材质对比度”的变化，而不会导致整体过亮或过暗
    table = np.array([((i - mean) * alpha + mean) for i in range(256)]).clip(0, 255).astype('uint8')
    adjusted_image = cv2.LUT(image, table)

    return adjusted_image

# 粉尘干扰模拟（叠加高斯噪声）
def apply_dust_interference(image, mean, sigma_range):
    """
    粉尘干扰模拟 (叠加高斯噪声)
    :param image: 输入图像 (BGR 或 Gray)
    :param mean: 噪声均值 (建议为 0，防止整体图像偏色)
    :param sigma_range: 噪声标准差 (建议 [0.01, 0.05]) 相对于归一化图像 [0,1]
    """
    # 支持灰度或彩色图像
    if image.ndim == 2:
        h, w = image.shape
        ch = 1
    else:
        h, w, ch = image.shape

    img_float = image.astype(np.float32) / 255.0
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    # 生成噪声（均值 mean，标准差 sigma）
    if ch == 1:
        noise = np.random.normal(mean, sigma, (h, w))
        noisy = img_float + noise
    else:
        noise = np.random.normal(mean, sigma, (h, w, ch))
        noisy = img_float + noise

    noisy_clipped = np.clip(noisy, 0.0, 1.0)
    noisy_uint8 = (noisy_clipped * 255).astype(np.uint8)
    return noisy_uint8
