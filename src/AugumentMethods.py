from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from src.augmentations.geometric import mechanical_deviation
    from src.augmentations.noise import gaussian_noise
    from src.augmentations.photometric import brightness, contrast
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.augmentations.geometric import mechanical_deviation
    from src.augmentations.noise import gaussian_noise
    from src.augmentations.photometric import brightness, contrast


def apply_mechanical_deviation(image: np.ndarray, dx: float, dy: float, angle_deg: float) -> np.ndarray:
    """模拟机械位姿偏差，兼容旧版亚像素平移与微小旋转接口。"""
    return mechanical_deviation(image, {"dx": dx, "dy": dy, "angle_deg": angle_deg})


def apply_light_fluctuation(image: np.ndarray, alpha: float) -> np.ndarray:
    """模拟光源波动，兼容旧版亮度缩放接口。"""
    return brightness(image, {"alpha": alpha, "strength": 1.0})


def apply_reflectivity_variation(image: np.ndarray, alpha: float) -> np.ndarray:
    """模拟反射率差异，兼容旧版均值中心对比度接口。"""
    return contrast(image, {"alpha": alpha, "strength": 1.0})


def apply_dust_interference(image: np.ndarray, mean: float, sigma_range: list[float] | tuple[float, float]) -> np.ndarray:
    """模拟粉尘干扰，兼容旧版高斯噪声接口。"""
    return gaussian_noise(image, {"mean": mean, "sigma_range": sigma_range, "strength": 1.0})
