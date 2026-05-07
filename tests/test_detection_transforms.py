from __future__ import annotations

import cv2
import numpy as np

from AutoAugment.bbox.affine import apply_affine_to_bboxes
from AutoAugment.bbox.convert import xyxy_to_yolo, yolo_to_xyxy
from AutoAugment.pipelines.compose import Compose
from AutoAugment.transforms.geometric import HorizontalFlip, RandomCrop, Rotate, Scale, Translate, VerticalFlip


def make_sample() -> dict:
    """构造基础目标检测样本。"""
    return {
        "image": np.zeros((10, 20, 3), dtype=np.uint8),
        "bboxes": np.asarray([[2, 3, 8, 9]], dtype=np.float32),
        "labels": np.asarray([5], dtype=np.int64),
    }


def test_horizontal_flip_updates_xyxy_bboxes() -> None:
    """验证水平翻转同步更新 bbox。"""
    sample = make_sample()
    result = HorizontalFlip()(sample)
    np.testing.assert_allclose(result["bboxes"], np.asarray([[12, 3, 18, 9]], dtype=np.float32))
    np.testing.assert_array_equal(result["labels"], sample["labels"])


def test_vertical_flip_updates_xyxy_bboxes() -> None:
    """验证垂直翻转同步更新 bbox。"""
    sample = make_sample()
    result = VerticalFlip()(sample)
    np.testing.assert_allclose(result["bboxes"], np.asarray([[2, 1, 8, 7]], dtype=np.float32))
    np.testing.assert_array_equal(result["labels"], sample["labels"])


def test_translate_clips_and_keeps_valid_bboxes() -> None:
    """验证平移后 bbox 被裁剪且仍合法。"""
    sample = make_sample()
    result = Translate(dx=5, dy=-2)(sample)
    np.testing.assert_allclose(result["bboxes"], np.asarray([[7, 1, 13, 7]], dtype=np.float32))
    assert np.all(result["bboxes"][:, 2] > result["bboxes"][:, 0])
    assert np.all(result["bboxes"][:, 3] > result["bboxes"][:, 1])


def test_scale_one_keeps_bboxes() -> None:
    """验证缩放倍率为 1 时 bbox 不变。"""
    sample = make_sample()
    result = Scale(scale=1.0)(sample)
    np.testing.assert_allclose(result["bboxes"], sample["bboxes"])


def test_rotate_uses_affine_corner_transform() -> None:
    """验证旋转通过四角点仿射变换重算外接矩形。"""
    sample = {
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
        "bboxes": np.asarray([[2, 2, 4, 4]], dtype=np.float32),
        "labels": np.asarray([1], dtype=np.int64),
    }
    result = Rotate(angle=90)(sample)
    matrix = cv2.getRotationMatrix2D((5.0, 5.0), 90, 1.0).astype(np.float32)
    expected = apply_affine_to_bboxes(sample["bboxes"], matrix)
    np.testing.assert_allclose(result["bboxes"], expected, atol=1e-5)


def test_random_crop_shifts_and_clips_bboxes() -> None:
    """验证固定裁剪窗口会平移并裁剪 bbox。"""
    sample = {
        "image": np.zeros((10, 20, 3), dtype=np.uint8),
        "bboxes": np.asarray([[4, 4, 8, 8]], dtype=np.float32),
        "labels": np.asarray([3], dtype=np.int64),
    }
    result = RandomCrop(crop_box=(2, 2, 12, 10))(sample)
    assert result["image"].shape == (8, 10, 3)
    np.testing.assert_allclose(result["bboxes"], np.asarray([[2, 2, 6, 6]], dtype=np.float32))


def test_compose_uses_detection_sample_structure() -> None:
    """验证 Compose 输入输出均为目标检测 sample。"""
    sample = make_sample()
    pipeline = Compose([HorizontalFlip(), Translate(dx=1, dy=1)], seed=123)
    result = pipeline(sample)
    assert set(["image", "bboxes", "labels"]).issubset(result)
    assert result["bboxes"].shape[1] == 4
    assert len(result["labels"]) == len(result["bboxes"])


def test_yolo_conversion_round_trip() -> None:
    """验证 YOLO 与 xyxy 格式可互相转换。"""
    bboxes = np.asarray([[2, 3, 8, 9]], dtype=np.float32)
    yolo_boxes = xyxy_to_yolo(bboxes, width=20, height=10)
    restored = yolo_to_xyxy(yolo_boxes, width=20, height=10)
    np.testing.assert_allclose(restored, bboxes, atol=1e-5)
