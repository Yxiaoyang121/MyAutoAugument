from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.sample_router import SampleAwareAugmentationRouter
from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix


def test_roi_augmentation_keeps_bbox_and_class() -> None:
    policy = initial_policy_matrix({0: "target"})
    policy["classes"]["0"]["status"] = "active"
    policy["classes"]["0"]["ops"]["local_contrast"]["prob"] = 1.0
    policy["classes"]["0"]["ops"]["local_contrast"]["strength"] = 0.4
    router = SampleAwareAugmentationRouter(policy, seed=5, num_classes=1, roi_aware=True)
    image = np.full((80, 80, 3), 120, dtype=np.uint8)
    labels = np.array([0], dtype=np.int64)
    bboxes = np.array([[20, 20, 55, 55]], dtype=np.float32)

    result = router.apply(image, labels, bboxes)

    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    assert router.roi_stats.roi_aug_applied == 1
    assert router.stats.to_dict()["class_id_oob_count"] == 0


def test_roi_too_small_is_skipped() -> None:
    policy = initial_policy_matrix({0: "target"})
    policy["classes"]["0"]["status"] = "active"
    policy["classes"]["0"]["ops"]["sharpen_mild"]["prob"] = 1.0
    policy["classes"]["0"]["ops"]["sharpen_mild"]["strength"] = 0.4
    router = SampleAwareAugmentationRouter(policy, seed=6, num_classes=1, roi_aware=True)
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    labels = np.array([0], dtype=np.int64)
    bboxes = np.array([[10, 10, 13, 13]], dtype=np.float32)

    result = router.apply(image, labels, bboxes)

    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    assert router.roi_stats.roi_aug_skipped_small_roi == 1
    assert result.audit["applied_ops"] == []
