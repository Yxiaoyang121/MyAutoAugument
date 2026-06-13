from __future__ import annotations

import numpy as np

from AutoAugment.catf_v2.policy_matrix import initial_policy_matrix
from AutoAugment.catf_v2.sample_router import SampleAwareAugmentationRouter


def sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    labels = np.array([0], dtype=np.int64)
    bboxes = np.array([[16, 16, 40, 40]], dtype=np.float32)
    return image, labels, bboxes


def test_sample_router_skips_stable_only_image() -> None:
    policy = initial_policy_matrix({0: "stable"})
    policy["classes"]["0"]["status"] = "frozen"
    policy["classes"]["0"]["state"] = "frozen"
    policy["classes"]["0"]["ops"]["local_contrast"]["prob"] = 1.0
    policy["classes"]["0"]["ops"]["local_contrast"]["strength"] = 0.3
    router = SampleAwareAugmentationRouter(policy, seed=1, num_classes=1)
    image, labels, bboxes = sample()

    result = router.apply(image, labels, bboxes)

    assert result.audit["applied_ops"] == []
    assert router.stats.to_dict()["samples_augmented"] == 0


def test_sample_router_applies_active_class_roi_op() -> None:
    policy = initial_policy_matrix({0: "target"})
    policy["classes"]["0"]["status"] = "active"
    policy["classes"]["0"]["ops"]["sharpen_mild"]["prob"] = 1.0
    policy["classes"]["0"]["ops"]["sharpen_mild"]["strength"] = 0.3
    router = SampleAwareAugmentationRouter(policy, seed=2, num_classes=1, roi_aware=True)
    image, labels, bboxes = sample()

    result = router.apply(image, labels, bboxes)

    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    assert router.roi_stats.to_dict()["roi_aug_applied"] == 1


def test_sample_router_high_fp_conflict_halves_photometric_probability() -> None:
    policy = initial_policy_matrix({0: "target", 1: "guard"})
    policy["classes"]["0"]["status"] = "active"
    policy["classes"]["0"]["ops"]["gamma"]["prob"] = 1.0
    policy["classes"]["0"]["ops"]["gamma"]["strength"] = 0.3
    policy["classes"]["1"]["status"] = "active"
    policy["classes"]["1"]["guards"]["high_fp_guarded"] = True
    router = SampleAwareAugmentationRouter(policy, seed=3, num_classes=2, roi_aware=True)
    image = np.full((64, 64, 3), 100, dtype=np.uint8)
    labels = np.array([0, 1], dtype=np.int64)
    bboxes = np.array([[16, 16, 40, 40], [42, 42, 55, 55]], dtype=np.float32)

    result = router.apply(image, labels, bboxes)

    op = result.audit["operations"][0]
    assert op["name"] == "gamma"
    assert op["routed_prob"] == 0.5


def test_sample_router_respects_weak_image_aug_interval_cap() -> None:
    policy = initial_policy_matrix({0: "target"})
    row = policy["classes"]["0"]
    row["status"] = "active"
    row["ops"]["local_contrast"]["prob"] = 1.0
    row["ops"]["local_contrast"]["strength"] = 0.05
    row["weak_image_aug"] = {
        "enabled": True,
        "interval_start_epoch": 25,
        "max_aug_samples_per_interval": 1,
        "retained_op": "local_contrast",
    }
    router = SampleAwareAugmentationRouter(policy, seed=2, num_classes=1, roi_aware=True)
    router.set_epoch(25)
    image, labels, bboxes = sample()

    first = router.apply(image.copy(), labels.copy(), bboxes.copy())
    second = router.apply(image.copy(), labels.copy(), bboxes.copy())

    assert first.audit["applied_ops"]
    assert router.weak_image_aug_counts["0:25"] == 1
    assert second.audit["applied_ops"] == []
    assert second.audit["skipped_ops"][0]["skip_reason"] == "weak_image_aug_interval_cap"
