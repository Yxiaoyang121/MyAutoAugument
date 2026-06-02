from __future__ import annotations

import hashlib

import numpy as np

from AutoAugment.catf_v2 import ROIStats, SampleAwareAugmentationRouter, initial_policy_matrix
from AutoAugment.online_augmentation import OnlineAugmentationStats
from scripts.train_yolo_online_aug import (
    OnlineTrainingContext,
    UltralyticsOnlinePolicyTransform,
    instances_to_xyxy,
    make_instances,
)
from ultralytics.utils.instance import Instances


def _hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _labels(class_id: int = 6, bbox: np.ndarray | None = None, image: np.ndarray | None = None) -> dict:
    img = image if image is not None else np.full((1024, 1024, 3), 100, dtype=np.uint8)
    box = bbox if bbox is not None else np.array([[100.0, 100.0, 220.0, 220.0]], dtype=np.float32)
    return {
        "img": img,
        "cls": np.array([[class_id]], dtype=np.float32),
        "instances": make_instances(Instances, box, width=img.shape[1], height=img.shape[0]),
    }


def _transform(policy: dict, *, seed: int = 1, catf_noop: bool = False) -> tuple[UltralyticsOnlinePolicyTransform, SampleAwareAugmentationRouter]:
    stats = OnlineAugmentationStats()
    router = SampleAwareAugmentationRouter(
        policy,
        seed=seed,
        num_classes=len(policy.get("classes", {})),
        stats=stats,
        roi_stats=ROIStats(),
        roi_aware=True,
        sample_aware=True,
        total_epochs=50,
    )
    context = OnlineTrainingContext(
        augmentor=router,
        preview_dir=None,
        save_preview=False,
        preview_count=0,
        total_epochs=50,
        catf_noop=catf_noop,
    )
    return UltralyticsOnlinePolicyTransform(context, Instances), router


def _box_hash(labels: dict) -> str:
    image = labels["img"]
    return _hash(instances_to_xyxy(labels["instances"], width=image.shape[1], height=image.shape[0]))


def test_no_active_class_bypasses_without_rebuild() -> None:
    policy = initial_policy_matrix({6: "漏背锡"})
    transform, router = _transform(policy)
    labels = _labels(6)
    instance_id = id(labels["instances"])

    out = transform(labels)

    assert out is labels
    assert id(out["instances"]) == instance_id
    assert router.random_draw_count == 0
    assert router.stats.to_dict()["samples_augmented"] == 0


def test_no_aug_ok_class_bypasses_without_instances_rebuild() -> None:
    policy = initial_policy_matrix({1: "OK3"})
    policy["classes"]["1"]["ops"]["local_contrast"]["prob"] = 1.0
    policy["classes"]["1"]["ops"]["local_contrast"]["strength"] = 0.4
    transform, router = _transform(policy)
    labels = _labels(1)
    instance_id = id(labels["instances"])

    out = transform(labels)

    assert out is labels
    assert id(out["instances"]) == instance_id
    assert router.random_draw_count == 0


def test_high_fp_guarded_class_does_not_clip_bbox() -> None:
    policy = initial_policy_matrix({4: "油污"})
    row = policy["classes"]["4"]
    row["status"] = "active"
    row["guards"]["high_fp_guarded"] = True
    row["ops"]["gamma"]["prob"] = 1.0
    row["ops"]["gamma"]["strength"] = 0.3
    transform, router = _transform(policy)
    bbox = np.array([[302.000122, 920.0, 364.998657, 1024.000488]], dtype=np.float32)
    labels = _labels(4, bbox=bbox)
    before_hash = _box_hash(labels)

    out = transform(labels)

    assert out is labels
    assert _box_hash(out) == before_hash
    np.testing.assert_allclose(instances_to_xyxy(out["instances"], width=1024, height=1024), bbox)
    assert router.random_draw_count == 0


def test_all_op_prob_zero_preserves_image_bbox_and_cls_hash() -> None:
    policy = initial_policy_matrix({6: "漏背锡"})
    policy["classes"]["6"]["status"] = "active"
    for op in policy["classes"]["6"]["ops"].values():
        op["prob"] = 0.0
        op["strength"] = 0.0
    transform, router = _transform(policy)
    labels = _labels(6)
    before = (_hash(labels["img"]), _hash(labels["cls"]), _box_hash(labels))

    out = transform(labels)

    assert out is labels
    assert (_hash(out["img"]), _hash(out["cls"]), _box_hash(out)) == before
    assert router.random_draw_count == 0


def test_force_skip_path_does_not_clip_border_bbox() -> None:
    policy = initial_policy_matrix({4: "油污"})
    policy["classes"]["4"]["status"] = "active"
    bbox = np.array([[302.000122, 920.0, 364.998657, 1024.000488]], dtype=np.float32)
    labels = _labels(4, bbox=bbox)
    transform, router = _transform(policy)

    out = transform(labels)

    assert out is labels
    np.testing.assert_allclose(instances_to_xyxy(out["instances"], width=1024, height=1024), bbox)
    assert router.random_draw_count == 0


def test_roi_skipped_small_roi_preserves_bbox_and_avoids_random_draw() -> None:
    policy = initial_policy_matrix({6: "漏背锡"})
    policy["classes"]["6"]["status"] = "active"
    policy["classes"]["6"]["ops"]["sharpen_mild"]["prob"] = 1.0
    policy["classes"]["6"]["ops"]["sharpen_mild"]["strength"] = 0.4
    transform, router = _transform(policy)
    bbox = np.array([[10.0, 10.0, 12.0, 12.0]], dtype=np.float32)
    labels = _labels(6, bbox=bbox, image=np.full((32, 32, 3), 120, dtype=np.uint8))
    before_hash = _box_hash(labels)

    out = transform(labels)

    assert out is labels
    assert _box_hash(out) == before_hash
    assert router.random_draw_count == 0
    assert router.roi_stats.roi_aug_skipped_small_roi == 1


def test_catf_noop_true_returns_original_label_object() -> None:
    policy = initial_policy_matrix({6: "漏背锡"})
    policy["classes"]["6"]["status"] = "active"
    policy["classes"]["6"]["ops"]["local_contrast"]["prob"] = 1.0
    policy["classes"]["6"]["ops"]["local_contrast"]["strength"] = 0.4
    transform, router = _transform(policy, catf_noop=True)
    labels = _labels(6)

    out = transform(labels)

    assert out is labels
    assert router.random_draw_count == 0


def test_applied_aug_allows_label_rebuild() -> None:
    policy = initial_policy_matrix({6: "漏背锡"})
    policy["classes"]["6"]["status"] = "active"
    policy["classes"]["6"]["ops"]["local_contrast"]["prob"] = 1.0
    policy["classes"]["6"]["ops"]["local_contrast"]["strength"] = 0.4
    transform, router = _transform(policy)
    image = np.tile(np.arange(80, dtype=np.uint8).reshape(80, 1), (1, 80))
    image = np.stack([image, image, image], axis=2)
    labels = _labels(6, bbox=np.array([[20.0, 20.0, 55.0, 55.0]], dtype=np.float32), image=image)
    instance_id = id(labels["instances"])

    out = transform(labels)

    assert out is labels
    assert id(out["instances"]) != instance_id
    assert router.random_draw_count == 1
    assert router.roi_stats.roi_aug_applied == 1
