from __future__ import annotations

import numpy as np

from AutoAugment.augmentations import apply_augmentation, get_augmentation_spec, list_augmentations
from AutoAugment.policies import OperationSpec, Policy, apply_policy


def make_inputs():
    image = np.full((20, 40, 3), 100, dtype=np.uint8)
    labels = np.asarray([1, 2], dtype=np.int64)
    bboxes = np.asarray([[4, 5, 14, 15], [20, 2, 35, 18]], dtype=np.float32)
    return image, labels, bboxes


def test_registry_lists_image_and_geometry_ops() -> None:
    names = set(list_augmentations())
    assert {"brightness", "horizontal_flip", "rotate", "translate"}.issubset(names)
    assert not get_augmentation_spec("brightness").changes_bboxes
    assert get_augmentation_spec("horizontal_flip").changes_bboxes


def test_brightness_does_not_change_bboxes() -> None:
    image, labels, bboxes = make_inputs()
    out_image, out_labels, out_bboxes = apply_augmentation(
        "brightness", image, labels, bboxes, strength=0.5, rng=np.random.default_rng(1)
    )
    assert out_image.shape == image.shape
    np.testing.assert_array_equal(out_labels, labels)
    np.testing.assert_allclose(out_bboxes, bboxes)


def test_contrast_preserves_uint8_dtype() -> None:
    image, labels, bboxes = make_inputs()
    out_image, _, _ = apply_augmentation(
        "contrast", image, labels, bboxes, strength=0.5, rng=np.random.default_rng(1)
    )
    assert out_image.dtype == np.uint8


def test_contrast_then_clahe_policy_does_not_fail() -> None:
    image, labels, bboxes = make_inputs()
    policy = Policy(
        "contrast_clahe",
        operations=[
            OperationSpec("contrast", prob=1.0, strength=0.4, params={"max_delta": 0.35}),
            OperationSpec("clahe", prob=1.0, strength=0.3, params={"max_clip_limit": 3.0}),
        ],
    )
    out_image, out_labels, out_bboxes = apply_policy(image, labels, bboxes, policy, rng=np.random.default_rng(2))
    assert out_image.dtype == np.uint8
    assert out_image.shape == image.shape
    np.testing.assert_array_equal(out_labels, labels)
    np.testing.assert_allclose(out_bboxes, bboxes)


def test_gaussian_blur_low_strength_does_not_fail() -> None:
    image, labels, bboxes = make_inputs()
    out_image, out_labels, out_bboxes = apply_augmentation(
        "gaussian_blur",
        image,
        labels,
        bboxes,
        strength=0.01,
        rng=np.random.default_rng(3),
    )
    assert out_image.shape == image.shape
    np.testing.assert_array_equal(out_labels, labels)
    np.testing.assert_allclose(out_bboxes, bboxes)


def test_empty_labels_do_not_fail() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    labels = np.zeros((0,), dtype=np.int64)
    bboxes = np.zeros((0, 4), dtype=np.float32)
    _, out_labels, out_bboxes = apply_augmentation("rotate", image, labels, bboxes, strength=0.5)
    assert out_labels.shape == (0,)
    assert out_bboxes.shape == (0, 4)


def test_horizontal_flip_updates_x_center() -> None:
    image, labels, bboxes = make_inputs()
    _, out_labels, out_bboxes = apply_augmentation("horizontal_flip", image, labels, bboxes)
    np.testing.assert_array_equal(out_labels, labels)
    np.testing.assert_allclose(out_bboxes[0], np.asarray([26, 5, 36, 15], dtype=np.float32))


def test_translate_clips_to_image_bounds() -> None:
    image, labels, bboxes = make_inputs()
    _, out_labels, out_bboxes = apply_augmentation(
        "translate",
        image,
        labels,
        bboxes,
        params={"dx": 30, "dy": 0},
        strength=1.0,
        rng=np.random.default_rng(2),
    )
    assert len(out_labels) == len(out_bboxes)
    assert np.all(out_bboxes[:, 0] >= 0)
    assert np.all(out_bboxes[:, 2] <= image.shape[1])
    assert np.all(out_bboxes[:, 2] > out_bboxes[:, 0])


def test_rotate_keeps_bboxes_inside_image() -> None:
    image, labels, bboxes = make_inputs()
    _, out_labels, out_bboxes = apply_augmentation(
        "rotate",
        image,
        labels,
        bboxes,
        params={"angle": 12},
        strength=0.8,
        rng=np.random.default_rng(3),
    )
    assert len(out_labels) == len(out_bboxes)
    assert np.all(out_bboxes[:, [0, 2]] >= 0)
    assert np.all(out_bboxes[:, [0, 2]] <= image.shape[1])
    assert np.all(out_bboxes[:, [1, 3]] >= 0)
    assert np.all(out_bboxes[:, [1, 3]] <= image.shape[0])
