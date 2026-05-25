from __future__ import annotations

import numpy as np

from AutoAugment.online_augmentation import OnlineAugmentationStats, OnlinePolicyAugmentor


def make_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((64, 96, 3), 80, dtype=np.uint8)
    image[25, 35] = [10, 200, 30]
    labels = np.asarray([1, 2], dtype=np.int64)
    bboxes = np.asarray([[20, 18, 50, 42], [60, 10, 82, 30]], dtype=np.float32)
    return image, labels, bboxes


def test_photometric_op_does_not_change_bbox() -> None:
    image, labels, bboxes = make_sample()
    policy = {"operations": [{"name": "brightness", "prob": 1.0, "strength": 0.5, "params": {"max_delta": 0.2}}]}
    augmentor = OnlinePolicyAugmentor(policy, seed=1, num_classes=3)

    result = augmentor.apply(image, labels, bboxes)

    assert result.image.shape == image.shape
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    assert augmentor.stats.to_dict()["ops"]["brightness"]["applied"] == 1


def test_hsv_jitter_does_not_change_bbox() -> None:
    image, labels, bboxes = make_sample()
    policy = {
        "operations": [
            {
                "name": "hsv_jitter",
                "prob": 1.0,
                "strength": 1.0,
                "params": {"hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4},
            }
        ]
    }
    augmentor = OnlinePolicyAugmentor(policy, seed=11, num_classes=3)

    result = augmentor.apply(image, labels, bboxes)

    assert result.image.shape == image.shape
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    assert augmentor.stats.to_dict()["ops"]["hsv_jitter"]["applied"] == 1


def test_horizontal_flip_updates_bbox() -> None:
    image, labels, bboxes = make_sample()
    policy = {"operations": [{"name": "horizontal_flip", "prob": 1.0, "strength": 1.0, "params": {}}]}
    augmentor = OnlinePolicyAugmentor(policy, seed=2, num_classes=3)

    result = augmentor.apply(image, labels, bboxes)

    expected = np.asarray([[46, 18, 76, 42], [14, 10, 36, 30]], dtype=np.float32)
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, expected)


def test_random_scale_translate_keeps_bboxes_legal() -> None:
    image, labels, bboxes = make_sample()
    policy = {
        "operations": [
            {
                "name": "random_scale_translate",
                "prob": 1.0,
                "strength": 1.0,
                "params": {"translate": 0.1, "scale": 0.5, "scale_value": 1.0, "tx": 0.05, "ty": -0.04},
            }
        ]
    }
    augmentor = OnlinePolicyAugmentor(policy, seed=12, num_classes=3)

    result = augmentor.apply(image, labels, bboxes)

    assert result.image.shape == image.shape
    assert np.all(result.bboxes[:, [0, 2]] >= 0)
    assert np.all(result.bboxes[:, [0, 2]] <= image.shape[1])
    assert np.all(result.bboxes[:, [1, 3]] >= 0)
    assert np.all(result.bboxes[:, [1, 3]] <= image.shape[0])
    assert np.all(result.labels >= 0)
    assert np.all(result.labels < 3)


def test_mosaic4_outputs_target_size_and_legal_bboxes() -> None:
    image, labels, bboxes = make_sample()
    sources = [make_sample(), make_sample(), make_sample()]

    def sample_provider(_rng: np.random.Generator):
        return sources.pop(0) if sources else make_sample()

    policy = {
        "operations": [
            {
                "name": "mosaic4",
                "prob": 1.0,
                "strength": 1.0,
                "params": {"target_size": 128, "close_mosaic": 10},
            }
        ]
    }
    augmentor = OnlinePolicyAugmentor(policy, seed=13, num_classes=3, sample_provider=sample_provider, total_epochs=1)

    result = augmentor.apply(image, labels, bboxes)

    assert result.image.shape == (128, 128, 3)
    assert len(result.labels) == 8
    assert np.all(result.bboxes[:, [0, 2]] >= 0)
    assert np.all(result.bboxes[:, [0, 2]] <= 128)
    assert np.all(result.bboxes[:, [1, 3]] >= 0)
    assert np.all(result.bboxes[:, [1, 3]] <= 128)
    assert augmentor.stats.to_dict()["mosaic4"]["applied"] == 1


def test_cutout_safe_does_not_cover_bbox_center() -> None:
    image, labels, bboxes = make_sample()
    center_x = int((bboxes[0, 0] + bboxes[0, 2]) / 2)
    center_y = int((bboxes[0, 1] + bboxes[0, 3]) / 2)
    image[center_y, center_x] = [250, 5, 15]
    policy = {
        "operations": [
            {
                "name": "cutout_safe",
                "prob": 1.0,
                "strength": 1.0,
                "params": {"max_holes": 8, "max_fraction": 0.45, "max_overlap_ratio": 0.0, "max_attempts": 80},
            }
        ]
    }
    augmentor = OnlinePolicyAugmentor(policy, seed=4, num_classes=3)

    result = augmentor.apply(image, labels, bboxes)

    np.testing.assert_array_equal(result.image[center_y, center_x], image[center_y, center_x])
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    assert augmentor.stats.to_dict()["cutout_safe"]["holes_requested"] == 8


def test_randaugment_like_does_not_break_labels() -> None:
    image, labels, bboxes = make_sample()
    policy = {
        "operations": [
            {
                "name": "randaugment_like",
                "prob": 1.0,
                "strength": 0.5,
                "params": {"min_ops": 1, "max_ops": 2, "candidate_ops": ["brightness", "contrast", "cutout_safe"]},
            }
        ]
    }
    augmentor = OnlinePolicyAugmentor(policy, seed=14, num_classes=3)

    result = augmentor.apply(image, labels, bboxes)

    assert result.image.shape == image.shape
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    assert augmentor.stats.to_dict()["ops"]["randaugment_like"]["applied"] == 1


def test_close_mosaic_disables_last_n_epochs() -> None:
    image, labels, bboxes = make_sample()
    policy = {"operations": [{"name": "mosaic4", "prob": 1.0, "strength": 1.0, "params": {"target_size": 128, "close_mosaic": 2}}]}
    augmentor = OnlinePolicyAugmentor(policy, seed=15, num_classes=3, total_epochs=5)
    augmentor.set_epoch(4, 5)

    result = augmentor.apply(image, labels, bboxes)

    assert result.image.shape == image.shape
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    assert augmentor.stats.to_dict()["ops"]["mosaic4"]["skipped_close_mosaic"] == 1


def test_online_policy_random_execution_keeps_class_ids() -> None:
    image, labels, bboxes = make_sample()
    policy = {
        "operations": [
            {"name": "brightness", "prob": 0.5, "strength": 0.3, "params": {}},
            {"name": "horizontal_flip", "prob": 0.5, "strength": 1.0, "params": {}},
        ]
    }
    augmentor = OnlinePolicyAugmentor(policy, seed=7, num_classes=3)

    seen_applied = False
    for _ in range(12):
        result = augmentor.apply(image, labels, bboxes)
        assert set(result.labels.tolist()) == set(labels.tolist())
        assert result.bboxes.shape[1] == 4
        seen_applied = seen_applied or bool(result.audit["applied_ops"])

    assert seen_applied
    assert augmentor.stats.to_dict()["class_id_oob_count"] == 0


def test_output_labels_remain_legal_after_mild_translate() -> None:
    image, labels, _ = make_sample()
    bboxes = np.asarray([[0, 1, 16, 18], [80, 52, 95, 63]], dtype=np.float32)
    policy = {
        "operations": [
            {"name": "mild_translate", "prob": 1.0, "strength": 1.0, "params": {"dx": 10, "dy": -8, "max_translate": 0.05}}
        ]
    }
    augmentor = OnlinePolicyAugmentor(policy, seed=8, num_classes=3)

    result = augmentor.apply(image, labels, bboxes)

    assert np.all(result.labels >= 0)
    assert np.all(result.labels < 3)
    assert np.all(result.bboxes[:, [0, 2]] >= 0)
    assert np.all(result.bboxes[:, [0, 2]] <= image.shape[1])
    assert np.all(result.bboxes[:, [1, 3]] >= 0)
    assert np.all(result.bboxes[:, [1, 3]] <= image.shape[0])


def test_copy_paste_pending_when_not_enabled_does_not_change_sample() -> None:
    image, labels, bboxes = make_sample()
    stats = OnlineAugmentationStats()
    policy = {
        "operations": [
            {"name": "copy_paste", "prob": 1.0, "strength": 1.0, "params": {"max_paste_count": 2}}
        ]
    }
    augmentor = OnlinePolicyAugmentor(policy, seed=9, num_classes=3, copy_paste_enabled=False, stats=stats)

    result = augmentor.apply(image, labels, bboxes)

    np.testing.assert_array_equal(result.image, image)
    np.testing.assert_array_equal(result.labels, labels)
    np.testing.assert_allclose(result.bboxes, bboxes)
    payload = stats.to_dict()
    assert payload["copy_paste"]["pending"] == 1
    assert payload["ops"]["copy_paste"]["skipped_copy_paste_pending"] == 1
