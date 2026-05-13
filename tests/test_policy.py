from __future__ import annotations

import numpy as np

from AutoAugment.policies import OperationSpec, Policy, apply_policy


def test_policy_json_round_trip() -> None:
    policy = Policy(
        name="policy_test",
        operations=[
            OperationSpec("brightness", prob=0.8, strength=0.3, params={"max_delta": 0.2}),
            OperationSpec("horizontal_flip", prob=1.0, strength=1.0),
        ],
    )
    loaded = Policy.from_json(policy.to_json())
    assert loaded.to_dict() == policy.to_dict()


def test_apply_policy_runs_and_respects_bbox_updates() -> None:
    image = np.full((20, 40, 3), 100, dtype=np.uint8)
    labels = np.asarray([1], dtype=np.int64)
    bboxes = np.asarray([[4, 5, 14, 15]], dtype=np.float32)
    policy = Policy(
        name="policy_test",
        operations=[
            OperationSpec("brightness", prob=1.0, strength=0.2),
            OperationSpec("horizontal_flip", prob=1.0, strength=1.0),
        ],
    )
    out_image, out_labels, out_bboxes = apply_policy(image, labels, bboxes, policy, rng=np.random.default_rng(1))
    assert out_image.shape == image.shape
    np.testing.assert_array_equal(out_labels, labels)
    np.testing.assert_allclose(out_bboxes, np.asarray([[26, 5, 36, 15]], dtype=np.float32))
