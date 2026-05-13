from __future__ import annotations

import numpy as np

from AutoAugment.policies import Policy, default_detection_search_space


def test_search_space_samples_policy() -> None:
    space = default_detection_search_space(operation_count_range=(2, 4))
    policy = space.sample_policy(np.random.default_rng(4))
    assert isinstance(policy, Policy)
    assert 2 <= len(policy.operations) <= 4
    assert all(0.0 <= operation.prob <= 1.0 for operation in policy.operations)
    assert all(0.0 <= operation.strength <= 1.0 for operation in policy.operations)
