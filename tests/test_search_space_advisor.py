from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from AutoAugment.policies import SearchSpace, default_detection_search_space


def test_search_space_loads_advisor_json_and_applies_ranges() -> None:
    path = fresh_json(
        {
            "allowed_operations": ["contrast", "clahe", "motion_blur"],
            "operation_weights": {"clahe": 3.0, "motion_blur": 0.1},
            "strength_ranges": {"motion_blur": [0.0, 0.2]},
            "prob_ranges": {"contrast": [0.4, 0.6]},
            "forbidden_combinations": [["contrast", "motion_blur"]],
            "max_ops_per_policy": 2,
        }
    )
    space = SearchSpace.from_advisor_json(path, operation_count_range=(2, 4))
    assert [operation.name for operation in space.operations] == ["contrast", "clahe", "motion_blur"]
    assert space.operation_count_range == (2, 2)
    contrast = next(operation for operation in space.operations if operation.name == "contrast")
    motion_blur = next(operation for operation in space.operations if operation.name == "motion_blur")
    assert contrast.prob_range == (0.4, 0.6)
    assert motion_blur.strength_range == (0.0, 0.2)


def test_operation_weights_affect_sampling() -> None:
    path = fresh_json(
        {
            "allowed_operations": ["contrast", "clahe"],
            "operation_weights": {"contrast": 0.01, "clahe": 20.0},
            "max_ops_per_policy": 1,
        }
    )
    space = SearchSpace.from_advisor_json(path, operation_count_range=(1, 1))
    rng = np.random.default_rng(42)
    names = [space.sample_policy(rng).operations[0].name for _ in range(200)]
    assert names.count("clahe") > 190


def test_forbidden_combinations_are_not_sampled() -> None:
    path = fresh_json(
        {
            "allowed_operations": ["contrast", "clahe", "motion_blur"],
            "operation_weights": {"contrast": 1.0, "clahe": 1.0, "motion_blur": 1.0},
            "forbidden_combinations": [["contrast", "motion_blur"]],
            "max_ops_per_policy": 2,
        }
    )
    space = SearchSpace.from_advisor_json(path, operation_count_range=(2, 2))
    rng = np.random.default_rng(7)
    for _ in range(100):
        names = {operation.name for operation in space.sample_policy(rng).operations}
        assert not {"contrast", "motion_blur"}.issubset(names)


def test_default_search_space_still_works() -> None:
    space = default_detection_search_space(operation_count_range=(2, 2))
    policy = space.sample_policy(np.random.default_rng(3))
    assert len(policy.operations) == 2


def fresh_json(data: dict) -> Path:
    path = Path("outputs/tests/pytest_tmp") / f"advisor_{uuid.uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
