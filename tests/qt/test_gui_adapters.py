from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from AutoAugment.augmentations import get_augmentation, list_augmentations
from AutoAugment.policies import Policy
from gui.adapters import BackendCapabilityAdapter, PolicyAdapter
from gui.models import ExperimentConfig
from gui.models.experiment_config import (
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_FIXED_CATF,
    TRAINING_MODE_INTELLIGENT,
    TRAINING_MODE_LEGACY_SEARCH,
    TRAINING_MODE_NORMAL,
)
from gui.services import ResultLoader


def test_gui_augmentation_capability_list_loads_real_backend_functions() -> None:
    adapter = BackendCapabilityAdapter()
    adapter.validate_schema()
    schemas = adapter.list_augmentations()
    names = [schema.name for schema in schemas]
    assert names == sorted(list_augmentations())
    assert names
    for name in names:
        assert callable(get_augmentation(name))


def test_policy_json_generation_is_backend_compatible() -> None:
    policy_adapter = PolicyAdapter()
    operations = [
        policy_adapter.default_operation("brightness", prob=0.3, strength=0.4),
        policy_adapter.default_operation("horizontal_flip", prob=1.0, strength=1.0),
    ]
    policy_dict = policy_adapter.policy_dict(operations, name="unit_gui_policy")
    assert policy_dict["name"] == "unit_gui_policy"
    assert policy_dict["operations"][0]["name"] == "brightness"
    assert "enabled" not in policy_dict["operations"][0]
    loaded = Policy.from_dict(policy_dict)
    assert loaded.to_dict() == policy_dict


def test_policy_adapter_loads_policy_dict_and_operations() -> None:
    tmp_path = workspace_tmp("policy_load")
    policy_path = tmp_path / "custom_policy.json"
    policy = {
        "name": "custom_preview_policy",
        "operations": [
            {"name": "contrast", "prob": 0.8, "strength": 0.6, "params": {"max_delta": 0.4}},
        ],
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    policy_adapter = PolicyAdapter()
    loaded = policy_adapter.load_policy_dict(policy_path)
    operations = policy_adapter.operations_from_policy_dict(loaded)
    assert loaded["name"] == "custom_preview_policy"
    assert operations == [
        {"enabled": True, "name": "contrast", "prob": 0.8, "strength": 0.6, "params": {"max_delta": 0.4}}
    ]


def workspace_tmp(name: str) -> Path:
    path = Path("outputs") / "tests" / "qt_tmp" / f"{name}_{uuid.uuid4().hex}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_result_loader_tolerates_corrupt_json() -> None:
    tmp_path = workspace_tmp("corrupt_json")
    (tmp_path / "summary.json").write_text("{", encoding="utf-8")
    (tmp_path / "trials.json").write_text(
        json.dumps(
            [
                {
                    "trial_index": 0,
                    "score": 0.7,
                    "metrics": {"bbox_valid_rate": 1.0, "diversity_score": 0.5},
                    "policy": {"name": "p", "operations": []},
                }
            ]
        ),
        encoding="utf-8",
    )
    data = ResultLoader(Path.cwd()).load(tmp_path)
    assert data["read_errors"]
    assert len(data["trials"]) == 1
    assert data["metrics"]["bbox_valid_rate"] == [1.0]


def test_experiment_config_unifies_dataset_policy_and_run_params() -> None:
    tmp_path = workspace_tmp("experiment_config")
    policy = {
        "name": "gui_policy",
        "operations": [{"name": "brightness", "prob": 0.5, "strength": 0.5, "params": {"max_delta": 0.2}}],
    }
    cfg = ExperimentConfig.from_gui(
        dataset_path="dataset",
        policy=policy,
        params={
            "output_dir": str(tmp_path),
            "trials": 3,
            "epochs": 5,
            "imgsz": 640,
            "workers": 0,
            "batch": 4,
            "seed": 123,
            "proxy_prefilter": True,
            "real_yolo_validation": False,
            "smoke_test": False,
        },
    )
    saved = cfg.save(tmp_path / "gui_experiment_config.json")
    loaded = ExperimentConfig.from_dict(json.loads(saved.read_text(encoding="utf-8")))
    assert loaded.dataset_path == "dataset"
    assert loaded.policy == policy
    assert loaded.run_mode == TRAINING_MODE_CUSTOM
    assert loaded.proxy_prefilter is False
    assert loaded.evaluator == "proxy"


def test_training_mode_profiles_hide_catf_ablation_by_default() -> None:
    cfg = ExperimentConfig.from_gui(
        dataset_path="dataset",
        policy={"name": "gui_policy", "operations": []},
        params={"run_mode": TRAINING_MODE_INTELLIGENT, "trials": 8, "output_dir": "outputs", "real_yolo_validation": True},
    )
    profile = cfg.training_profile()
    assert cfg.run_mode == TRAINING_MODE_INTELLIGENT
    assert cfg.trials == 8
    assert cfg.proxy_prefilter is True
    assert cfg.adaptive_policy is True
    assert profile["profile_id"] == "preserve_weak_image_only_catf"
    assert profile["backend_profile"] == "Preserve-Weak Image-only CATF"
    assert profile["image_only"] is True
    assert profile["sampler_only"] is False
    assert profile["weighted_index_list"] is False
    assert profile["sampled_distribution_changed"] is False

    normal = ExperimentConfig.from_gui(
        dataset_path="dataset",
        policy={"name": "gui_policy", "operations": []},
        params={"run_mode": TRAINING_MODE_NORMAL, "trials": 8, "output_dir": "outputs", "real_yolo_validation": True},
    )
    assert normal.trials == 1
    assert normal.proxy_prefilter is False
    assert normal.adaptive_policy is False
    assert normal.training_profile()["profile_id"] == "clean_yolo_default"

    custom = ExperimentConfig.from_gui(
        dataset_path="dataset",
        policy={"name": "gui_policy", "operations": []},
        params={"run_mode": TRAINING_MODE_CUSTOM, "trials": 8, "output_dir": "outputs", "real_yolo_validation": True},
    )
    assert custom.trials == 1
    assert custom.proxy_prefilter is False
    assert custom.adaptive_policy is False

    fixed = ExperimentConfig.from_gui(
        dataset_path="dataset",
        policy={"name": "gui_policy", "operations": []},
        params={"run_mode": TRAINING_MODE_FIXED_CATF, "trials": 8, "output_dir": "outputs", "real_yolo_validation": True},
    )
    assert fixed.trials == 1
    assert fixed.training_profile()["hidden_by_default"] is True

    legacy = ExperimentConfig.from_gui(
        dataset_path="dataset",
        policy={"name": "gui_policy", "operations": []},
        params={"run_mode": TRAINING_MODE_LEGACY_SEARCH, "trials": 8, "output_dir": "outputs", "real_yolo_validation": True},
    )
    assert legacy.trials == 8
    assert legacy.proxy_prefilter is False
    assert legacy.adaptive_policy is False
