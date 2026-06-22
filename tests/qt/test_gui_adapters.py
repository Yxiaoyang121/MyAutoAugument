from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from AutoAugment.augmentations import get_augmentation, list_augmentations
from AutoAugment.policies import Policy
from gui.adapters import BackendCapabilityAdapter, PolicyAdapter
from gui.adapters.gui_experiment_launcher import build_training_command
from gui.models import ExperimentConfig
from gui.models.experiment_config import (
    PRESERVE_WEAK_DATA_DEFAULT,
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_PRESERVE_WEAK,
    TRAINING_MODE_YOLO_DEFAULT,
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
            "epochs": 5,
            "imgsz": 640,
            "workers": 0,
            "batch": 4,
            "seed": 123,
            "device": "cpu",
            "run_mode": TRAINING_MODE_CUSTOM,
            "yolo_aug_params": {"hsv_h": 0.02, "fliplr": 0.6},
        },
    )
    saved = cfg.save(tmp_path / "gui_experiment_config.json")
    loaded = ExperimentConfig.from_dict(json.loads(saved.read_text(encoding="utf-8")))
    assert loaded.dataset_path == "dataset"
    assert loaded.policy == policy
    assert loaded.run_mode == TRAINING_MODE_CUSTOM
    assert loaded.proxy_prefilter is False
    assert loaded.evaluator == "train_yolo"
    assert loaded.device == "cpu"
    assert loaded.yolo_aug_params["fliplr"] == 0.6


def test_training_modes_are_limited_to_custom_default_and_preserve_weak() -> None:
    cfg = ExperimentConfig.from_gui(
        dataset_path=PRESERVE_WEAK_DATA_DEFAULT,
        policy={"name": "gui_policy", "operations": []},
        params={"run_mode": TRAINING_MODE_PRESERVE_WEAK, "output_dir": "outputs"},
    )
    profile = cfg.training_profile()
    assert cfg.run_mode == TRAINING_MODE_PRESERVE_WEAK
    assert cfg.model == "yolo11n.pt"
    assert cfg.epochs == 50
    assert cfg.imgsz == 1024
    assert cfg.batch == 2
    assert cfg.workers == 0
    assert cfg.device == "0"
    assert cfg.seed == 0
    assert profile["profile_id"] == "preserve_weak_image_only_catf"
    assert profile["backend_profile"] == "Preserve-Weak Image-only CATF"
    assert profile["image_only"] is True
    assert profile["sampler_only"] is False
    assert profile["weighted_index_list"] is False
    assert profile["sampled_distribution_changed"] is False
    assert cfg.algorithm_lock["disable_sampler_only"] is True
    assert cfg.algorithm_lock["sampler_only"] is False
    assert cfg.algorithm_lock["weighted_index_list"] is False
    assert cfg.algorithm_lock["sampled_distribution_changed"] is False

    yolo_default = ExperimentConfig.from_gui(
        dataset_path="dataset",
        policy={"name": "gui_policy", "operations": []},
        params={"run_mode": TRAINING_MODE_YOLO_DEFAULT, "output_dir": "outputs", "model": "yolo11n.pt"},
    )
    assert yolo_default.proxy_prefilter is False
    assert yolo_default.adaptive_policy is False
    assert yolo_default.training_profile()["profile_id"] == "clean_yolo_default"

    custom = ExperimentConfig.from_gui(
        dataset_path="dataset",
        policy={"name": "gui_policy", "operations": []},
        params={"run_mode": TRAINING_MODE_CUSTOM, "output_dir": "outputs", "yolo_aug_params": {"mosaic": 0.2}},
    )
    assert custom.proxy_prefilter is False
    assert custom.adaptive_policy is False
    assert custom.training_profile()["custom_yolo_augment_enabled"] is True
    assert custom.yolo_aug_params == {"mosaic": 0.2}


def test_training_command_generation_matches_three_modes() -> None:
    output = workspace_tmp("commands")
    policy = {"name": "gui_policy", "operations": []}

    preserve = ExperimentConfig.from_gui(
        dataset_path=PRESERVE_WEAK_DATA_DEFAULT,
        policy=policy,
        params={"run_mode": TRAINING_MODE_PRESERVE_WEAK, "output_dir": str(output)},
    )
    preserve_command = build_training_command(preserve, output)
    preserve_text = " ".join(preserve_command)
    assert "scripts\\train_yolo_default_with_inloop_feedback.py" in preserve_text or "scripts/train_yolo_default_with_inloop_feedback.py" in preserve_text
    assert "--catf-version" in preserve_command and "v2" in preserve_command
    assert "--image-only-mainline" in preserve_command
    assert "--preserve-original-enabled" in preserve_command
    assert "--weak-image-aug-enabled" in preserve_command
    assert "--weak-only-for-moderate-risk" in preserve_command
    assert "--disable-sampler-only" in preserve_command
    assert "--industrial-aug-enabled" in preserve_command

    yolo_default = ExperimentConfig.from_gui(
        dataset_path="dataset/data.yaml",
        policy=policy,
        params={"run_mode": TRAINING_MODE_YOLO_DEFAULT, "output_dir": str(output), "model": "yolo11n.pt"},
    )
    yolo_command = build_training_command(yolo_default, output)
    yolo_text = " ".join(yolo_command)
    assert yolo_command[:3] == ["yolo", "detect", "train"]
    assert "--catf-version" not in yolo_command
    assert "mosaic=" not in yolo_text

    custom = ExperimentConfig.from_gui(
        dataset_path="dataset/data.yaml",
        policy=policy,
        params={
            "run_mode": TRAINING_MODE_CUSTOM,
            "output_dir": str(output),
            "model": "yolo11n.pt",
            "yolo_aug_params": {"hsv_h": 0.02, "mosaic": 0.3, "fliplr": 0.4},
        },
    )
    custom_command = build_training_command(custom, output)
    custom_text = " ".join(custom_command)
    assert custom_command[:3] == ["yolo", "detect", "train"]
    assert "hsv_h=0.02" in custom_text
    assert "mosaic=0.3" in custom_text
    assert "fliplr=0.4" in custom_text
    assert "--catf-version" not in custom_command
