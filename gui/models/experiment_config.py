from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TRAINING_MODE_CUSTOM = "custom_yolo_augment_train"
TRAINING_MODE_YOLO_DEFAULT = "yolo_default_train"
TRAINING_MODE_PRESERVE_WEAK = "preserve_weak_image_only_catf"

# Backward-compatible names used by older GUI configs/tests. They are not
# displayed in the experiment page.
TRAINING_MODE_NORMAL = TRAINING_MODE_YOLO_DEFAULT
TRAINING_MODE_INTELLIGENT = TRAINING_MODE_PRESERVE_WEAK
TRAINING_MODE_FIXED_CATF = "fixed_catf_v2_ablation"
TRAINING_MODE_LEGACY_SEARCH = "legacy_autoaugment_search"

VISIBLE_TRAINING_MODES = (
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_YOLO_DEFAULT,
    TRAINING_MODE_PRESERVE_WEAK,
)

LEGACY_MODE_ALIASES = {
    "manual_policy": TRAINING_MODE_CUSTOM,
    "auto_search": TRAINING_MODE_PRESERVE_WEAK,
    "custom_augment_train": TRAINING_MODE_CUSTOM,
    "normal_train": TRAINING_MODE_YOLO_DEFAULT,
    "intelligent_auto_augment": TRAINING_MODE_PRESERVE_WEAK,
    TRAINING_MODE_FIXED_CATF: TRAINING_MODE_PRESERVE_WEAK,
    TRAINING_MODE_LEGACY_SEARCH: TRAINING_MODE_PRESERVE_WEAK,
}

PRESERVE_WEAK_DATA_DEFAULT = (
    "outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml"
)

PRESERVE_WEAK_DEFAULTS: dict[str, Any] = {
    "dataset_path": PRESERVE_WEAK_DATA_DEFAULT,
    "model": "yolo11n.pt",
    "epochs": 50,
    "imgsz": 1024,
    "batch": 2,
    "workers": 0,
    "device": "0",
    "seed": 0,
}

YOLO_AUG_DEFAULTS: dict[str, float | int] = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 10,
}

PRESERVE_WEAK_ALGORITHM_LOCK: dict[str, Any] = {
    "image_only_mainline": True,
    "preserve_original_enabled": True,
    "weak_image_aug_enabled": True,
    "weak_only_for_moderate_risk": True,
    "attenuation_ratio": 0.25,
    "disable_sampler_only": True,
    "sampler_only": False,
    "weighted_index_list": False,
    "sampled_distribution_changed": False,
}

TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    TRAINING_MODE_CUSTOM: {
        "profile_id": "custom_yolo_augmentation",
        "display_name": "自定义增强策略训练",
        "catf_enabled": False,
        "auto_augment_enabled": False,
        "custom_yolo_augment_enabled": True,
        "sampler_only": False,
        "weighted_index_list": False,
        "sampled_distribution_changed": False,
    },
    TRAINING_MODE_YOLO_DEFAULT: {
        "profile_id": "clean_yolo_default",
        "display_name": "YOLO 默认训练",
        "catf_enabled": False,
        "auto_augment_enabled": False,
        "custom_yolo_augment_enabled": False,
        "sampler_only": False,
        "weighted_index_list": False,
        "sampled_distribution_changed": False,
    },
    TRAINING_MODE_PRESERVE_WEAK: {
        "profile_id": "preserve_weak_image_only_catf",
        "backend_profile": "Preserve-Weak Image-only CATF",
        "display_name": "Preserve-Weak Image-only CATF",
        "catf_enabled": True,
        "auto_augment_enabled": True,
        "image_only": True,
        "risk_control": "preserve/weak/no-op",
        "sampler_only": False,
        "weighted_index_list": False,
        "sampled_distribution_changed": False,
        "algorithm_lock": dict(PRESERVE_WEAK_ALGORITHM_LOCK),
    },
}


def normalize_run_mode(value: str | None) -> str:
    mode = str(value or TRAINING_MODE_PRESERVE_WEAK).strip()
    mode = LEGACY_MODE_ALIASES.get(mode, mode)
    return mode if mode in TRAINING_PROFILES else TRAINING_MODE_PRESERVE_WEAK


def training_profile_for_mode(value: str | None) -> dict[str, Any]:
    return dict(TRAINING_PROFILES[normalize_run_mode(value)])


@dataclass
class ExperimentConfig:
    dataset_path: str
    output_dir: str
    policy: dict[str, Any] = field(default_factory=lambda: {"name": "gui_policy", "operations": []})
    epochs: int = 50
    imgsz: int = 1024
    workers: int = 0
    batch: int = 2
    seed: int = 0
    device: str = "0"
    metric: str = "map50"
    model: str = "yolo11n.pt"
    run_mode: str = TRAINING_MODE_PRESERVE_WEAK
    yolo_aug_params: dict[str, Any] = field(default_factory=dict)
    algorithm_lock: dict[str, Any] = field(default_factory=dict)
    trials: int = 1
    samples: int = 0
    evaluator: str = "train_yolo"
    proxy_prefilter: bool = False
    real_yolo_validation: bool = True
    smoke_test: bool = False
    adaptive_policy: bool = False
    diagnose_trial_errors: bool = False
    entry_script: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_mode = normalize_run_mode(self.run_mode)
        self.device = str(self.device)
        self.yolo_aug_params = dict(self.yolo_aug_params or {})
        self.algorithm_lock = self._normalized_algorithm_lock()
        self.extra = dict(self.extra or {})
        self.extra["training_profile"] = self.training_profile()

    @classmethod
    def from_gui(
        cls,
        *,
        dataset_path: str,
        policy: dict[str, Any],
        params: dict[str, Any],
    ) -> "ExperimentConfig":
        run_mode = normalize_run_mode(str(params.get("run_mode", TRAINING_MODE_PRESERVE_WEAK)))
        defaults = PRESERVE_WEAK_DEFAULTS if run_mode == TRAINING_MODE_PRESERVE_WEAK else {}

        yolo_aug_params = dict(params.get("yolo_aug_params", {}) or {})
        if run_mode != TRAINING_MODE_CUSTOM:
            yolo_aug_params = {}

        algorithm_lock = dict(PRESERVE_WEAK_ALGORITHM_LOCK) if run_mode == TRAINING_MODE_PRESERVE_WEAK else {
            "sampler_only": False,
            "weighted_index_list": False,
            "sampled_distribution_changed": False,
        }

        extra = dict(params.get("extra", {}) or {})
        extra["training_profile"] = training_profile_for_mode(run_mode)

        return cls(
            dataset_path=str(params.get("dataset_path", dataset_path) or dataset_path or defaults.get("dataset_path", "")),
            output_dir=str(params.get("output_dir", "")),
            policy=policy,
            epochs=int(params.get("epochs", defaults.get("epochs", 50))),
            imgsz=int(params.get("imgsz", defaults.get("imgsz", 640))),
            workers=int(params.get("workers", defaults.get("workers", 0))),
            batch=int(params.get("batch", defaults.get("batch", 8))),
            seed=int(params.get("seed", defaults.get("seed", 0))),
            device=str(params.get("device", defaults.get("device", "0"))),
            metric=str(params.get("metric", "map50")),
            model=str(params.get("model", defaults.get("model", "yolo11n.pt"))),
            run_mode=run_mode,
            yolo_aug_params=yolo_aug_params,
            algorithm_lock=algorithm_lock,
            trials=1,
            samples=0,
            evaluator="train_yolo",
            proxy_prefilter=False,
            real_yolo_validation=True,
            smoke_test=False,
            adaptive_policy=False,
            diagnose_trial_errors=False,
            entry_script=str(params.get("entry_script", "")),
            extra=extra,
        )

    def _normalized_algorithm_lock(self) -> dict[str, Any]:
        if self.run_mode == TRAINING_MODE_PRESERVE_WEAK:
            lock = dict(PRESERVE_WEAK_ALGORITHM_LOCK)
            lock.update(dict(self.algorithm_lock or {}))
            lock["disable_sampler_only"] = True
            lock["sampler_only"] = False
            lock["weighted_index_list"] = False
            lock["sampled_distribution_changed"] = False
            return lock
        return {
            "sampler_only": False,
            "weighted_index_list": False,
            "sampled_distribution_changed": False,
        }

    def training_profile(self) -> dict[str, Any]:
        profile = training_profile_for_mode(self.run_mode)
        profile.update(dict((self.extra or {}).get("training_profile", {}) or {}))
        profile["run_mode"] = self.run_mode
        profile["sampler_only"] = False
        profile["weighted_index_list"] = False
        profile["sampled_distribution_changed"] = False
        if self.run_mode == TRAINING_MODE_PRESERVE_WEAK:
            profile["algorithm_lock"] = dict(self.algorithm_lock)
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "policy": self.policy,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "workers": self.workers,
            "batch": self.batch,
            "seed": self.seed,
            "device": self.device,
            "metric": self.metric,
            "model": self.model,
            "run_mode": self.run_mode,
            "yolo_aug_params": self.yolo_aug_params,
            "algorithm_lock": self.algorithm_lock,
            "trials": self.trials,
            "samples": self.samples,
            "evaluator": self.evaluator,
            "proxy_prefilter": self.proxy_prefilter,
            "real_yolo_validation": self.real_yolo_validation,
            "smoke_test": self.smoke_test,
            "adaptive_policy": self.adaptive_policy,
            "diagnose_trial_errors": self.diagnose_trial_errors,
            "entry_script": self.entry_script,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            dataset_path=str(data.get("dataset_path", "")),
            output_dir=str(data.get("output_dir", "")),
            policy=dict(data.get("policy", {}) or {"name": "gui_policy", "operations": []}),
            epochs=int(data.get("epochs", 50)),
            imgsz=int(data.get("imgsz", 1024)),
            workers=int(data.get("workers", 0)),
            batch=int(data.get("batch", 2)),
            seed=int(data.get("seed", 0)),
            device=str(data.get("device", "0")),
            metric=str(data.get("metric", "map50")),
            model=str(data.get("model", "yolo11n.pt")),
            run_mode=normalize_run_mode(str(data.get("run_mode", TRAINING_MODE_PRESERVE_WEAK))),
            yolo_aug_params=dict(data.get("yolo_aug_params", {}) or {}),
            algorithm_lock=dict(data.get("algorithm_lock", {}) or {}),
            trials=int(data.get("trials", 1)),
            samples=int(data.get("samples", 0)),
            evaluator=str(data.get("evaluator", "train_yolo")),
            proxy_prefilter=bool(data.get("proxy_prefilter", False)),
            real_yolo_validation=bool(data.get("real_yolo_validation", True)),
            smoke_test=bool(data.get("smoke_test", False)),
            adaptive_policy=bool(data.get("adaptive_policy", False)),
            diagnose_trial_errors=bool(data.get("diagnose_trial_errors", False)),
            entry_script=str(data.get("entry_script", "")),
            extra=dict(data.get("extra", {}) or {}),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target
