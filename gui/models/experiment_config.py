from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TRAINING_MODE_NORMAL = "normal_train"
TRAINING_MODE_CUSTOM = "custom_augment_train"
TRAINING_MODE_INTELLIGENT = "intelligent_auto_augment"
TRAINING_MODE_FIXED_CATF = "fixed_catf_v2_ablation"
TRAINING_MODE_LEGACY_SEARCH = "legacy_autoaugment_search"

LEGACY_MODE_ALIASES = {
    "manual_policy": TRAINING_MODE_CUSTOM,
    "auto_search": TRAINING_MODE_LEGACY_SEARCH,
}

SINGLE_TRIAL_MODES = {
    TRAINING_MODE_NORMAL,
    TRAINING_MODE_CUSTOM,
    TRAINING_MODE_FIXED_CATF,
}

TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    TRAINING_MODE_NORMAL: {
        "profile_id": "clean_yolo_default",
        "display_name": "普通训练",
        "catf_enabled": False,
        "auto_augment_enabled": False,
        "image_only": False,
        "sampler_only": False,
        "weighted_index_list": False,
        "sampled_distribution_changed": False,
        "recommended": True,
    },
    TRAINING_MODE_CUSTOM: {
        "profile_id": "custom_yolo_augmentation",
        "display_name": "自定义增强训练",
        "catf_enabled": False,
        "auto_augment_enabled": False,
        "image_only": False,
        "sampler_only": False,
        "weighted_index_list": False,
        "sampled_distribution_changed": False,
        "recommended": True,
    },
    TRAINING_MODE_INTELLIGENT: {
        "profile_id": "preserve_weak_image_only_catf",
        "backend_profile": "Preserve-Weak Image-only CATF",
        "display_name": "智能自动增强训练",
        "catf_enabled": True,
        "auto_augment_enabled": True,
        "image_only": True,
        "risk_control": "preserve/weak/no-op",
        "sampler_only": False,
        "weighted_index_list": False,
        "sampled_distribution_changed": False,
        "recommended": True,
    },
    TRAINING_MODE_FIXED_CATF: {
        "profile_id": "fixed_catf_v2",
        "display_name": "fixed CATF-v2 对照实验",
        "catf_enabled": True,
        "auto_augment_enabled": False,
        "image_only": False,
        "sampler_only": False,
        "weighted_index_list": False,
        "sampled_distribution_changed": False,
        "recommended": False,
        "hidden_by_default": True,
        "purpose": "ablation/reproduce/debug",
    },
    TRAINING_MODE_LEGACY_SEARCH: {
        "profile_id": "legacy_autoaugment_search",
        "display_name": "Legacy AutoAugment Search",
        "catf_enabled": False,
        "auto_augment_enabled": True,
        "image_only": False,
        "sampler_only": False,
        "weighted_index_list": False,
        "sampled_distribution_changed": False,
        "recommended": False,
        "hidden_by_default": True,
        "purpose": "legacy_search",
    },
}


def normalize_run_mode(value: str | None) -> str:
    mode = str(value or TRAINING_MODE_CUSTOM).strip()
    mode = LEGACY_MODE_ALIASES.get(mode, mode)
    return mode if mode in TRAINING_PROFILES else TRAINING_MODE_CUSTOM


def training_profile_for_mode(value: str | None) -> dict[str, Any]:
    mode = normalize_run_mode(value)
    return dict(TRAINING_PROFILES[mode])


@dataclass
class ExperimentConfig:
    dataset_path: str
    output_dir: str
    policy: dict[str, Any] = field(default_factory=lambda: {"name": "gui_policy", "operations": []})
    trials: int = 5
    epochs: int = 5
    imgsz: int = 640
    workers: int = 0
    batch: int = 8
    seed: int = 42
    samples: int = 20
    evaluator: str = "proxy"
    metric: str = "map50"
    model: str = "yolov8n.pt"
    run_mode: str = TRAINING_MODE_CUSTOM
    proxy_prefilter: bool = False
    real_yolo_validation: bool = False
    smoke_test: bool = True
    adaptive_policy: bool = True
    diagnose_trial_errors: bool = False
    entry_script: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_mode = normalize_run_mode(self.run_mode)
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
        real_yolo = bool(params.get("real_yolo_validation", False))
        run_mode = normalize_run_mode(str(params.get("run_mode", TRAINING_MODE_CUSTOM) or TRAINING_MODE_CUSTOM))
        smoke = bool(params.get("smoke_test", not real_yolo))
        trials = int(params.get("trials", 5))
        if run_mode in SINGLE_TRIAL_MODES:
            trials = 1
        proxy_prefilter = bool(params.get("proxy_prefilter", False))
        adaptive_policy = bool(params.get("adaptive_policy", True))
        if run_mode in {TRAINING_MODE_NORMAL, TRAINING_MODE_CUSTOM, TRAINING_MODE_FIXED_CATF, TRAINING_MODE_LEGACY_SEARCH}:
            proxy_prefilter = False
            adaptive_policy = False
        if run_mode == TRAINING_MODE_INTELLIGENT:
            proxy_prefilter = True
            adaptive_policy = True
        evaluator = "train_yolo" if real_yolo else "proxy"
        if smoke and not real_yolo:
            params = {**params, "samples": min(int(params.get("samples", 4)), 4)}
            trials = 1
        extra = dict(params.get("extra", {}) or {})
        extra["training_profile"] = training_profile_for_mode(run_mode)
        return cls(
            dataset_path=dataset_path,
            output_dir=str(params.get("output_dir", "")),
            policy=policy,
            trials=trials,
            epochs=int(params.get("epochs", 5)),
            imgsz=int(params.get("imgsz", 640)),
            workers=int(params.get("workers", 0)),
            batch=int(params.get("batch", 8)),
            seed=int(params.get("seed", 42)),
            samples=int(params.get("samples", 20)),
            evaluator=evaluator,
            metric=str(params.get("metric", "map50")),
            model=str(params.get("model", "yolov8n.pt")),
            run_mode=run_mode,
            proxy_prefilter=proxy_prefilter,
            real_yolo_validation=real_yolo,
            smoke_test=smoke,
            adaptive_policy=adaptive_policy,
            diagnose_trial_errors=bool(params.get("diagnose_trial_errors", False)),
            entry_script=str(params.get("entry_script", "")),
            extra=extra,
        )

    def training_profile(self) -> dict[str, Any]:
        profile = training_profile_for_mode(self.run_mode)
        profile.update(dict((self.extra or {}).get("training_profile", {}) or {}))
        profile["run_mode"] = self.run_mode
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "output_dir": self.output_dir,
            "policy": self.policy,
            "trials": self.trials,
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "workers": self.workers,
            "batch": self.batch,
            "seed": self.seed,
            "samples": self.samples,
            "evaluator": self.evaluator,
            "metric": self.metric,
            "model": self.model,
            "run_mode": self.run_mode,
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
            trials=int(data.get("trials", 5)),
            epochs=int(data.get("epochs", 5)),
            imgsz=int(data.get("imgsz", 640)),
            workers=int(data.get("workers", 0)),
            batch=int(data.get("batch", 8)),
            seed=int(data.get("seed", 42)),
            samples=int(data.get("samples", 20)),
            evaluator=str(data.get("evaluator", "proxy")),
            metric=str(data.get("metric", "map50")),
            model=str(data.get("model", "yolov8n.pt")),
            run_mode=normalize_run_mode(str(data.get("run_mode", TRAINING_MODE_CUSTOM) or TRAINING_MODE_CUSTOM)),
            proxy_prefilter=bool(data.get("proxy_prefilter", False)),
            real_yolo_validation=bool(data.get("real_yolo_validation", False)),
            smoke_test=bool(data.get("smoke_test", False)),
            adaptive_policy=bool(data.get("adaptive_policy", True)),
            diagnose_trial_errors=bool(data.get("diagnose_trial_errors", False)),
            entry_script=str(data.get("entry_script", "")),
            extra=dict(data.get("extra", {}) or {}),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target
