from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json

from AutoAugment.catf_v2.high_risk_class_ops import risk_info


BENEFIT_WEIGHTS = {
    "fn_recovery_rate": 1.0,
    "localization_iou_gain": 0.5,
    "low_conf_tp_conf_gain": 0.3,
}

RISK_WEIGHTS = {
    "fp_increase_rate": 1.0,
    "high_fp_spillover_rate": 1.2,
    "non_active_regression_rate": 1.0,
    "ok_class_false_activation": 1.0,
    "bbox_instability_rate": 0.5,
}

PRECISION_GATE_KEYS = (
    "estimated_precision_drop",
    "non_active_fp_delta",
    "high_confidence_fp_delta",
)

DEFAULT_PRECISION_GATE_THRESHOLDS = {
    "estimated_precision_drop": 0.005,
    "non_active_fp_delta": 0.005,
    "high_confidence_fp_delta": 0.0,
}

DEFAULT_CANDIDATE_POLICIES: dict[str, dict[str, Any]] = {
    "candidate_policy_0_noop": {
        "policy_id": "candidate_policy_0_noop",
        "action": "no_op",
        "op_list": [],
        "image_modification": False,
        "sample_weighting": False,
    },
    "candidate_policy_1_roi_texture": {
        "policy_id": "candidate_policy_1_roi_texture",
        "action": "roi_image_aug",
        "op_list": ["sharpen_mild", "local_contrast"],
        "image_modification": True,
        "sample_weighting": False,
    },
    "candidate_policy_1b_weak_roi_texture": {
        "policy_id": "candidate_policy_1b_weak_roi_texture",
        "action": "roi_image_aug",
        "op_list": ["local_contrast"],
        "image_modification": True,
        "sample_weighting": False,
        "weak_image_aug": True,
        "attenuation_ratio": 0.25,
        "max_aug_samples_per_interval": 16,
        "derived_from_policy_id": "candidate_policy_1_roi_texture",
    },
    "candidate_policy_2_roi_low_contrast": {
        "policy_id": "candidate_policy_2_roi_low_contrast",
        "action": "roi_image_aug",
        "op_list": ["local_contrast", "gamma"],
        "image_modification": True,
        "sample_weighting": False,
    },
    "candidate_policy_3_sampler_only": {
        "policy_id": "candidate_policy_3_sampler_only",
        "action": "sampler_only",
        "op_list": [],
        "image_modification": False,
        "sample_weighting": True,
        "sample_weighting_effective": False,
        "sample_weighting_status": "pending_dataloader_support",
    },
    "candidate_policy_4_reject_high_risk": {
        "policy_id": "candidate_policy_4_reject_high_risk",
        "action": "no_op",
        "op_list": [],
        "image_modification": False,
        "sample_weighting": False,
    },
}


@dataclass(frozen=True)
class ProbeConfig:
    mode: str = "development"
    development_probe_uses_existing_val_diagnostics: bool = True
    max_samples_per_bucket: int = 30
    min_evidence_count: int = 5
    min_diagnosis_confidence: float = 0.50


@dataclass
class ProbeSet:
    class_id: int
    candidate_policy_id: str
    mode: str = "development"
    development_probe_uses_existing_val_diagnostics: bool = True
    buckets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    evidence_count: int = 0
    diagnosis_confidence: float = 0.0
    audit_priors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "candidate_policy_id": self.candidate_policy_id,
            "mode": self.mode,
            "development_probe_uses_existing_val_diagnostics": bool(
                self.development_probe_uses_existing_val_diagnostics
            ),
            "bucket_counts": {key: len(value) for key, value in sorted(self.buckets.items())},
            "evidence_count": int(self.evidence_count),
            "diagnosis_confidence": float(self.diagnosis_confidence),
            "audit_priors": deepcopy(self.audit_priors),
        }


def candidate_policy_catalog() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULT_CANDIDATE_POLICIES)


def build_probe_set(
    *,
    candidate_class_id: int,
    candidate_policy_id: str,
    diagnosis_record: dict[str, Any] | None = None,
    issue_record: dict[str, Any] | None = None,
    high_fp_samples: list[dict[str, Any]] | None = None,
    non_active_samples: list[dict[str, Any]] | None = None,
    config: ProbeConfig | None = None,
) -> ProbeSet:
    """Build a class-agnostic probe set description from available diagnostics.

    Development mode may use existing validation diagnostics. Paper mode should
    pass train/probe-split examples through the same buckets without involving
    the final evaluation split.
    """

    cfg = config or ProbeConfig()
    diagnosis = diagnosis_record or {}
    issue = issue_record or {}
    evidence_count = int(
        issue.get("evidence_count")
        or diagnosis.get("evidence_count")
        or diagnosis.get("instances")
        or diagnosis.get("val_instances")
        or 0
    )
    diagnosis_confidence = float(
        issue.get("diagnosis_confidence") or diagnosis.get("diagnosis_confidence") or 0.0
    )
    max_count = max(1, int(cfg.max_samples_per_bucket))
    buckets = {
        "fn_samples": _placeholder_samples(
            min(max_count, max(0, int(issue.get("fn_count") or diagnosis.get("fn_count") or evidence_count))),
            candidate_class_id,
            "fn",
        ),
        "low_conf_tp_samples": _placeholder_samples(
            min(max_count, max(0, int(issue.get("low_conf_tp_count") or diagnosis.get("low_conf_tp_count") or 0))),
            candidate_class_id,
            "low_conf_tp",
        ),
        "weak_localization_samples": _placeholder_samples(
            min(max_count, max(0, int(issue.get("weak_loc_count") or diagnosis.get("weak_loc_count") or 0))),
            candidate_class_id,
            "weak_localization",
        ),
        "high_fp_risky_samples": list((high_fp_samples or [])[:max_count]),
        "non_active_class_samples": list((non_active_samples or [])[:max_count]),
    }
    catalog = candidate_policy_catalog()
    op_list = catalog.get(candidate_policy_id, {}).get("op_list", [])
    audit_priors = []
    for op_name in op_list:
        prior = risk_info(candidate_class_id, op_name)
        if prior:
            item = deepcopy(prior)
            item["class_id"] = int(candidate_class_id)
            item["op_name"] = str(op_name)
            item["audit_prior_only"] = True
            audit_priors.append(item)
    return ProbeSet(
        class_id=int(candidate_class_id),
        candidate_policy_id=str(candidate_policy_id),
        mode=str(cfg.mode),
        development_probe_uses_existing_val_diagnostics=bool(
            cfg.development_probe_uses_existing_val_diagnostics
        ),
        buckets=buckets,
        evidence_count=evidence_count,
        diagnosis_confidence=diagnosis_confidence,
        audit_priors=audit_priors,
    )


def apply_candidate_view(
    *,
    candidate_policy: dict[str, Any],
    sample: dict[str, Any] | None = None,
    model: Any | None = None,
    optimizer: Any | None = None,
    ema: Any | None = None,
    rng_state: Any | None = None,
) -> dict[str, Any]:
    """Create a probe-time candidate view without mutating training state."""

    sample = sample or {}
    policy = deepcopy(candidate_policy)
    image_modification = bool(policy.get("image_modification", False))
    return {
        "candidate_policy": policy,
        "sample": sample,
        "image": sample.get("image"),
        "labels": sample.get("labels"),
        "bboxes": sample.get("bboxes"),
        "image_modification_planned": image_modification,
        "image_modified": False,
        "labels_rewritten": False,
        "instances_rewritten": False,
        "model_state_updated": False,
        "optimizer_state_updated": False,
        "ema_state_updated": False,
        "rng_state_updated": False,
        "model_object_id": None if model is None else id(model),
        "optimizer_object_id": None if optimizer is None else id(optimizer),
        "ema_object_id": None if ema is None else id(ema),
        "rng_state_before": deepcopy(rng_state),
        "rng_state_after": deepcopy(rng_state),
    }


def evaluate_candidate_policy(
    *,
    candidate_policy: dict[str, Any],
    probe_set: ProbeSet | dict[str, Any] | None = None,
    benefit_metrics: dict[str, float] | None = None,
    risk_metrics: dict[str, float] | None = None,
    evidence_count: int | None = None,
    diagnosis_confidence: float | None = None,
) -> dict[str, Any]:
    probe_payload = probe_set.to_dict() if isinstance(probe_set, ProbeSet) else dict(probe_set or {})
    effective_evidence_count = int(
        evidence_count
        if evidence_count is not None
        else probe_payload.get("evidence_count", 0)
    )
    effective_diagnosis_confidence = float(
        diagnosis_confidence
        if diagnosis_confidence is not None
        else probe_payload.get("diagnosis_confidence", 0.0)
    )
    raw_risk = dict(risk_metrics or {})
    benefit = _normalise_metrics(benefit_metrics or {}, BENEFIT_WEIGHTS)
    risk = _normalise_metrics(raw_risk, RISK_WEIGHTS)
    precision_gate = _precision_gate_metrics(raw_risk)
    decision_risk = {**risk, **precision_gate}
    decision = decide_candidate_acceptance(
        candidate_policy=candidate_policy,
        benefit_metrics=benefit,
        risk_metrics=decision_risk,
        evidence_count=effective_evidence_count,
        diagnosis_confidence=effective_diagnosis_confidence,
        audit_priors=probe_payload.get("audit_priors") or [],
    )
    return {
        "candidate_policy_id": str(candidate_policy.get("policy_id", "candidate_policy_unknown")),
        "candidate_policy": deepcopy(candidate_policy),
        "probe_set": probe_payload,
        "benefit_metrics": benefit,
        "risk_metrics": decision_risk,
        "weighted_risk_metrics": risk,
        "precision_gate_metrics": precision_gate,
        "causal_score": decision["causal_score"],
        "decision": decision,
    }


def compute_causal_score(
    benefit_metrics: dict[str, float] | None = None,
    risk_metrics: dict[str, float] | None = None,
) -> float:
    benefit = _normalise_metrics(benefit_metrics or {}, BENEFIT_WEIGHTS)
    risk = _normalise_metrics(risk_metrics or {}, RISK_WEIGHTS)
    score = 0.0
    for key, weight in BENEFIT_WEIGHTS.items():
        score += weight * float(benefit.get(key, 0.0) or 0.0)
    for key, weight in RISK_WEIGHTS.items():
        score -= weight * float(risk.get(key, 0.0) or 0.0)
    return float(score)


def decide_candidate_acceptance(
    *,
    candidate_policy: dict[str, Any],
    benefit_metrics: dict[str, float] | None = None,
    risk_metrics: dict[str, float] | None = None,
    evidence_count: int = 0,
    diagnosis_confidence: float = 0.0,
    audit_priors: list[dict[str, Any]] | None = None,
    min_evidence_count: int = 5,
    min_diagnosis_confidence: float = 0.50,
    max_estimated_precision_drop: float = DEFAULT_PRECISION_GATE_THRESHOLDS["estimated_precision_drop"],
    max_non_active_fp_delta: float = DEFAULT_PRECISION_GATE_THRESHOLDS["non_active_fp_delta"],
    max_high_confidence_fp_delta: float = DEFAULT_PRECISION_GATE_THRESHOLDS["high_confidence_fp_delta"],
) -> dict[str, Any]:
    raw_risk = dict(risk_metrics or {})
    benefit = _normalise_metrics(benefit_metrics or {}, BENEFIT_WEIGHTS)
    risk = _normalise_metrics(raw_risk, RISK_WEIGHTS)
    precision_gate = _precision_gate_metrics(raw_risk)
    precision_gate_thresholds = {
        "estimated_precision_drop": float(max_estimated_precision_drop),
        "non_active_fp_delta": float(max_non_active_fp_delta),
        "high_confidence_fp_delta": float(max_high_confidence_fp_delta),
    }
    score = compute_causal_score(benefit, risk)
    policy_id = str(candidate_policy.get("policy_id", "candidate_policy_unknown"))
    action = str(candidate_policy.get("action", "roi_image_aug"))
    reasons: list[str] = []

    if action == "no_op":
        return {
            "candidate_policy_id": policy_id,
            "accepted": False,
            "decision": "no_op",
            "causal_score": score,
            "rejection_reasons": ["explicit_no_op_candidate"],
            "audit_prior_only": bool(audit_priors),
            "audit_priors": deepcopy(audit_priors or []),
            "precision_gate": precision_gate,
            "precision_gate_thresholds": precision_gate_thresholds,
            "image_modification_allowed": False,
            "sample_weighting_allowed": False,
            "strict_noop": True,
        }
    if action == "sampler_only":
        return {
            "candidate_policy_id": policy_id,
            "accepted": True,
            "decision": "sampler_only",
            "causal_score": score,
            "rejection_reasons": [],
            "audit_prior_only": bool(audit_priors),
            "audit_priors": deepcopy(audit_priors or []),
            "precision_gate": precision_gate,
            "precision_gate_thresholds": precision_gate_thresholds,
            "image_modification_allowed": False,
            "sample_weighting_allowed": True,
            "sample_weighting_effective": bool(candidate_policy.get("sample_weighting_effective", False)),
            "sample_weighting_status": str(candidate_policy.get("sample_weighting_status", "pending")),
            "strict_noop": False,
        }

    if score <= 0.0:
        reasons.append("causal_score_not_positive")
    if float(benefit.get("fn_recovery_rate", 0.0)) < 0.03 and float(
        benefit.get("localization_iou_gain", 0.0)
    ) <= 0.0:
        reasons.append("insufficient_active_class_benefit")
    if float(risk.get("fp_increase_rate", 0.0)) > 0.02:
        reasons.append("fp_increase_rate_too_high")
    if float(risk.get("high_fp_spillover_rate", 0.0)) > 0.01:
        reasons.append("high_fp_spillover_rate_too_high")
    if float(risk.get("non_active_regression_rate", 0.0)) > 0.02:
        reasons.append("non_active_regression_rate_too_high")
    if float(risk.get("ok_class_false_activation", 0.0)) != 0.0:
        reasons.append("ok_class_false_activation")
    if float(risk.get("bbox_instability_rate", 0.0)) > 0.02:
        reasons.append("bbox_instability_rate_too_high")
    if float(precision_gate.get("estimated_precision_drop", 0.0)) > float(max_estimated_precision_drop):
        reasons.append("estimated_precision_drop_too_high")
    if float(precision_gate.get("non_active_fp_delta", 0.0)) > float(max_non_active_fp_delta):
        reasons.append("non_active_fp_delta_too_high")
    if float(precision_gate.get("high_confidence_fp_delta", 0.0)) > float(max_high_confidence_fp_delta):
        reasons.append("high_confidence_fp_delta_too_high")
    if int(evidence_count) < int(min_evidence_count):
        reasons.append("insufficient_evidence_count")
    if float(diagnosis_confidence) < float(min_diagnosis_confidence):
        reasons.append("diagnosis_confidence_too_low")

    accepted = not reasons
    return {
        "candidate_policy_id": policy_id,
        "accepted": accepted,
        "decision": "accept" if accepted else "reject",
        "causal_score": score,
        "rejection_reasons": reasons,
        "audit_prior_only": bool(audit_priors),
        "audit_priors": deepcopy(audit_priors or []),
        "precision_gate": precision_gate,
        "precision_gate_thresholds": precision_gate_thresholds,
        "image_modification_allowed": bool(accepted and candidate_policy.get("image_modification", False)),
        "sample_weighting_allowed": bool(accepted and candidate_policy.get("sample_weighting", False)),
        "strict_noop": False,
    }


def select_best_candidate(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [item for item in evaluations if item.get("decision", {}).get("accepted")]
    image_candidates = [
        item for item in accepted if item.get("decision", {}).get("image_modification_allowed")
    ]
    if image_candidates:
        return max(image_candidates, key=lambda item: float(item.get("causal_score", 0.0) or 0.0))
    sampler_candidates = [
        item for item in accepted if item.get("decision", {}).get("decision") == "sampler_only"
    ]
    if sampler_candidates:
        return max(sampler_candidates, key=lambda item: float(item.get("causal_score", 0.0) or 0.0))
    noop = {
        "candidate_policy_id": "candidate_policy_0_noop",
        "candidate_policy": candidate_policy_catalog()["candidate_policy_0_noop"],
        "benefit_metrics": _normalise_metrics({}, BENEFIT_WEIGHTS),
        "risk_metrics": _normalise_metrics({}, RISK_WEIGHTS),
        "causal_score": 0.0,
        "decision": {
            "candidate_policy_id": "candidate_policy_0_noop",
            "accepted": False,
            "decision": "no_op",
            "rejection_reasons": ["no_candidate_passed_causal_probe"],
            "causal_score": 0.0,
            "image_modification_allowed": False,
            "sample_weighting_allowed": False,
            "strict_noop": True,
        },
    }
    return noop


def export_probe_report(payload: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _placeholder_samples(count: int, class_id: int, bucket: str) -> list[dict[str, Any]]:
    return [
        {"sample_id": f"{bucket}_{int(class_id)}_{idx:03d}", "class_id": int(class_id), "bucket": bucket}
        for idx in range(max(0, int(count)))
    ]


def _normalise_metrics(metrics: dict[str, float], weights: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0) or 0.0) for key in weights}


def _precision_gate_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(metrics.get(key, 0.0) or 0.0) for key in PRECISION_GATE_KEYS}
