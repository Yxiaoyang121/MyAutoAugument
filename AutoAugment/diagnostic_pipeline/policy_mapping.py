from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.augmentations import list_augmentations
from AutoAugment.diagnostic_pipeline.common import write_json, write_markdown


VECTOR_BY_ISSUE = {
    "small_object_low_recall": "small_object_score",
    "low_contrast_missed_defect": "low_contrast_score",
    "localization_bias": "localization_score",
    "high_false_positive": "false_positive_score",
    "class_imbalance": "class_imbalance_score",
    "background_interference": "low_contrast_score",
    "stable_validation_keep_light_policy": None,
}

ISSUE_PROFILES = {
    "small_object_low_recall": {
        "operations": [
            ("copy_paste", 0.95, {"max_paste_count": 2, "max_overlap": 0.15, "prefer_small": True}),
            ("scale", 0.55, {"max_delta": 0.16}),
            ("translate", 0.35, {"max_translate": 0.05}),
            ("clahe", 0.35, {"max_clip_limit": 3.0}),
            ("contrast", 0.25, {"max_delta": 0.25}),
        ],
        "expected_effect": "Increase small-defect exposure while keeping geometry mild.",
        "risk_control": "copy_paste uses low overlap; geometry strengths stay conservative to protect bbox visibility.",
    },
    "low_contrast_missed_defect": {
        "operations": [
            ("clahe", 0.80, {"max_clip_limit": 3.0}),
            ("contrast", 0.75, {"max_delta": 0.35}),
            ("gamma", 0.55, {"min_gamma": 0.75, "max_gamma": 1.35}),
            ("brightness", 0.30, {"max_delta": 0.16}),
        ],
        "expected_effect": "Improve low-contrast and exposure-related missed defects.",
        "risk_control": "Exposure changes are capped and later penalized by SafetyScore exposure_score.",
    },
    "localization_bias": {
        "operations": [
            ("translate", 0.75, {"max_translate": 0.055}),
            ("scale", 0.60, {"max_delta": 0.12}),
            ("rotate", 0.28, {"max_angle": 4.0}),
        ],
        "expected_effect": "Teach tolerance to small localization shifts without aggressive rotation.",
        "risk_control": "Rotation is low-probability and low-strength; bbox retention is checked by the proxy audit.",
    },
    "high_false_positive": {
        "operations": [
            ("contrast", 0.35, {"max_delta": 0.22}),
            ("gamma", 0.32, {"min_gamma": 0.85, "max_gamma": 1.20}),
            ("gaussian_blur", 0.12, {"max_kernel": 3}),
        ],
        "expected_effect": "Reduce brittle background activations with mild visual smoothing.",
        "risk_control": "Noise is avoided for high-FP diagnoses; blur probability remains low.",
    },
    "class_imbalance": {
        "operations": [
            ("copy_paste", 0.88, {"max_paste_count": 2, "max_overlap": 0.20, "class_balanced": True}),
            ("scale", 0.42, {"max_delta": 0.12}),
            ("translate", 0.30, {"max_translate": 0.04}),
            ("contrast", 0.25, {"max_delta": 0.22}),
        ],
        "expected_effect": "Increase minority-class presentation through class-balanced copy_paste.",
        "risk_control": "Hard class-out-of-range errors still reject; ordinary distribution drift becomes a soft safety penalty.",
    },
    "background_interference": {
        "operations": [
            ("brightness", 0.35, {"max_delta": 0.14}),
            ("gamma", 0.35, {"min_gamma": 0.85, "max_gamma": 1.20}),
            ("gaussian_noise", 0.18, {"max_std": 0.025}),
        ],
        "expected_effect": "Add mild illumination and texture diversity for background-sensitive misses.",
        "risk_control": "Noise strength is capped and will be down-ranked by strength and exposure penalties.",
    },
    "stable_validation_keep_light_policy": {
        "operations": [
            ("brightness", 0.18, {"max_delta": 0.12}),
            ("contrast", 0.20, {"max_delta": 0.22}),
            ("scale", 0.18, {"max_delta": 0.10}),
            ("horizontal_flip", 0.22, {}),
        ],
        "expected_effect": "Keep a light fallback policy for dry-run or stable validation cases.",
        "risk_control": "Low probabilities and low strengths minimize destructive augmentation.",
    },
}

OP_BASE = {
    "copy_paste": (0.18, 0.30),
    "clahe": (0.18, 0.22),
    "contrast": (0.20, 0.22),
    "gamma": (0.16, 0.20),
    "brightness": (0.14, 0.18),
    "translate": (0.12, 0.10),
    "scale": (0.16, 0.12),
    "rotate": (0.06, 0.06),
    "gaussian_noise": (0.05, 0.04),
    "gaussian_blur": (0.05, 0.04),
    "horizontal_flip": (0.18, 1.0),
}

PROB_FORMULA = "prob=clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0, 1)"
STRENGTH_FORMULA = "strength=clip(base_strength + op_weight * severity_score * 0.40, 0, 1)"


def generate_candidate_policies(
    diagnosis: dict[str, Any],
    *,
    output_dir: str | Path,
    seed: int = 42,
    max_policies: int = 8,
) -> dict[str, Any]:
    """Generate augmentation policies from validation error diagnosis."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    vector = _flatten_diagnosis_vector(diagnosis.get("diagnosis_vector", {}))
    issues = list(diagnosis.get("issues", []))
    if not issues:
        issues = [{"type": "stable_validation_keep_light_policy", "severity": "low", "evidence": {}}]

    policies: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for issue in issues:
        policy = _policy_for_issue(issue, len(policies) + 1, rng, vector)
        signature = _policy_signature(policy)
        if signature not in seen_signatures:
            policies.append(policy)
            seen_signatures.add(signature)
        if len(policies) >= max_policies:
            break

    combined = _combined_policy(issues, len(policies) + 1, vector, rng)
    signature = _policy_signature(combined)
    if len(policies) < max_policies and signature not in seen_signatures:
        policies.append(combined)

    runtime_validation = _validate_policy_runtime(policies)
    payload = {
        "stage": "diagnosis_to_policy_mapping",
        "status": "completed",
        "dry_run": bool(diagnosis.get("dry_run", False)),
        "seed": int(seed),
        "policy_count": len(policies),
        "diagnosis_vector": vector,
        "policies": policies,
        "runtime_validation": runtime_validation,
        "mapping_rules": {
            "mode": "severity_score_dynamic_weight_formula",
            "prob_formula": PROB_FORMULA,
            "strength_formula": STRENGTH_FORMULA,
            "operators": ["copy_paste", "clahe", "contrast", "gamma", "translate", "scale", "rotate", "gaussian_noise"],
        },
    }
    write_json(output / "candidate_policies.json", payload)
    write_policy_update_report(output / "policy_update_report.md", payload)
    return payload


def _policy_for_issue(
    issue: dict[str, Any],
    index: int,
    rng: np.random.Generator,
    vector: dict[str, float],
) -> dict[str, Any]:
    issue_type = str(issue.get("type", "stable_validation_keep_light_policy"))
    profile = ISSUE_PROFILES.get(issue_type, ISSUE_PROFILES["stable_validation_keep_light_policy"])
    severity_score = _severity_score(issue_type, issue, vector)
    operations = [
        _dynamic_op(name, op_weight, severity_score, params, rng)
        for name, op_weight, params in profile["operations"]
    ]
    policy_id = f"diag_policy_{index:03d}"
    return {
        "policy_id": policy_id,
        "name": policy_id,
        "source_issue": issue_type,
        "source_issues": [issue_type],
        "severity_score": severity_score,
        "prob_formula": PROB_FORMULA,
        "strength_formula": STRENGTH_FORMULA,
        "expected_effect": profile["expected_effect"],
        "risk_control": profile["risk_control"],
        "operations": operations,
        "explanation": f"Dynamic severity-weighted policy for {issue_type}.",
        "metadata": {
            "source_issues": [issue_type],
            "diagnostic_evidence": issue.get("evidence", {}),
            "severity": issue.get("severity", "medium"),
            "severity_score": severity_score,
        },
    }


def _combined_policy(
    issues: list[dict[str, Any]],
    index: int,
    vector: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, Any]:
    issue_types = [str(issue.get("type", "stable_validation_keep_light_policy")) for issue in issues]
    operator_weights = {
        "copy_paste": max(vector.get("small_object_score", 0.0), vector.get("class_imbalance_score", 0.0)),
        "clahe": vector.get("low_contrast_score", 0.0),
        "contrast": max(vector.get("low_contrast_score", 0.0), 0.35 * vector.get("false_positive_score", 0.0)),
        "gamma": max(vector.get("low_contrast_score", 0.0), 0.25 * vector.get("false_positive_score", 0.0)),
        "translate": vector.get("localization_score", 0.0),
        "scale": max(vector.get("small_object_score", 0.0), vector.get("localization_score", 0.0)),
        "rotate": 0.50 * vector.get("localization_score", 0.0),
        "gaussian_noise": max(0.0, vector.get("low_contrast_score", 0.0) - vector.get("false_positive_score", 0.0)) * 0.25,
    }
    severity_score = float(np.clip(max(operator_weights.values()) if operator_weights else 0.15, 0.10, 1.0))
    operations = []
    for name, issue_weight in operator_weights.items():
        if issue_weight <= 0.04 and name not in {"contrast", "scale"}:
            continue
        params = _default_params(name, severity_score)
        operations.append(_dynamic_op(name, max(0.20, issue_weight), severity_score, params, rng))
    if not operations:
        operations = [
            _dynamic_op("contrast", 0.25, 0.15, {"max_delta": 0.22}, rng),
            _dynamic_op("scale", 0.20, 0.15, {"max_delta": 0.10}, rng),
        ]
    policy_id = f"diag_policy_{index:03d}"
    return {
        "policy_id": policy_id,
        "name": policy_id,
        "source_issue": "combined",
        "source_issues": issue_types,
        "severity_score": severity_score,
        "prob_formula": PROB_FORMULA,
        "strength_formula": STRENGTH_FORMULA,
        "expected_effect": "Blend the dominant diagnosis-vector severities into one conservative policy.",
        "risk_control": "All candidates are still reranked with proxy score and SafetyScore before short training.",
        "operations": operations,
        "explanation": "Combined dynamic policy generated from diagnosis_vector operator weights.",
        "metadata": {"source_issues": issue_types, "combined": True, "severity_score": severity_score},
    }


def _dynamic_op(
    name: str,
    op_weight: float,
    severity_score: float,
    params: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    base_prob, base_strength = OP_BASE[name]
    jitter = float(rng.uniform(-0.015, 0.015))
    prob = float(np.clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0.0, 1.0))
    strength = float(np.clip(base_strength + op_weight * severity_score * 0.40, 0.0, 1.0))
    dynamic_params = dict(params)
    if name == "copy_paste":
        dynamic_params["max_paste_count"] = max(1, int(round(1 + severity_score * dynamic_params.get("max_paste_count", 2))))
        dynamic_params.setdefault("max_attempts", 60)
    return {"name": name, "params": dynamic_params, "prob": prob, "strength": strength}


def _default_params(name: str, severity_score: float) -> dict[str, Any]:
    if name == "copy_paste":
        return {"max_paste_count": 2, "max_overlap": 0.18, "prefer_small": True, "class_balanced": True}
    if name == "clahe":
        return {"max_clip_limit": 2.5 + severity_score}
    if name == "contrast":
        return {"max_delta": 0.20 + 0.15 * severity_score}
    if name == "gamma":
        return {"min_gamma": 0.80, "max_gamma": 1.30}
    if name == "translate":
        return {"max_translate": 0.035 + 0.025 * severity_score}
    if name == "scale":
        return {"max_delta": 0.08 + 0.08 * severity_score}
    if name == "rotate":
        return {"max_angle": 2.0 + 3.0 * severity_score}
    if name == "gaussian_noise":
        return {"max_std": 0.015 + 0.015 * severity_score}
    return {}


def _severity_score(issue_type: str, issue: dict[str, Any], vector: dict[str, float]) -> float:
    vector_key = VECTOR_BY_ISSUE.get(issue_type)
    if vector_key is not None:
        value = vector.get(vector_key)
        if value is not None:
            return float(np.clip(value, 0.05, 1.0))
    severity = str(issue.get("severity", "medium")).lower()
    return {"low": 0.20, "medium": 0.50, "high": 0.80, "critical": 1.0}.get(severity, 0.35)


def _flatten_diagnosis_vector(vector: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in [
        "small_object_score",
        "low_contrast_score",
        "class_imbalance_score",
        "localization_score",
        "false_positive_score",
    ]:
        item = vector.get(key, 0.0)
        if isinstance(item, dict):
            item = item.get("score", 0.0)
        out[key] = float(np.clip(float(item or 0.0), 0.0, 1.0))
    return out


def _policy_signature(policy: dict[str, Any]) -> str:
    return "|".join(f"{op['name']}:{op['prob']:.2f}:{op['strength']:.2f}" for op in policy.get("operations", []))


def _validate_policy_runtime(policies: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate that generated policies only reference executable augmentations."""

    registered = set(list_augmentations())
    missing: dict[str, list[str]] = {}
    for policy in policies:
        policy_id = str(policy.get("policy_id", policy.get("name", "<unknown>")))
        missing_ops = sorted({str(op.get("name")) for op in policy.get("operations", []) if str(op.get("name")) not in registered})
        if missing_ops:
            missing[policy_id] = missing_ops
    if missing:
        raise RuntimeError(
            "Generated diagnosis-driven policies reference unavailable augmentation ops: "
            + "; ".join(f"{policy_id}={ops}" for policy_id, ops in missing.items())
        )
    return {
        "status": "passed",
        "registered_augmentations": sorted(registered),
        "policy_count": len(policies),
    }


def write_policy_update_report(path: str | Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Policy Update Report",
        "",
        "- Mapping mode: severity_score dynamic weight formula",
        f"- Probability formula: `{PROB_FORMULA}`",
        f"- Strength formula: `{STRENGTH_FORMULA}`",
        "",
        "## Diagnosis Vector",
    ]
    for key, value in payload.get("diagnosis_vector", {}).items():
        lines.append(f"- {key}: {float(value):.4f}")
    lines.extend(["", "## Candidate Policies"])
    for policy in payload.get("policies", []):
        ops = ", ".join(
            f"{op['name']}(p={float(op['prob']):.2f}, s={float(op['strength']):.2f})"
            for op in policy.get("operations", [])
        )
        lines.extend(
            [
                f"- {policy.get('policy_id')}: source_issue={policy.get('source_issue')} severity_score={float(policy.get('severity_score', 0.0)):.4f}",
                f"  - ops: {ops}",
                f"  - expected_effect: {policy.get('expected_effect')}",
                f"  - risk_control: {policy.get('risk_control')}",
            ]
        )
    write_markdown(path, lines)
