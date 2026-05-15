from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from AutoAugment.diagnostic_pipeline.common import write_json


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
    issues = list(diagnosis.get("issues", []))
    if not issues:
        issues = [{"type": "stable_validation_keep_light_policy", "severity": "low", "evidence": {}}]

    policies: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for issue in issues:
        policy = _policy_for_issue(issue, len(policies) + 1, rng)
        signature = _policy_signature(policy)
        if signature not in seen_signatures:
            policies.append(policy)
            seen_signatures.add(signature)
        if len(policies) >= max_policies:
            break

    combined = _combined_policy(issues, len(policies) + 1)
    signature = _policy_signature(combined)
    if len(policies) < max_policies and signature not in seen_signatures:
        policies.append(combined)

    payload = {
        "stage": "diagnosis_to_policy_mapping",
        "status": "completed",
        "dry_run": bool(diagnosis.get("dry_run", False)),
        "seed": int(seed),
        "policy_count": len(policies),
        "policies": policies,
        "mapping_rules": {
            "small_object_low_recall": ["scale", "crop", "sharpen", "clahe", "translate"],
            "low_contrast_missed_defect": ["contrast", "gamma", "clahe", "brightness", "sharpen"],
            "localization_bias": ["mild translate", "mild scale", "reduced rotate/affine strength"],
            "high_false_positive": ["avoid strong noise/blur", "mild contrast/gamma"],
            "class_imbalance": ["targeted class metadata", "crop", "scale", "visibility-preserving color ops"],
            "background_interference": ["mild brightness/gamma", "light gaussian_noise"],
        },
    }
    write_json(output / "candidate_policies.json", payload)
    return payload


def _policy_for_issue(issue: dict[str, Any], index: int, rng: np.random.Generator) -> dict[str, Any]:
    issue_type = str(issue.get("type", "stable_validation_keep_light_policy"))
    if issue_type == "small_object_low_recall":
        operations = [
            _op("scale", 0.65, 0.28, {"max_delta": 0.16}),
            _op("sharpen", 0.55, 0.28, {"amount": 0.8}),
            _op("clahe", 0.50, 0.35, {"max_clip_limit": 3.0}),
            _op("translate", 0.35, 0.18, {"max_translate": 0.05}),
        ]
        explanation = "Triggered by small-object recall loss; uses visibility and mild geometry while avoiding destructive blur."
    elif issue_type == "low_contrast_missed_defect":
        operations = [
            _op("contrast", 0.75, 0.45, {"max_delta": 0.35}),
            _op("gamma", 0.60, 0.40, {"min_gamma": 0.75, "max_gamma": 1.35}),
            _op("clahe", 0.55, 0.42, {"max_clip_limit": 3.0}),
            _op("brightness", 0.45, 0.30, {"max_delta": 0.16}),
        ]
        explanation = "Triggered by low-contrast or exposure-related misses; emphasizes contrast, gamma, CLAHE, and mild brightness."
    elif issue_type == "localization_bias":
        operations = [
            _op("translate", 0.55, 0.22, {"max_translate": 0.055}),
            _op("scale", 0.55, 0.22, {"max_delta": 0.12}),
            _op("rotate", 0.25, 0.12, {"max_angle": 4.0}),
        ]
        explanation = "Triggered by localization weakness; keeps geometry mild and limits rotation strength."
    elif issue_type == "high_false_positive":
        operations = [
            _op("contrast", 0.45, 0.25, {"max_delta": 0.22}),
            _op("gamma", 0.35, 0.22, {"min_gamma": 0.85, "max_gamma": 1.20}),
            _op("gaussian_blur", 0.12, 0.08, {"max_kernel": 3}),
        ]
        explanation = "Triggered by false positives; avoids strong noise, strong blur, and aggressive illumination shifts."
    elif issue_type == "class_imbalance":
        operations = [
            _op("crop", 0.45, 0.22, {"max_crop_fraction": 0.18, "min_visibility": 0.45}),
            _op("scale", 0.55, 0.20, {"max_delta": 0.12}),
            _op("contrast", 0.45, 0.25, {"max_delta": 0.25}),
        ]
        explanation = "Triggered by class imbalance; records targeted-class metadata and uses conservative object-preserving transforms."
    elif issue_type == "background_interference":
        operations = [
            _op("brightness", 0.40, 0.25, {"max_delta": 0.14}),
            _op("gamma", 0.40, 0.25, {"min_gamma": 0.85, "max_gamma": 1.20}),
            _op("gaussian_noise", 0.18, 0.10, {"max_std": 0.025}),
        ]
        explanation = "Triggered by background interference; adds mild illumination and texture variation without strong corruption."
    else:
        operations = [
            _op("brightness", 0.35, 0.20, {"max_delta": 0.12}),
            _op("contrast", 0.40, 0.22, {"max_delta": 0.22}),
            _op("scale", 0.35, 0.16, {"max_delta": 0.10}),
            _op("horizontal_flip", 0.35, 1.0, {}),
        ]
        explanation = "Fallback conservative policy for stable validation or dry-run planning."

    jitter = float(rng.uniform(-0.02, 0.02))
    for operation in operations:
        operation["prob"] = float(np.clip(operation["prob"] + jitter, 0.0, 1.0))
    policy_id = f"diag_policy_{index:03d}"
    return {
        "policy_id": policy_id,
        "name": policy_id,
        "source_issues": [issue_type],
        "operations": operations,
        "explanation": explanation,
        "metadata": {
            "source_issues": [issue_type],
            "diagnostic_evidence": issue.get("evidence", {}),
            "severity": issue.get("severity", "medium"),
        },
    }


def _combined_policy(issues: list[dict[str, Any]], index: int) -> dict[str, Any]:
    issue_types = [str(issue.get("type", "")) for issue in issues]
    operations = [
        _op("contrast", 0.55, 0.32, {"max_delta": 0.28}),
        _op("clahe", 0.45, 0.32, {"max_clip_limit": 2.8}),
        _op("scale", 0.45, 0.18, {"max_delta": 0.10}),
        _op("translate", 0.35, 0.16, {"max_translate": 0.04}),
    ]
    if "high_false_positive" in issue_types:
        operations = [operation for operation in operations if operation["name"] not in {"gaussian_noise", "gaussian_blur"}]
    policy_id = f"diag_policy_{index:03d}"
    return {
        "policy_id": policy_id,
        "name": policy_id,
        "source_issues": issue_types,
        "operations": operations,
        "explanation": "Combined conservative policy generated from all dominant validation error issues.",
        "metadata": {"source_issues": issue_types, "combined": True},
    }


def _op(name: str, prob: float, strength: float, params: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "params": dict(params), "prob": float(prob), "strength": float(strength)}


def _policy_signature(policy: dict[str, Any]) -> str:
    return "|".join(f"{op['name']}:{op['prob']:.2f}:{op['strength']:.2f}" for op in policy.get("operations", []))
