# Policy Update Report

- Mapping mode: severity_score dynamic weight formula
- Probability formula: `prob=clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0, 1)`
- Strength formula: `strength=clip(base_strength + op_weight * severity_score * 0.40, 0, 1)`

## Diagnosis Vector
- small_object_score: 0.3294
- low_contrast_score: 0.5065
- class_imbalance_score: 0.9953
- localization_score: 0.0243
- false_positive_score: 0.2048

## Candidate Policies
- diag_policy_001: source_issue=low_contrast_missed_defect severity_score=0.5065
  - ops: clahe(p=0.41, s=0.38), contrast(p=0.41, s=0.37), gamma(p=0.32, s=0.31), brightness(p=0.23, s=0.24)
  - expected_effect: Improve low-contrast and exposure-related missed defects.
  - risk_control: Exposure changes are capped and later penalized by SafetyScore exposure_score.
- diag_policy_002: source_issue=low_contrast_missed_defect severity_score=0.5065
  - ops: clahe(p=0.39, s=0.38), contrast(p=0.42, s=0.37), gamma(p=0.32, s=0.31), brightness(p=0.23, s=0.24)
  - expected_effect: Improve low-contrast and exposure-related missed defects.
  - risk_control: Exposure changes are capped and later penalized by SafetyScore exposure_score.
- diag_policy_003: source_issue=class_imbalance severity_score=0.9953
  - ops: copy_paste(p=0.65, s=0.65), scale(p=0.39, s=0.29), translate(p=0.28, s=0.22), contrast(p=0.35, s=0.32)
  - expected_effect: Increase minority-class presentation through class-balanced copy_paste.
  - risk_control: Hard class-out-of-range errors still reject; ordinary distribution drift becomes a soft safety penalty.
- diag_policy_004: source_issue=localization_bias severity_score=0.0500
  - ops: translate(p=0.14, s=0.12), scale(p=0.19, s=0.13), rotate(p=0.07, s=0.07)
  - expected_effect: Teach tolerance to small localization shifts without aggressive rotation.
  - risk_control: Rotation is low-probability and low-strength; bbox retention is checked by the proxy audit.
- diag_policy_005: source_issue=combined severity_score=0.9953
  - ops: copy_paste(p=0.72, s=0.70), clahe(p=0.46, s=0.42), contrast(p=0.46, s=0.42), gamma(p=0.45, s=0.40), scale(p=0.34, s=0.25), gaussian_noise(p=0.17, s=0.12)
  - expected_effect: Blend the dominant diagnosis-vector severities into one conservative policy.
  - risk_control: All candidates are still reranked with proxy score and SafetyScore before short training.
