# Candidate Policies From Baseline Diagnosis

- Mapping: diagnosis_vector + triggered issues -> severity-driven candidate policies.
- Candidate count: `5`
- Probability formula: `prob=clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0, 1)`
- Strength formula: `strength=clip(base_strength + op_weight * severity_score * 0.40, 0, 1)`

| policy_id | source_issue | severity_score | contains_copy_paste | operations | expected_effect | risk_control |
| --- | --- | ---: | --- | --- | --- | --- |
| diag_policy_001 | low_contrast_missed_defect | 0.5065 | False | clahe(p=0.411, s=0.382, params={"max_clip_limit": 3.0}), contrast(p=0.407, s=0.372, params={"max_delta": 0.35}), gamma(p=0.324, s=0.311, params={"min_gamma": 0.75, "max_gamma": 1.35}), brightness(p=0.229, s=0.241, params={"max_delta": 0.16}) | Improve low-contrast and exposure-related missed defects. | Exposure changes are capped and later penalized by SafetyScore exposure_score. |
| diag_policy_002 | low_contrast_missed_defect | 0.5065 | False | clahe(p=0.391, s=0.382, params={"max_clip_limit": 3.0}), contrast(p=0.423, s=0.372, params={"max_delta": 0.35}), gamma(p=0.321, s=0.311, params={"min_gamma": 0.75, "max_gamma": 1.35}), brightness(p=0.232, s=0.241, params={"max_delta": 0.16}) | Improve low-contrast and exposure-related missed defects. | Exposure changes are capped and later penalized by SafetyScore exposure_score. |
| diag_policy_003 | class_imbalance | 0.9953 | True | copy_paste(p=0.651, s=0.650, params={"max_paste_count": 3, "max_overlap": 0.2, "class_balanced": true, "max_attempts": 60}), scale(p=0.388, s=0.287, params={"max_delta": 0.12}), translate(p=0.280, s=0.219, params={"max_translate": 0.04}), contrast(p=0.350, s=0.320, params={"max_delta": 0.22}) | Increase minority-class presentation through class-balanced copy_paste. | Hard class-out-of-range errors still reject; ordinary distribution drift becomes a soft safety penalty. |
| diag_policy_004 | localization_bias | 0.0500 | False | translate(p=0.145, s=0.115, params={"max_translate": 0.055}), scale(p=0.186, s=0.132, params={"max_delta": 0.12}), rotate(p=0.066, s=0.066, params={"max_angle": 4.0}) | Teach tolerance to small localization shifts without aggressive rotation. | Rotation is low-probability and low-strength; bbox retention is checked by the proxy audit. |
| diag_policy_005 | combined | 0.9953 | True | copy_paste(p=0.717, s=0.696, params={"max_paste_count": 3, "max_overlap": 0.18, "prefer_small": true, "class_balanced": true, "max_attempts": 60}), clahe(p=0.459, s=0.422, params={"max_clip_limit": 3.495255474452555}), contrast(p=0.464, s=0.422, params={"max_delta": 0.3492883211678832}), gamma(p=0.447, s=0.402, params={"min_gamma": 0.8, "max_gamma": 1.3}), scale(p=0.344, s=0.251, params={"max_delta": 0.1596204379562044}), gaussian_noise(p=0.167, s=0.120, params={"max_std": 0.02992883211678832}) | Blend the dominant diagnosis-vector severities into one conservative policy. | All candidates are still reranked with proxy score and SafetyScore before short training. |

## Operation Details

### diag_policy_001

- source_issue: `low_contrast_missed_defect`
- source_issues: `low_contrast_missed_defect`
- severity_score: `0.5065`
- prob_formula: `prob=clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0, 1)`
- strength_formula: `strength=clip(base_strength + op_weight * severity_score * 0.40, 0, 1)`
- contains_copy_paste: `false`

| operation | prob | strength | params |
| --- | ---: | ---: | --- |
| `clahe` | 0.4111 | 0.3821 | `{"max_clip_limit": 3.0}` |
| `contrast` | 0.4071 | 0.3720 | `{"max_delta": 0.35}` |
| `gamma` | 0.3240 | 0.3114 | `{"min_gamma": 0.75, "max_gamma": 1.35}` |
| `brightness` | 0.2295 | 0.2408 | `{"max_delta": 0.16}` |

### diag_policy_002

- source_issue: `low_contrast_missed_defect`
- source_issues: `low_contrast_missed_defect`
- severity_score: `0.5065`
- prob_formula: `prob=clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0, 1)`
- strength_formula: `strength=clip(base_strength + op_weight * severity_score * 0.40, 0, 1)`
- contains_copy_paste: `false`

| operation | prob | strength | params |
| --- | ---: | ---: | --- |
| `clahe` | 0.3907 | 0.3821 | `{"max_clip_limit": 3.0}` |
| `contrast` | 0.4232 | 0.3720 | `{"max_delta": 0.35}` |
| `gamma` | 0.3211 | 0.3114 | `{"min_gamma": 0.75, "max_gamma": 1.35}` |
| `brightness` | 0.2322 | 0.2408 | `{"max_delta": 0.16}` |

### diag_policy_003

- source_issue: `class_imbalance`
- source_issues: `class_imbalance`
- severity_score: `0.9953`
- prob_formula: `prob=clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0, 1)`
- strength_formula: `strength=clip(base_strength + op_weight * severity_score * 0.40, 0, 1)`
- contains_copy_paste: `true`

| operation | prob | strength | params |
| --- | ---: | ---: | --- |
| `copy_paste` | 0.6505 | 0.6503 | `{"max_paste_count": 3, "max_overlap": 0.2, "class_balanced": true, "max_attempts": 60}` |
| `scale` | 0.3884 | 0.2872 | `{"max_delta": 0.12}` |
| `translate` | 0.2803 | 0.2194 | `{"max_translate": 0.04}` |
| `contrast` | 0.3497 | 0.3195 | `{"max_delta": 0.22}` |

### diag_policy_004

- source_issue: `localization_bias`
- source_issues: `localization_bias`
- severity_score: `0.0500`
- prob_formula: `prob=clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0, 1)`
- strength_formula: `strength=clip(base_strength + op_weight * severity_score * 0.40, 0, 1)`
- contains_copy_paste: `false`

| operation | prob | strength | params |
| --- | ---: | ---: | --- |
| `translate` | 0.1449 | 0.1150 | `{"max_translate": 0.055}` |
| `scale` | 0.1862 | 0.1320 | `{"max_delta": 0.12}` |
| `rotate` | 0.0660 | 0.0656 | `{"max_angle": 4.0}` |

### diag_policy_005

- source_issue: `combined`
- source_issues: `low_contrast_missed_defect, low_contrast_missed_defect, class_imbalance, localization_bias`
- severity_score: `0.9953`
- prob_formula: `prob=clip(base_prob + op_weight * severity_score * 0.55 + jitter, 0, 1)`
- strength_formula: `strength=clip(base_strength + op_weight * severity_score * 0.40, 0, 1)`
- contains_copy_paste: `true`

| operation | prob | strength | params |
| --- | ---: | ---: | --- |
| `copy_paste` | 0.7166 | 0.6962 | `{"max_paste_count": 3, "max_overlap": 0.18, "prefer_small": true, "class_balanced": true, "max_attempts": 60}` |
| `clahe` | 0.4589 | 0.4216 | `{"max_clip_limit": 3.495255474452555}` |
| `contrast` | 0.4642 | 0.4216 | `{"max_delta": 0.3492883211678832}` |
| `gamma` | 0.4471 | 0.4016 | `{"min_gamma": 0.8, "max_gamma": 1.3}` |
| `scale` | 0.3442 | 0.2511 | `{"max_delta": 0.1596204379562044}` |
| `gaussian_noise` | 0.1672 | 0.1196 | `{"max_std": 0.02992883211678832}` |

