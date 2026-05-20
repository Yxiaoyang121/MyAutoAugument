# Top3 Proxy Policies From Baseline Diagnosis

- Basis: proxy/safety ranking only.
- No top3 short-training was run.
- These top3 policies are candidates for later short-training validation, not final selected training policy.

| rank | policy_id | source_issue | operations | contains_copy_paste | proxy_score | safety_score | combined_score | hard_filter_pass | reason |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | diag_policy_001 | low_contrast_missed_defect | clahe(p=0.411, s=0.382, params={"max_clip_limit": 3.0}), contrast(p=0.407, s=0.372, params={"max_delta": 0.35}), gamma(p=0.324, s=0.311, params={"min_gamma": 0.75, "max_gamma": 1.35}), brightness(p=0.229, s=0.241, params={"max_delta": 0.16}) | False | 0.9386 | 1.0000 | 0.9570 | True | Baseline diagnosis shows many false negatives in low-contrast or dark regions; visibility operators match that error structure. |
| 2 | diag_policy_005 | combined | copy_paste(p=0.717, s=0.696, params={"max_paste_count": 3, "max_overlap": 0.18, "prefer_small": true, "class_balanced": true, "max_attempts": 60}), clahe(p=0.459, s=0.422, params={"max_clip_limit": 3.495255474452555}), contrast(p=0.464, s=0.422, params={"max_delta": 0.3492883211678832}), gamma(p=0.447, s=0.402, params={"min_gamma": 0.8, "max_gamma": 1.3}), scale(p=0.344, s=0.251, params={"max_delta": 0.1596204379562044}), gaussian_noise(p=0.167, s=0.120, params={"max_std": 0.02992883211678832}) | True | 0.9373 | 1.0000 | 0.9561 | True | The policy blends dominant diagnosis scores, especially class imbalance and low contrast, while keeping geometry conservative. |
| 3 | diag_policy_002 | low_contrast_missed_defect | clahe(p=0.391, s=0.382, params={"max_clip_limit": 3.0}), contrast(p=0.423, s=0.372, params={"max_delta": 0.35}), gamma(p=0.321, s=0.311, params={"min_gamma": 0.75, "max_gamma": 1.35}), brightness(p=0.232, s=0.241, params={"max_delta": 0.16}) | False | 0.9337 | 1.0000 | 0.9536 | True | Baseline diagnosis shows many false negatives in low-contrast or dark regions; visibility operators match that error structure. |

## 1. diag_policy_001

1. Source diagnosis issue: `low_contrast_missed_defect`; source_issues: `low_contrast_missed_defect`; evidence: `{"issue": "low_contrast_fn_high", "severity": "medium", "observed_rate": 0.5953488372093023}`.
2. Why suitable: Baseline diagnosis shows many false negatives in low-contrast or dark regions; visibility operators match that error structure.
3. Expected improvement: Improve low-contrast and exposure-related missed defects.
4. Risk: Exposure changes are capped and later penalized by SafetyScore exposure_score.
5. Proxy/safety allowance: hard_filter_pass=True, combined=0.9570, safety=1.0000, hard_reasons=[]
6. copy_paste audit: this policy does not contain copy_paste.

## 2. diag_policy_005

1. Source diagnosis issue: `combined`; source_issues: `low_contrast_missed_defect, low_contrast_missed_defect, class_imbalance, localization_bias`; evidence: `{"source_issues": ["low_contrast_missed_defect", "low_contrast_missed_defect", "class_imbalance", "localization_bias"]}`.
2. Why suitable: The policy blends dominant diagnosis scores, especially class imbalance and low contrast, while keeping geometry conservative.
3. Expected improvement: Blend the dominant diagnosis-vector severities into one conservative policy.
4. Risk: All candidates are still reranked with proxy score and SafetyScore before short training.
5. Proxy/safety allowance: hard_filter_pass=True, combined=0.9561, safety=1.0000, hard_reasons=[]
6. copy_paste audit: new_bbox_count=`74`, new_bbox_valid_rate=`1.0000`, hard_rejected=`false`.

## 3. diag_policy_002

1. Source diagnosis issue: `low_contrast_missed_defect`; source_issues: `low_contrast_missed_defect`; evidence: `{"issue": "low_contrast_fn_high", "severity": "medium", "observed_rate": 0.5953488372093023}`.
2. Why suitable: Baseline diagnosis shows many false negatives in low-contrast or dark regions; visibility operators match that error structure.
3. Expected improvement: Improve low-contrast and exposure-related missed defects.
4. Risk: Exposure changes are capped and later penalized by SafetyScore exposure_score.
5. Proxy/safety allowance: hard_filter_pass=True, combined=0.9536, safety=1.0000, hard_reasons=[]
6. copy_paste audit: this policy does not contain copy_paste.
