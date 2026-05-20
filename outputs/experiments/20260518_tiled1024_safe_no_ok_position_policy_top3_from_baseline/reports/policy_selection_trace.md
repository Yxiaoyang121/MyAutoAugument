# Policy Selection Trace

## Scope

- This is a single-round baseline diagnosis.
- Flow: baseline best.pt -> diagnosis -> candidate policies -> proxy ranking -> top3.
- This is not multi-round optimization.
- This run did not perform top3 short-training.
- Formal final strategy selection requires short-training for each top3 policy and selection by short_train_score.

## Baseline

- baseline best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep\train\weights\best.pt`
- data.yaml: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`
- metrics: Precision=0.690, Recall=0.615, mAP50=0.669, mAP50-95=0.434

## Triggered Issues

- `low_contrast_missed_defect` severity_score=`0.5065` evidence=`{"issue": "low_contrast_fn_high", "severity": "medium", "observed_rate": 0.5953488372093023}`
- `low_contrast_missed_defect` severity_score=`0.5065` evidence=`{"issue": "exposure_fn_high", "severity": "medium", "dark_rate": 0.4744186046511628, "bright_rate": 0.009302325581395349}`
- `class_imbalance` severity_score=`0.9953` evidence=`{"issue": "class_recall_imbalance", "severity": "medium", "classes": [{"class_id": 3, "class_name": "开裂", "fn_rate": 1.0}, {"class_id": 4, "class_name": "油污", "fn_rate": 0.7222222222222222}, {"class_id": 5, "class_name": "浅划伤", "fn_rate": 0.46153846153846156}, {"class_id": 8, "class_name": "脏污", "fn_rate": 0.47619047619047616}, {"class_id": 10, "class_name": "锡丝残留", "fn_rate": 0.5833333333333334}]}`
- `localization_bias` severity_score=`0.0243` evidence=`{"localization_weak_count": 22}`

## Top3

| rank | policy_id | source_issue | operations | contains_copy_paste | combined_score |
| ---: | --- | --- | --- | --- | ---: |
| 1 | diag_policy_001 | low_contrast_missed_defect | clahe(p=0.411, s=0.382, params={"max_clip_limit": 3.0}), contrast(p=0.407, s=0.372, params={"max_delta": 0.35}), gamma(p=0.324, s=0.311, params={"min_gamma": 0.75, "max_gamma": 1.35}), brightness(p=0.229, s=0.241, params={"max_delta": 0.16}) | False | 0.9570 |
| 2 | diag_policy_005 | combined | copy_paste(p=0.717, s=0.696, params={"max_paste_count": 3, "max_overlap": 0.18, "prefer_small": true, "class_balanced": true, "max_attempts": 60}), clahe(p=0.459, s=0.422, params={"max_clip_limit": 3.495255474452555}), contrast(p=0.464, s=0.422, params={"max_delta": 0.3492883211678832}), gamma(p=0.447, s=0.402, params={"min_gamma": 0.8, "max_gamma": 1.3}), scale(p=0.344, s=0.251, params={"max_delta": 0.1596204379562044}), gaussian_noise(p=0.167, s=0.120, params={"max_std": 0.02992883211678832}) | True | 0.9561 |
| 3 | diag_policy_002 | low_contrast_missed_defect | clahe(p=0.391, s=0.382, params={"max_clip_limit": 3.0}), contrast(p=0.423, s=0.372, params={"max_delta": 0.35}), gamma(p=0.321, s=0.311, params={"min_gamma": 0.75, "max_gamma": 1.35}), brightness(p=0.232, s=0.241, params={"max_delta": 0.16}) | False | 0.9536 |

## Artifacts

- diagnosis_json: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/diagnosis/diagnosis.json`
- diagnosis_summary_md: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/diagnosis/diagnosis_summary.md`
- candidate_policies_json: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/policies/candidate_policies.json`
- candidate_policies_md: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/policies/candidate_policies.md`
- proxy_ranking_json: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_ranking.json`
- proxy_ranking_md: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_ranking.md`
- proxy_safety_report_md: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/proxy/proxy_safety_report.md`
- top3_policies_json: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/top3_policies/top3_policies.json`
- top3_policies_md: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/top3_policies/top3_policies.md`
- policy_selection_trace_json: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/reports/policy_selection_trace.json`
- policy_selection_trace_md: `outputs/experiments/20260518_tiled1024_safe_no_ok_position_policy_top3_from_baseline/reports/policy_selection_trace.md`
