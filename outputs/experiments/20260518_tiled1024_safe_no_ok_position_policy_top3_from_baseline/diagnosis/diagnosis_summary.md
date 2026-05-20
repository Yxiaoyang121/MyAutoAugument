# Baseline Diagnosis Summary

- Mode: single-round baseline diagnosis.
- Flow: baseline best.pt -> prediction -> diagnosis -> candidate policies -> proxy ranking -> top3.
- No top3 short-training or final training was run.
- Baseline best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\20260518_tiled1024_safe_no_ok_position_baseline_yolo11n_50ep\train\weights\best.pt`
- Dataset: `outputs/datasets/tiled/tiled_1024_ov20_full_safe_no_ok_position/data.yaml`

## Diagnosis TP/FP/FN

- TP: `668`
- FP: `172`
- FN: `215`
- Diagnosis Precision: `0.7952`
- Diagnosis Recall: `0.7381`
- Baseline val Precision/Recall: `0.690` / `0.615`

## Diagnosis Vector

- small_object_score: `0.3294` evidence=`{"tiny_gt": 177, "small_gt": 412, "small_total_gt": 589, "small_total_tp": 395, "small_recall": 0.6706281833616299}`
- low_contrast_score: `0.5065` evidence=`{"fn_low_contrast_rate": 0.5953488372093023, "fn_dark_rate": 0.4744186046511628, "fn_bright_rate": 0.009302325581395349}`
- class_imbalance_score: `0.9953` evidence=`{"gt_counts": [64, 274, 8, 2, 18, 13, 46, 269, 42, 84, 12, 27, 46], "recalls": [1.0, 0.9817518248175182, 0.875, 0.0, 0.2777777777777778, 0.5384615384615384, 0.6956521739130435, 0.6133828996282528, 0.42857142857142855, 0.5595238095238095, 0.4166666666666667, 0.8148148148148148, 0.5869565217391305], "count_imbalance": 0.9927007299270073, "recall_spread": 1.0}`
- localization_score: `0.0243` evidence=`{"gt": 905, "localization_weak": 22}`
- false_positive_score: `0.2048` evidence=`{"tp": 668, "fp": 172, "precision": 0.7952380952380952}`

## Triggered Issues

- `low_contrast_missed_defect` severity=`medium` severity_score=`0.5065` evidence=`{"issue": "low_contrast_fn_high", "severity": "medium", "observed_rate": 0.5953488372093023}`
- `low_contrast_missed_defect` severity=`medium` severity_score=`0.5065` evidence=`{"issue": "exposure_fn_high", "severity": "medium", "dark_rate": 0.4744186046511628, "bright_rate": 0.009302325581395349}`
- `class_imbalance` severity=`medium` severity_score=`0.9953` evidence=`{"issue": "class_recall_imbalance", "severity": "medium", "classes": [{"class_id": 3, "class_name": "开裂", "fn_rate": 1.0}, {"class_id": 4, "class_name": "油污", "fn_rate": 0.7222222222222222}, {"class_id": 5, "class_name": "浅划伤", "fn_rate": 0.46153846153846156}, {"class_id": 8, "class_name": "脏污", "fn_rate": 0.47619047619047616}, {"class_id": 10, "class_name": "锡丝残留", "fn_rate": 0.5833333333333334}]}`
- `localization_bias` severity=`medium` severity_score=`0.0243` evidence=`{"localization_weak_count": 22}`

## Per-Class AP/Recall

| class id | class | TP | FP | FN | diagnosis Recall | val Recall | AP50 | AP50-95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | OK2 | 64 | 6 | 0 | 1.000 | 1.000 | 0.995 | 0.841 |
| 1 | OK3 | 269 | 8 | 5 | 0.982 | 0.978 | 0.970 | 0.777 |
| 2 | 加强筋打伤 | 7 | 0 | 1 | 0.875 | 0.875 | 0.955 | 0.660 |
| 3 | 开裂 | 0 | 2 | 2 | 0.000 | 0.000 | 0.111 | 0.092 |
| 4 | 油污 | 5 | 19 | 13 | 0.278 | 0.333 | 0.242 | 0.111 |
| 5 | 浅划伤 | 7 | 3 | 6 | 0.538 | 0.550 | 0.712 | 0.338 |
| 6 | 漏背锡 | 32 | 23 | 11 | 0.696 | 0.712 | 0.694 | 0.325 |
| 7 | 碰伤 | 165 | 83 | 92 | 0.613 | 0.617 | 0.617 | 0.368 |
| 8 | 脏污 | 18 | 9 | 20 | 0.429 | 0.381 | 0.440 | 0.235 |
| 9 | 轮廓划伤 | 47 | 9 | 35 | 0.560 | 0.511 | 0.717 | 0.386 |
| 10 | 锡丝残留 | 5 | 1 | 7 | 0.417 | 0.583 | 0.619 | 0.375 |
| 11 | 锡尖 | 22 | 4 | 5 | 0.815 | 0.852 | 0.906 | 0.685 |
| 12 | 锡膏 | 27 | 5 | 18 | 0.587 | 0.609 | 0.725 | 0.450 |
