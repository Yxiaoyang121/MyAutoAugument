# In-Loop YOLO Default Feedback Report

## Run Integrity

- Training success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- Epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `true`
- Industrial augmentation dynamic: `true`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- BBox/class legal: `true`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_gated\seed_2\catf_v2_gated\train\weights\best.pt`

## Metrics

- Precision: `0.7850`
- Recall: `0.6795`
- mAP50: `0.7521`
- mAP50-95: `0.5083`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.6962`
- Recall: `0.7286`
- mAP50: `0.7692`
- mAP50-95: `0.5224`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF-v2`
- CATF-v2 safe mode enabled: `false`
- CATF-v2 gated mode enabled: `true`
- Safe no-op fallback triggered: `false`
- Safe controller reasons: `none`
- Gated no-op fallback triggered: `true`
- Gated controller reasons: `10:epoch10_bad_pattern_A, 15:epoch10_bad_pattern_A, 20:epoch10_bad_pattern_A, 25:epoch10_bad_pattern_A, 30:epoch10_bad_pattern_A, 35:epoch10_bad_pattern_A, 40:epoch10_bad_pattern_A, 45:epoch10_bad_pattern_A`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_2\clean_native_yolo_default\train\results.csv`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `local_contrast:2, sharpen_mild:2`
- Ops downregulated: `none`
- Guard-triggered epochs: `10:global_map50_guard/global_map95_guard/epoch10_bad_pattern_A/bad_pattern_A/bad_pattern_B, 15:global_precision_guard/epoch10_bad_pattern_A, 20:global_precision_guard/global_map50_guard/epoch10_bad_pattern_A, 25:epoch10_bad_pattern_A, 30:epoch10_bad_pattern_A, 35:epoch10_bad_pattern_A, 40:epoch_ge_40/epoch10_bad_pattern_A, 45:epoch_ge_40/epoch10_bad_pattern_A`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| local_contrast | 770 | 12 | 758 |
| sharpen_mild | 770 | 10 | 760 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `0.0887`
- Delta Recall: `-0.0491`
- Delta mAP50: `-0.0171`
- Delta mAP50-95: `-0.0141`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `false`
Not acceptable as the paper main method under current industrial constraints.
