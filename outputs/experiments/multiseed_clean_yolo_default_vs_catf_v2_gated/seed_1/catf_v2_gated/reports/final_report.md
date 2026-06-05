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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_gated\seed_1\catf_v2_gated\train\weights\best.pt`

## Metrics

- Precision: `0.7852`
- Recall: `0.7005`
- mAP50: `0.7826`
- mAP50-95: `0.5189`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7725`
- Recall: `0.6477`
- mAP50: `0.7542`
- mAP50-95: `0.4799`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF-v2`
- CATF-v2 safe mode enabled: `false`
- CATF-v2 gated mode enabled: `true`
- Safe no-op fallback triggered: `false`
- Safe controller reasons: `none`
- Gated no-op fallback triggered: `false`
- Gated controller reasons: `none`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_1\clean_native_yolo_default\train\results.csv`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `local_contrast:4, sharpen_mild:4`
- Ops downregulated: `none`
- Guard-triggered epochs: `15:global_precision_guard/global_map50_guard, 20:global_precision_guard, 25:global_precision_guard, 30:global_precision_guard/global_map50_guard/global_map95_guard, 35:global_precision_guard, 40:epoch_ge_40, 45:epoch_ge_40`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| local_contrast | 2130 | 28 | 2102 |
| sharpen_mild | 2130 | 27 | 2103 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `0.0127`
- Delta Recall: `0.0528`
- Delta mAP50: `0.0284`
- Delta mAP50-95: `0.0390`
- constraint_failed: `false`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `true`
- Exceeds clean native reference on mAP50: `true`
- mAP50-95 remains within constraint: `true`
Acceptable candidate: Recall improved without violating Precision/mAP constraints.
