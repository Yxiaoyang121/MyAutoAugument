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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_feedback\seed_1\catf_feedback\train\weights\best.pt`

## Metrics

- Precision: `0.7376`
- Recall: `0.7098`
- mAP50: `0.7550`
- mAP50-95: `0.5185`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7725`
- Recall: `0.6477`
- mAP50: `0.7542`
- mAP50-95: `0.4799`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_inloop_feedback\seed_1\clean_native_yolo_default\train\results.csv`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `brightness:4, clahe:4, contrast:4, cutout_safe:4, gamma:4, local_contrast:2, sharpen_mild:2`
- Ops downregulated: `brightness:4, clahe:4, contrast:4, cutout_safe:4, gamma:4, local_contrast:2, sharpen_mild:2`
- Guard-triggered epochs: `5:precision_guard/low_contrast_fn/recall_low, 10:precision_guard/map50_95_guard/low_contrast_fn/recall_low/constraint_warning/rollback, 15:precision_guard/low_contrast_fn, 20:precision_guard/map50_95_guard/low_contrast_fn/constraint_warning/rollback, 25:precision_guard/low_contrast_fn/constraint_warning/rollback, 30:precision_guard/map50_95_guard/low_contrast_fn/constraint_warning/rollback, 35:precision_guard/map50_95_guard/low_contrast_fn/constraint_warning/rollback, 40:precision_guard/low_contrast_fn, 45:precision_guard/low_contrast_fn/recall_low`
- Rollback triggered: `true`
- Cooldown triggered: `true`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| brightness | 115050 | 3041 | 112009 |
| clahe | 115050 | 5307 | 109743 |
| contrast | 115050 | 3074 | 111976 |
| cutout_safe | 115050 | 2919 | 112131 |
| gamma | 115050 | 5197 | 109853 |
| local_contrast | 115050 | 6971 | 108079 |
| sharpen_mild | 115050 | 9209 | 105841 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `-0.0349`
- Delta Recall: `0.0621`
- Delta mAP50: `0.0008`
- Delta mAP50-95: `0.0386`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `false`
- Exceeds clean native reference on Recall: `true`
- Exceeds clean native reference on mAP50: `true`
- mAP50-95 remains within constraint: `true`
Not acceptable as the paper main method under current industrial constraints.
