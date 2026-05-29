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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_feedback_seed42_50ep\train\weights\best.pt`

## Metrics

- Precision: `0.7750`
- Recall: `0.7092`
- mAP50: `0.7688`
- mAP50-95: `0.5166`

## Clean Native Reference

- Baseline: `clean_native_yolo_default_seed42`
- Precision: `0.7262`
- Recall: `0.6844`
- mAP50: `0.7616`
- mAP50-95: `0.5250`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\clean_native_yolo_default_seed42_50ep\train\results.csv`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `brightness:2, clahe:2, contrast:2, cutout_safe:2, gamma:2`
- Ops downregulated: `brightness:4, clahe:4, contrast:4, cutout_safe:4, gamma:4`
- Guard-triggered epochs: `5:precision_guard/low_contrast_fn, 10:precision_guard/low_contrast_fn/recall_low, 15:precision_guard/map50_95_guard/low_contrast_fn/constraint_warning/rollback, 20:precision_guard/map50_95_guard/low_contrast_fn/constraint_warning/rollback, 25:precision_guard/map50_95_guard/low_contrast_fn/recall_low/constraint_warning, 30:precision_guard/low_contrast_fn, 35:precision_guard/low_contrast_fn/constraint_warning/rollback, 40:precision_guard/low_contrast_fn/constraint_warning/rollback, 45:precision_guard/map50_95_guard/low_contrast_fn/constraint_warning`
- Rollback triggered: `true`
- Cooldown triggered: `true`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| brightness | 103545 | 1277 | 102268 |
| clahe | 115050 | 3394 | 111656 |
| contrast | 103545 | 1252 | 102293 |
| cutout_safe | 103545 | 1252 | 102293 |
| gamma | 115050 | 3404 | 111646 |
| local_contrast | 115050 | 6894 | 108156 |
| sharpen_mild | 115050 | 9359 | 105691 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default_seed42`
- Delta Precision: `0.0487`
- Delta Recall: `0.0248`
- Delta mAP50: `0.0072`
- Delta mAP50-95: `-0.0084`
- constraint_failed: `false`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `true`
- Exceeds clean native reference on mAP50: `true`
- mAP50-95 remains within constraint: `true`
Acceptable candidate: Recall improved without violating Precision/mAP constraints.
