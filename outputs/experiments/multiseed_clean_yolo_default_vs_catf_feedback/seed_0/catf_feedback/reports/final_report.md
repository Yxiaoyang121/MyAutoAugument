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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_feedback\seed_0\catf_feedback\train\weights\best.pt`

## Metrics

- Precision: `0.7574`
- Recall: `0.6751`
- mAP50: `0.7526`
- mAP50-95: `0.5204`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7846`
- Recall: `0.6765`
- mAP50: `0.7347`
- mAP50-95: `0.4759`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_inloop_feedback\seed_0\clean_native_yolo_default\train\results.csv`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `brightness:3, clahe:3, contrast:3, cutout_safe:3, gamma:3, local_contrast:4, sharpen_mild:4`
- Ops downregulated: `brightness:8, clahe:9, contrast:8, cutout_safe:8, gamma:9`
- Guard-triggered epochs: `5:precision_guard/low_contrast_fn/recall_low/constraint_warning/rollback, 10:precision_guard/low_contrast_fn, 15:precision_guard/low_contrast_fn/recall_low, 20:precision_guard/low_contrast_fn/constraint_warning/rollback, 25:precision_guard/low_contrast_fn/recall_low, 30:precision_guard/low_contrast_fn, 35:precision_guard/low_contrast_fn, 40:precision_guard/map50_95_guard/low_contrast_fn/constraint_warning/rollback, 45:precision_guard/low_contrast_fn/recall_low`
- Rollback triggered: `true`
- Cooldown triggered: `true`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| brightness | 46020 | 917 | 45103 |
| clahe | 69030 | 1980 | 67050 |
| contrast | 46020 | 909 | 45111 |
| cutout_safe | 46020 | 865 | 45155 |
| gamma | 69030 | 2025 | 67005 |
| local_contrast | 115050 | 7586 | 107464 |
| sharpen_mild | 115050 | 10016 | 105034 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `-0.0272`
- Delta Recall: `-0.0014`
- Delta mAP50: `0.0179`
- Delta mAP50-95: `0.0445`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `false`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `true`
- mAP50-95 remains within constraint: `true`
Not acceptable as the paper main method under current industrial constraints.
