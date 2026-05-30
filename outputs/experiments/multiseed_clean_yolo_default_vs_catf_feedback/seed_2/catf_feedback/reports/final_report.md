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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_feedback\seed_2\catf_feedback\train\weights\best.pt`

## Metrics

- Precision: `0.6846`
- Recall: `0.7166`
- mAP50: `0.7608`
- mAP50-95: `0.4800`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.6962`
- Recall: `0.7286`
- mAP50: `0.7692`
- mAP50-95: `0.5224`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_inloop_feedback\seed_2\clean_native_yolo_default\train\results.csv`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `none`
- Ops downregulated: `none`
- Guard-triggered epochs: `5:precision_guard/map50_95_guard/low_contrast_fn/recall_low/constraint_warning/rollback, 10:precision_guard/map50_95_guard/low_contrast_fn/recall_low/constraint_warning/rollback, 15:precision_guard/low_contrast_fn/recall_low/constraint_warning/rollback, 20:precision_guard/map50_95_guard/low_contrast_fn/constraint_warning/rollback, 25:precision_guard/map50_95_guard/low_contrast_fn/recall_low/constraint_warning/rollback, 30:precision_guard/map50_95_guard/low_contrast_fn/recall_low/constraint_warning/rollback, 35:precision_guard/low_contrast_fn/constraint_warning/rollback, 40:precision_guard/map50_95_guard/low_contrast_fn/recall_low/constraint_warning/rollback, 45:precision_guard/map50_95_guard/low_contrast_fn/recall_low/constraint_warning/rollback`
- Rollback triggered: `true`
- Cooldown triggered: `true`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| brightness | 115050 | 3580 | 111470 |
| clahe | 115050 | 5753 | 109297 |
| contrast | 115050 | 3500 | 111550 |
| cutout_safe | 115050 | 3357 | 111693 |
| gamma | 115050 | 5671 | 109379 |
| local_contrast | 115050 | 6847 | 108203 |
| sharpen_mild | 115050 | 9141 | 105909 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `-0.0116`
- Delta Recall: `-0.0120`
- Delta mAP50: `-0.0084`
- Delta mAP50-95: `-0.0424`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `false`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `false`
Not acceptable as the paper main method under current industrial constraints.
