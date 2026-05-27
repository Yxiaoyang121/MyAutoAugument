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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_inloop_feedback_clean_reference_50ep\train\weights\best.pt`

## Metrics

- Precision: `0.7536`
- Recall: `0.7124`
- mAP50: `0.7723`
- mAP50-95: `0.5153`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7262`
- Recall: `0.6844`
- mAP50: `0.7616`
- mAP50-95: `0.5250`

## Feedback

- Feedback enabled: `true`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `brightness:14, clahe:32, cutout_safe:18, gamma:32, local_contrast:33, sharpen_mild:33`
- Ops downregulated: `brightness:14, clahe:18, cutout_safe:14, gamma:18`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| clahe | 103545 | 5739 | 97806 |
| cutout_safe | 103545 | 8997 | 94548 |
| gamma | 103545 | 5737 | 97808 |
| local_contrast | 103545 | 23376 | 80169 |
| sharpen_mild | 103545 | 23277 | 80268 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `0.0274`
- Delta Recall: `0.0281`
- Delta mAP50: `0.0106`
- Delta mAP50-95: `-0.0098`
- constraint_failed: `false`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `true`
- Exceeds clean native reference on mAP50: `true`
- mAP50-95 remains within constraint: `true`
Acceptable candidate: Recall improved without violating Precision/mAP constraints.
