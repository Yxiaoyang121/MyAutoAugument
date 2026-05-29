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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_inloop_feedback\seed_1\inloop_feedback\train\weights\best.pt`

## Metrics

- Precision: `0.7756`
- Recall: `0.7276`
- mAP50: `0.7763`
- mAP50-95: `0.5147`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7725`
- Recall: `0.6477`
- mAP50: `0.7542`
- mAP50-95: `0.4799`

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
| clahe | 103545 | 5741 | 97804 |
| cutout_safe | 103545 | 9042 | 94503 |
| gamma | 103545 | 5854 | 97691 |
| local_contrast | 103545 | 23244 | 80301 |
| sharpen_mild | 103545 | 23147 | 80398 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `0.0031`
- Delta Recall: `0.0799`
- Delta mAP50: `0.0221`
- Delta mAP50-95: `0.0347`
- constraint_failed: `false`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `true`
- Exceeds clean native reference on mAP50: `true`
- mAP50-95 remains within constraint: `true`
Acceptable candidate: Recall improved without violating Precision/mAP constraints.
