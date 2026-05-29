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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_inloop_feedback\seed_0\inloop_feedback\train\weights\best.pt`

## Metrics

- Precision: `0.7188`
- Recall: `0.7273`
- mAP50: `0.7687`
- mAP50-95: `0.5276`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7846`
- Recall: `0.6765`
- mAP50: `0.7347`
- mAP50-95: `0.4759`

## Feedback

- Feedback enabled: `true`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `brightness:18, clahe:36, cutout_safe:18, gamma:36, local_contrast:33, sharpen_mild:33`
- Ops downregulated: `brightness:18, clahe:18, cutout_safe:16, gamma:18`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| clahe | 103545 | 6120 | 97425 |
| cutout_safe | 103545 | 7208 | 96337 |
| gamma | 103545 | 6319 | 97226 |
| local_contrast | 103545 | 24676 | 78869 |
| sharpen_mild | 103545 | 24424 | 79121 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `-0.0658`
- Delta Recall: `0.0509`
- Delta mAP50: `0.0340`
- Delta mAP50-95: `0.0517`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `false`
- Exceeds clean native reference on Recall: `true`
- Exceeds clean native reference on mAP50: `true`
- mAP50-95 remains within constraint: `true`
Not acceptable as the paper main method under current industrial constraints.
