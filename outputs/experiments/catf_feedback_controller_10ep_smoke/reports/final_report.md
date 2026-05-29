# In-Loop YOLO Default Feedback Report

## Run Integrity

- Training success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- Epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `true`
- Industrial augmentation dynamic: `true`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- BBox/class legal: `true`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_feedback_controller_10ep_smoke\train\weights\best.pt`

## Metrics

- Precision: `0.7572`
- Recall: `0.4420`
- mAP50: `0.6073`
- mAP50-95: `0.4204`

## Clean Native Reference

- Baseline: `no_feedback_control`
- Precision: `0.7262`
- Recall: `0.6844`
- mAP50: `0.7616`
- mAP50-95: `0.5250`

## Feedback

- Feedback enabled: `true`
- Feedback epochs: `[5]`
- Policy updates: `1`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `none`
- Ops downregulated: `brightness:2, clahe:2, contrast:2, cutout_safe:2, gamma:2`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| brightness | 23010 | 486 | 22524 |
| clahe | 23010 | 861 | 22149 |
| contrast | 23010 | 460 | 22550 |
| cutout_safe | 23010 | 461 | 22549 |
| gamma | 23010 | 950 | 22060 |
| local_contrast | 23010 | 1384 | 21626 |
| sharpen_mild | 23010 | 1884 | 21126 |

## Constraint Scoring

- Baseline: `no_feedback_control`
- Delta Precision: `0.0309`
- Delta Recall: `-0.2423`
- Delta mAP50: `-0.1544`
- Delta mAP50-95: `-0.1046`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `false`
Not acceptable as the paper main method under current industrial constraints.
