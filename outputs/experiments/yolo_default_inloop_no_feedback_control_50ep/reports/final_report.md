# In-Loop YOLO Default Feedback Report

## Run Integrity

- Training success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- Epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `false`
- Industrial augmentation dynamic: `false`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- BBox/class legal: `true`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\yolo_default_inloop_no_feedback_control_50ep\train\weights\best.pt`

## Metrics

- Precision: `0.7262`
- Recall: `0.6844`
- mAP50: `0.7616`
- mAP50-95: `0.5250`

## Feedback

- Feedback enabled: `false`
- Feedback epochs: `[]`
- Policy updates: `0`
- copy_paste status: `pending_object_bank_design`

## Constraint Scoring

- Baseline: `yolo_default_reference`
- Delta Precision: `0.0130`
- Delta Recall: `-0.0756`
- Delta mAP50: `-0.0142`
- Delta mAP50-95: `0.0010`
- constraint_failed: `true`

## Verdict

This is a no-feedback control run.
