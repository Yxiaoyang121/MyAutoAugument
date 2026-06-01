# In-Loop YOLO Default Feedback Report

## Run Integrity

- Training success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- Epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `false`
- Industrial augmentation dynamic: `false`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- BBox/class legal: `true`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\diagnosis_only_inloop_control_10ep_smoke\train\weights\best.pt`

## Metrics

- Precision: `0.6952`
- Recall: `0.5176`
- mAP50: `0.5749`
- mAP50-95: `0.3701`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7262`
- Recall: `0.6844`
- mAP50: `0.7616`
- mAP50-95: `0.5250`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF-v2`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\clean_native_yolo_default_seed42_50ep\train\results.csv`
- Feedback epochs: `[5]`
- Policy updates: `1`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `none`
- Ops downregulated: `none`
- Guard-triggered epochs: `none`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `false`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `-0.0311`
- Delta Recall: `-0.1668`
- Delta mAP50: `-0.1867`
- Delta mAP50-95: `-0.1549`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `false`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `false`
Not acceptable as the paper main method under current industrial constraints.
