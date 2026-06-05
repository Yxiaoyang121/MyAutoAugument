# In-Loop YOLO Default Feedback Report

## Run Integrity

- Training success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- Epoch sequence continuous: `true`
- Epoch sequence: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]`
- YOLO default augmentation enabled: `true`
- Industrial augmentation enabled: `true`
- Industrial augmentation dynamic: `false`
- Train image count: `2301`
- Fixed augmented dataset generated: `false`
- BBox/class legal: `true`
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2_safe\seed_1\catf_v2_safe\train\weights\best.pt`

## Metrics

- Precision: `0.7725`
- Recall: `0.6477`
- mAP50: `0.7542`
- mAP50-95: `0.4799`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.7725`
- Recall: `0.6477`
- mAP50: `0.7542`
- mAP50-95: `0.4799`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF-v2`
- CATF-v2 safe mode enabled: `true`
- Safe no-op fallback triggered: `true`
- Safe controller reasons: `5:early_abstention_no_recall_or_map_gain, 10:early_abstention_no_recall_or_map_gain, 15:early_abstention_no_recall_or_map_gain, 20:early_abstention_no_recall_or_map_gain, 25:early_abstention_no_recall_or_map_gain, 30:early_abstention_no_recall_or_map_gain, 35:early_abstention_no_recall_or_map_gain, 40:early_abstention_no_recall_or_map_gain, 45:early_abstention_no_recall_or_map_gain`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_1\clean_native_yolo_default\train\results.csv`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `local_contrast:4, sharpen_mild:4`
- Ops downregulated: `none`
- Guard-triggered epochs: `5:early_abstention_no_recall_or_map_gain, 10:early_abstention_no_recall_or_map_gain, 15:early_abstention_no_recall_or_map_gain, 20:early_abstention_no_recall_or_map_gain, 25:early_abstention_no_recall_or_map_gain, 30:early_abstention_no_recall_or_map_gain, 35:early_abstention_no_recall_or_map_gain, 40:epoch_ge_40/early_abstention_no_recall_or_map_gain, 45:epoch_ge_40/early_abstention_no_recall_or_map_gain`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `0.0000`
- Delta Recall: `0.0000`
- Delta mAP50: `0.0000`
- Delta mAP50-95: `0.0000`
- constraint_failed: `false`

## Verdict

- Exceeds clean native reference on Precision: `false`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `true`
Not yet a strong paper main method: constraints passed but Recall did not improve.
