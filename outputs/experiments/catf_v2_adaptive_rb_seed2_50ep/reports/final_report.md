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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_adaptive_rb_seed2_50ep\train\weights\best.pt`

## Metrics

- Precision: `0.6962`
- Recall: `0.7286`
- mAP50: `0.7692`
- mAP50-95: `0.5224`

## Clean Native Reference

- Baseline: `clean_native_yolo_default`
- Precision: `0.6962`
- Recall: `0.7286`
- mAP50: `0.7692`
- mAP50-95: `0.5224`

## Feedback

- Feedback enabled: `true`
- Feedback controller: `CATF-v2`
- CATF-v2 safe mode enabled: `false`
- CATF-v2 gated mode enabled: `false`
- CATF-v2 rollback mode enabled: `true`
- Adaptive burn-in enabled: `true`
- Adaptive start epoch: `None`
- Adaptive candidate started: `false`
- Adaptive no-op fallback: `true`
- RB rollback triggered: `false`
- Safe no-op fallback triggered: `false`
- Safe controller reasons: `none`
- Gated no-op fallback triggered: `false`
- Gated controller reasons: `none`
- Reference curve loaded: `true`
- Reference curve path: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\multiseed_clean_yolo_default_vs_catf_v2\seed_2\clean_native_yolo_default\train\results.csv`
- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Policy updates: `9`
- copy_paste status: `pending_object_bank_design`
- Ops upregulated: `gamma:2, local_contrast:6, sharpen_mild:4`
- Ops downregulated: `none`
- Guard-triggered epochs: `15:adaptive_burnin_not_ready, 20:adaptive_burnin_not_ready, 25:adaptive_burnin_not_ready, 30:adaptive_burnin_not_ready, 35:adaptive_burnin_not_ready, 40:epoch_ge_40/adaptive_burnin_not_ready, 45:epoch_ge_40/adaptive_burnin_not_ready`
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
