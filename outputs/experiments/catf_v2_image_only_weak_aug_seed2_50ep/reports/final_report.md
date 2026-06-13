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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_image_only_weak_aug_seed2_50ep\train\weights\best.pt`

## Metrics

- Precision: `0.7533`
- Recall: `0.6942`
- mAP50: `0.7727`
- mAP50-95: `0.5151`

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
- CATF-v2 rollback mode enabled: `false`
- CATF-v2 RiskGuard enabled: `false`
- CATF-v2 causal probe enabled: `true`
- Offline causal probe decisions used: `true`
- RiskGuard blocked ops: `0`
- RiskGuard sampler-only fallback: `false`
- Adaptive burn-in enabled: `false`
- Adaptive start epoch: `None`
- Adaptive candidate started: `false`
- Adaptive no-op fallback: `false`
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
- Ops upregulated: `local_contrast:2, sharpen_mild:2`
- Ops downregulated: `none`
- Guard-triggered epochs: `5:causal_probe_no_candidate_passed, 10:causal_probe_no_candidate_passed, 15:causal_probe_no_candidate_passed, 35:global_precision_guard, 40:epoch_ge_40/causal_probe_no_candidate_passed, 45:epoch_ge_40`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| local_contrast | 1952 | 80 | 1842 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `0.0570`
- Delta Recall: `-0.0344`
- Delta mAP50: `0.0035`
- Delta mAP50-95: `-0.0072`
- constraint_failed: `false`

## Verdict

- Exceeds clean native reference on Precision: `true`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `true`
- mAP50-95 remains within constraint: `true`
Not yet a strong paper main method: constraints passed but Recall did not improve.
