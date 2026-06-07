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
- Global best.pt: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_riskguard_seed2_50ep\train\weights\best.pt`

## Metrics

- Precision: `0.6394`
- Recall: `0.7231`
- mAP50: `0.7552`
- mAP50-95: `0.5073`

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
- CATF-v2 RiskGuard enabled: `true`
- RiskGuard blocked ops: `2`
- RiskGuard sampler-only fallback: `true`
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
- Ops upregulated: `local_contrast:4, sharpen_mild:4`
- Ops downregulated: `none`
- Guard-triggered epochs: `5:riskguard_class_op_block, 15:global_precision_guard, 20:global_precision_guard/global_map50_guard/global_map95_guard, 30:global_precision_guard, 35:global_precision_guard/global_map50_guard/global_map95_guard, 40:epoch_ge_40, 45:epoch_ge_40`
- Rollback triggered: `false`
- Cooldown triggered: `false`
- Freeze triggered: `true`

## Industrial Augmentation Stats

| op | seen | applied | skipped_probability |
|---|---:|---:|---:|
| local_contrast | 540 | 8 | 472 |
| sharpen_mild | 540 | 5 | 475 |

## Constraint Scoring

- Baseline: `clean_native_yolo_default`
- Delta Precision: `-0.0569`
- Delta Recall: `-0.0056`
- Delta mAP50: `-0.0140`
- Delta mAP50-95: `-0.0150`
- constraint_failed: `true`

## Verdict

- Exceeds clean native reference on Precision: `false`
- Exceeds clean native reference on Recall: `false`
- Exceeds clean native reference on mAP50: `false`
- mAP50-95 remains within constraint: `false`
Not acceptable as the paper main method under current industrial constraints.

## RiskGuard Seed2 Validation

## Verdict

- Completed 50ep: `true`
- constraint_failed: `true`
- Failure reasons: `['precision_drop_gt_0.01', 'map50_drop_gt_0.01', 'map50_95_drop_gt_0.01']`
- Seed0/seed1 sanity check: `not run` because seed2 did not pass the constraint gate.
- Interpretation: RiskGuard successfully blocks the audited class 9 texture intervention, but it is not sufficient to recover seed2 overall.

## Metrics

| group | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 |
| RiskGuard seed2 | 0.6394 | 0.7231 | 0.7552 | 0.5073 |

## Deltas

- RiskGuard vs clean: P `-0.0569`, R `-0.0056`, mAP50 `-0.0140`, mAP50-95 `-0.0150`.
- RiskGuard vs fixed CATF-v2: P `-0.1243`, R `0.0367`, mAP50 `-0.0030`, mAP50-95 `0.0107`.

## RiskGuard Events

- Epoch5 class 9 dominant issue: `texture_boundary_weak`
- Class 9 texture ops blocked: `['local_contrast', 'sharpen_mild']`
- Blocked op count: `2`
- Sampler-only fallback: `true`
- Sample weighting effective: `false`; dataloader integration remains pending.
- Event JSON: `E:\TJGY\MinPaper\MyAutoAugument\outputs\experiments\catf_v2_riskguard_seed2_50ep\reports\riskguard_events.json`

## Augmentation Stats

- Industrial samples augmented: `13` vs fixed `86`.
- ROI applied: `15` vs fixed `90`.
- ROI affected classes: `{'12': 15}`.
- Router random draw count: `960`.
- OK3 active: `false`; OK3 ROI applied: `0`.

## Class 9

- Recall clean/fixed/RiskGuard: `0.6310` / `0.4643` / `0.7960`.
- AP50 clean/fixed/RiskGuard: `0.7440` / `0.6973` / `0.7944`.
- AP50-95 clean/fixed/RiskGuard: `0.4139` / `0.3484` / `0.4198`.
- Class 9 recovered versus fixed CATF-v2 after blocking the texture ROI combination, but global constraints still failed.

## Per-Class Residual Risk

| class | name | dRecall vs clean | dAP50 vs clean | dAP50-95 vs clean |
|---:|---|---:|---:|---:|
| 3 | 开裂 | 0.0000 | 0.0000 | -0.0995 |
| 2 | 加强筋打伤 | -0.1955 | -0.0364 | -0.0818 |
| 8 | 脏污 | -0.1052 | -0.0836 | -0.0433 |
| 12 | 锡膏 | 0.0429 | -0.0072 | -0.0199 |
| 4 | 油污 | 0.0124 | -0.0242 | -0.0170 |
| 11 | 锡尖 | 0.0417 | 0.0018 | -0.0146 |
| 7 | 碰伤 | 0.0029 | -0.0119 | -0.0082 |
| 5 | 浅划伤 | 0.0000 | -0.0425 | -0.0025 |
| 10 | 锡丝残留 | 0.0000 | -0.0355 | 0.0666 |

## Conclusion

- RiskGuard should be added as a targeted CATF-v2 safety mechanism because it prevents the known class 9 + ROI texture failure path without seed-specific logic.
- This seed2 validation does not justify running seed0/seed1 sanity yet: seed2 still fails due to residual global Precision/mAP degradation and class12-only ROI activity.
- The next minimal fix should combine RiskGuard with a gate/causal probe that rejects later class-op interventions when global Precision/mAP constraints deteriorate.
