# Fixed CATF-v2 Seed2 Failure Analysis

Seed2 is the only fixed CATF-v2 seed that still fails the industrial constraint.

## Global Metrics

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| clean native seed2 | 0.6962 | 0.7286 | 0.7692 | 0.5224 |
| fixed CATF-v2 seed2 | 0.7637 | 0.6863 | 0.7582 | 0.4967 |
| delta | +0.0674 | -0.0423 | -0.0110 | -0.0257 |

## Recall Drop Classes

| class | ΔRecall | FN clean est | FN fixed est | ΔFN est |
|---|---:|---:|---:|---:|
| 5:浅划伤 | -0.2704 | 5.00 | 8.52 | +3.52 |
| 9:轮廓划伤 | -0.1667 | 31.00 | 45.00 | +14.00 |
| 2:加强筋打伤 | -0.1250 | 1.00 | 2.00 | +1.00 |
| 6:漏背锡 | -0.0745 | 14.00 | 17.43 | +3.43 |
| 12:锡膏 | -0.0044 | 15.12 | 15.32 | +0.20 |
| 0:OK2 | +0.0000 | 0.00 | 0.00 | +0.00 |

## FN Increase Classes

| class | ΔFN est | ΔRecall |
|---|---:|---:|
| 9:轮廓划伤 | +14.00 | -0.1667 |
| 5:浅划伤 | +3.52 | -0.2704 |
| 6:漏背锡 | +3.43 | -0.0745 |
| 2:加强筋打伤 | +1.00 | -0.1250 |
| 12:锡膏 | +0.20 | -0.0044 |
| 0:OK2 | +0.00 | +0.0000 |

## AP Drop Classes

| class | ΔAP50 | ΔAP50-95 |
|---|---:|---:|
| 10:锡丝残留 | -0.0748 | -0.0880 |
| 8:脏污 | -0.0559 | -0.0502 |
| 12:锡膏 | -0.0483 | -0.0417 |
| 9:轮廓划伤 | -0.0466 | -0.0654 |
| 6:漏背锡 | -0.0370 | +0.0196 |
| 7:碰伤 | -0.0243 | -0.0077 |

## AP50-95 Drop Classes

| class | ΔAP50-95 | ΔAP50 |
|---|---:|---:|
| 3:开裂 | -0.1492 | +0.0000 |
| 10:锡丝残留 | -0.0880 | -0.0748 |
| 9:轮廓划伤 | -0.0654 | -0.0466 |
| 8:脏污 | -0.0502 | -0.0559 |
| 2:加强筋打伤 | -0.0463 | +0.0045 |
| 12:锡膏 | -0.0417 | -0.0483 |

## Precision Increase / FP Reduction

| class | ΔFP est | ΔPrecision |
|---|---:|---:|
| 5:浅划伤 | -10.40 | +0.5651 |
| 9:轮廓划伤 | -9.51 | +0.0476 |
| 3:开裂 | -5.05 | +0.2665 |
| 0:OK2 | -4.41 | +0.0517 |
| 4:油污 | -3.11 | +0.0456 |
| 2:加强筋打伤 | -0.83 | +0.0958 |

## CATF-v2 Activation And ROI

- Active classes: `{'9': 1, '11': 1, '8': 1}`.
- Active details: `[{'epoch': 5, 'class_id': 9, 'action': 'propose', 'adjustment_count': 4, 'before_status': 'inactive', 'after_status': 'active', 'dominant_issue': 'texture_boundary_weak'}, {'epoch': 25, 'class_id': 11, 'action': 'propose', 'adjustment_count': 4, 'before_status': 'inactive', 'after_status': 'active', 'dominant_issue': 'weak_localization'}, {'epoch': 35, 'class_id': 8, 'action': 'propose', 'adjustment_count': 4, 'before_status': 'inactive', 'after_status': 'active', 'dominant_issue': 'weak_localization'}]`.
- ROI stats: `{'roi_aug_applied': 90, 'roi_aug_skipped_small_roi': 0, 'roi_aug_skipped_conflict': 20, 'affected_classes': {'8': 6, '9': 25, '11': 59}}`.
- Online stats summary: samples_augmented=`86`, ops=`{'local_contrast': {'seen': 2815, 'skipped_probability': 2761, 'applied': 44, 'skipped_roi_unavailable': 10}, 'sharpen_mild': {'seen': 2815, 'skipped_probability': 2763, 'applied': 42, 'skipped_roi_unavailable': 10}}`.
- Policy actions: `{'accept': 3, 'shrink': 4, 'freeze': 2}`.
- Class policy actions: `{'propose': 3, 'observe': 4}`.

## Interpretation

- Seed2 clean native is already the strongest clean run for Recall and mAP50-95, so CATF-v2 operated on a high-recall baseline.
- Fixed CATF-v2 increases Precision substantially, but lowers Recall and both mAP metrics; this is a少报但更准 pattern.
- ROI activity was concentrated on classes 9, 11, and 8, while the largest Recall/AP drops include several non-active or only-observed classes. This suggests activation did not fully align with final degradation targets.
- The controller did not rollback seed2; actions are mostly shrink/accept/freeze. This supports adding negative-effect attribution or class-level rollback in future work.
