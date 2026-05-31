# CATF-v2 Seed42 50 Epoch Report

Generated: 2026-05-31T16:14:32

## Run Integrity

- Output: `outputs/experiments/catf_v2_seed42_50ep/`
- Train success: `true`
- Val success: `true`
- Single-run continuous training: `true`
- Stage restart count: `0`
- results.csv epoch range: `1..50` with `50` rows
- Train images: `2301`
- Fixed augmented dataset generated: `false`
- YOLO default augmentation enabled: `true`
- Industrial augmentation dynamic: `true`
- copy_paste: `pending_object_bank_design`
- Final best.pt: `outputs/experiments/catf_v2_seed42_50ep/train/weights/best.pt`

## Metrics

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Clean native YOLO default seed42 | 0.7262 | 0.6844 | 0.7616 | 0.5250 |
| CATF-v2 seed42 | 0.7498 | 0.7257 | 0.7679 | 0.5212 |
| Delta | +0.0236 | +0.0413 | +0.0062 | -0.0039 |

- constraint_failed: `false`
- failure_reasons: `[]`

## Per-Class Recall/AP

| class_id | class_name | Recall | AP50 | AP50-95 |
|---:|---|---:|---:|---:|
| 0 | OK2 | 1.0000 | 0.9950 | 0.8235 |
| 1 | OK3 | 0.9887 | 0.9947 | 0.8251 |
| 2 | 加强筋打伤 | 0.8750 | 0.8925 | 0.5539 |
| 3 | 开裂 | 1.0000 | 0.9950 | 0.9208 |
| 4 | 油污 | 0.4444 | 0.2700 | 0.1377 |
| 5 | 浅划伤 | 0.4615 | 0.5146 | 0.2907 |
| 6 | 漏背锡 | 0.6957 | 0.7382 | 0.3902 |
| 7 | 碰伤 | 0.5694 | 0.6814 | 0.4259 |
| 8 | 脏污 | 0.5207 | 0.5272 | 0.2983 |
| 9 | 轮廓划伤 | 0.5752 | 0.7484 | 0.4111 |
| 10 | 锡丝残留 | 0.7933 | 0.8982 | 0.5709 |
| 11 | 锡尖 | 0.9104 | 0.9750 | 0.6469 |
| 12 | 锡膏 | 0.5997 | 0.7523 | 0.4802 |

## CATF-v2 Behavior

- Feedback epochs: `[5, 10, 15, 20, 25, 30, 35, 40, 45]`
- Feedback actions: `{'observe': 2, 'accept': 1, 'shrink': 4, 'freeze': 2}`
- Class actions: `{'propose': 2, 'observe': 3}`
- Active class count max per feedback: `2`
- OK2/OK3 active epochs: `[]`
- OK2 ROI applied: `0`
- OK3 ROI applied: `0`
- ROI stats: `{'roi_aug_applied': 37, 'roi_aug_skipped_small_roi': 0, 'roi_aug_skipped_conflict': 0, 'affected_classes': {'6': 18, '12': 19}}`
- Online aug samples seen/augmented: `115050` / `31`
- Online op stats: `{'local_contrast': {'seen': 1330, 'skipped_probability': 1308, 'applied': 22}, 'sharpen_mild': {'seen': 1330, 'skipped_probability': 1321, 'applied': 9}}`

## Active Classes By Feedback Epoch

- epoch 5: []
- epoch 10: [('6', '漏背锡', 'texture_boundary_weak'), ('12', '锡膏', 'texture_boundary_weak')]
- epoch 15: []
- epoch 20: []
- epoch 25: []
- epoch 30: []
- epoch 35: []
- epoch 40: []
- epoch 45: []

## Guard / Freeze Summary

- High-FP/domain guarded counts: `{'0:OK2': 9, '1:OK3': 9, '4:油污': 9, '6:漏背锡': 7, '7:碰伤': 9, '8:脏污': 9, '9:轮廓划伤': 9, '11:锡尖': 2, '12:锡膏': 4, '3:开裂': 3, '5:浅划伤': 5, '10:锡丝残留': 6}`
- Frozen counts: `{'0:OK2': 2, '1:OK3': 2, '2:加强筋打伤': 2, '3:开裂': 2, '4:油污': 2, '5:浅划伤': 2, '6:漏背锡': 2, '7:碰伤': 2, '8:脏污': 2, '9:轮廓划伤': 2, '10:锡丝残留': 2, '11:锡尖': 2, '12:锡膏': 2}`
- Domain high-FP prior classes observed: `{'0': 'OK2', '1': 'OK3', '4': '油污', '8': '脏污'}`
- rollback count: `0`
- cooldown count: `0`
- freeze records: `4`

## Threshold Calibration

Threshold calibration is report-only and did not alter training predictions or labels.

- class `0` OK2: threshold `0.4`, reason `high_fp_raise_threshold`
- class `1` OK3: threshold `0.4`, reason `high_fp_raise_threshold`
- class `4` 油污: threshold `0.4`, reason `high_fp_raise_threshold`
- class `8` 脏污: threshold `0.4`, reason `high_fp_raise_threshold`

## Legality

- invalid_bbox_count: `0`
- class_id_oob_count: `0`
- bbox_class_valid: `true`

## Conclusion

CATF-v2 seed42 completed as a single continuous 50 epoch run. Compared with clean native YOLO default seed42, it improves Precision, Recall, and mAP50 while keeping mAP50-95 within the industrial constraint. The run is `constraint_failed=false`. OK2/OK3 stayed out of active classes and OK3 ROI applied remained `0`. ROI-aware augmentation was sparse and targeted class 6 / class 12 only. Compared with CATF-v1 seed42, CATF-v2 gives a larger Recall gain and a smaller mAP50-95 drop, with no constraint failure on seed42. It is reasonable to proceed to CATF-v2 multiseed validation; do not claim the method as final until multiseed constraints pass.
