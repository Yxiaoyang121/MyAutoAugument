# Fixed CATF-v2 Seed0 Failure Analysis

Seed0 is not an official fixed-training failure: fixed CATF-v2 seed0 already passes the original industrial constraints. The remaining issue is that the earlier post-hoc threshold search still failed mAP50-95 in the custom evaluator.

## Overall Metrics

| Run | Precision | Recall | mAP50 | mAP50-95 | constraint_failed |
|---|---:|---:|---:|---:|:---:|
| clean native seed0 | 0.7846 | 0.6765 | 0.7347 | 0.4759 | false |
| fixed CATF-v2 seed0 | 0.7785 | 0.6697 | 0.7437 | 0.4895 | false |
| delta | -0.0060 | -0.0068 | +0.0090 | +0.0136 | - |

## Constraint Trigger

- Official fixed seed0 failure reasons: `[]`.
- Post-hoc default threshold evaluator failure reasons: `['map50_drop_gt_0.01', 'map50_95_drop_gt_0.01']`.
- Selected RC calibration failure reasons: `[]`.

## Recall Drop Classes

| class | Delta Recall | FN clean est | FN fixed est | Delta FN est |
|---|---:|---:|---:|---:|
| 2:加强筋打伤 | -0.2500 | 1.00 | 3.00 | +2.00 |
| 12:锡膏 | -0.1130 | 15.26 | 20.46 | +5.20 |
| 9:轮廓划伤 | -0.1022 | 36.00 | 44.59 | +8.59 |
| 7:碰伤 | -0.0470 | 116.98 | 129.63 | +12.65 |
| 8:脏污 | -0.0238 | 21.00 | 22.00 | +1.00 |
| 0:OK2 | +0.0000 | 1.00 | 1.00 | +0.00 |
| 3:开裂 | +0.0000 | 0.00 | 0.00 | +0.00 |
| 5:浅划伤 | +0.0000 | 7.00 | 7.00 | +0.00 |

## AP50-95 Drop Classes

| class | Delta AP50-95 | Delta AP50 |
|---|---:|---:|
| 2:加强筋打伤 | -0.1133 | -0.0615 |
| 1:OK3 | -0.0031 | +0.0001 |
| 9:轮廓划伤 | -0.0013 | -0.0237 |
| 12:锡膏 | +0.0137 | -0.0033 |
| 8:脏污 | +0.0138 | +0.0274 |
| 11:锡尖 | +0.0152 | +0.0171 |
| 7:碰伤 | +0.0172 | -0.0004 |
| 5:浅划伤 | +0.0233 | -0.0263 |

## AP50 Drop Classes

| class | Delta AP50 | Delta AP50-95 |
|---|---:|---:|
| 2:加强筋打伤 | -0.0615 | -0.1133 |
| 5:浅划伤 | -0.0263 | +0.0233 |
| 9:轮廓划伤 | -0.0237 | -0.0013 |
| 12:锡膏 | -0.0033 | +0.0137 |
| 7:碰伤 | -0.0004 | +0.0172 |
| 3:开裂 | +0.0000 | +0.0663 |
| 0:OK2 | +0.0000 | +0.0350 |
| 1:OK3 | +0.0001 | -0.0031 |

## Failure Type

- Official metrics: seed0 is mostly a mild Recall tradeoff, not a hard constraint failure.
- Post-hoc custom evaluator: the default threshold path has mAP50/mAP50-95 deficits, and the earlier constrained search fixed mAP50 but not mAP50-95.
- The failure is therefore closer to `confidence threshold mismatch + AP50-95 guard weakness` than a clear training collapse.
- The strongest rollback candidates from official per-class AP50-95 drops are `加强筋打伤`, `碰伤`, `OK3`, and `轮廓划伤`; however only `加强筋打伤` is a strong low-support rollback candidate.

## RC Threshold Calibration

- Selected objective: `catf_v2_rc_threshold_score`.
- Metrics after selected calibration: P/R/mAP50/mAP50-95 = `0.6636/0.7915/0.6737/0.4303`.
- Delta vs clean default evaluator: `+0.0397/+0.0269/+0.0038/-0.0099`.
- Constraint failed after RC calibration: `false`.
- Threshold raises: `[{'class_id': 0, 'old_threshold': 0.25, 'new_threshold': 0.7, 'class_name': 'OK2'}, {'class_id': 2, 'old_threshold': 0.25, 'new_threshold': 0.35, 'class_name': '加强筋打伤'}, {'class_id': 3, 'old_threshold': 0.25, 'new_threshold': 0.5, 'class_name': '开裂'}, {'class_id': 4, 'old_threshold': 0.25, 'new_threshold': 0.7, 'class_name': '油污'}, {'class_id': 10, 'old_threshold': 0.25, 'new_threshold': 0.7, 'class_name': '锡丝残留'}]`.
- Threshold lowers: `[{'class_id': 1, 'old_threshold': 0.25, 'new_threshold': 0.1, 'class_name': 'OK3'}, {'class_id': 5, 'old_threshold': 0.25, 'new_threshold': 0.15, 'class_name': '浅划伤'}, {'class_id': 6, 'old_threshold': 0.25, 'new_threshold': 0.1, 'class_name': '漏背锡'}, {'class_id': 7, 'old_threshold': 0.25, 'new_threshold': 0.1, 'class_name': '碰伤'}, {'class_id': 9, 'old_threshold': 0.25, 'new_threshold': 0.15, 'class_name': '轮廓划伤'}, {'class_id': 12, 'old_threshold': 0.25, 'new_threshold': 0.1, 'class_name': '锡膏'}]`.
