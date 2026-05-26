# Constraint Scoring

- Reference: YOLO default.
- Reject if Precision drops by more than 0.02.
- Reject if mAP50 drops by more than 0.01.
- Reject if mAP50-95 drops by more than 0.01.
- Among passing policies, maximize Recall delta; tie-break by mAP50-95.

| run | Precision | Recall | mAP50 | mAP50-95 | dP | dR | d_mAP50 | d_mAP50-95 | pass constraints |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| YOLO default | 0.7132 | 0.7600 | 0.7759 | 0.5241 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | yes |
| YOLO default + diagnosis_light | 0.7343 | 0.6862 | 0.7555 | 0.4993 | +0.0211 | -0.0738 | -0.0204 | -0.0247 | no |
| YOLO default + diagnosis_precision_safe | 0.6647 | 0.7493 | 0.7557 | 0.5055 | -0.0485 | -0.0107 | -0.0202 | -0.0185 | no |
| YOLO default + diagnosis_recall_safe | 0.6712 | 0.7659 | 0.7411 | 0.5044 | -0.0420 | +0.0059 | -0.0348 | -0.0196 | no |
| custom_yolo_like_base old control | 0.6661 | 0.7490 | 0.7129 | 0.4415 | -0.0471 | -0.0110 | -0.0629 | -0.0826 | no |

## Result

- Best: `YOLO default`
- Recall improved under constraints: `false`
- Failure driver if no improvement: `diagnosis_light: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis; diagnosis_precision_safe: FP increased by 18; diagnosis_precision_safe: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis; diagnosis_recall_safe: mAP50-95 dropped without localization_weak increase in conf=0.25 diagnosis`
