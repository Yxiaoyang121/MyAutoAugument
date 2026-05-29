# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7750 | 0.7262 | 0.0487 |
| Recall | 0.7092 | 0.6844 | 0.0248 |
| mAP50 | 0.7688 | 0.7616 | 0.0072 |
| mAP50-95 | 0.5166 | 0.5250 | -0.0084 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7750 | 0.7262 | 0.0487 |
| Recall | 0.7092 | 0.6844 | 0.0248 |
| mAP50 | 0.7688 | 0.7616 | 0.0072 |
| mAP50-95 | 0.5166 | 0.5250 | -0.0084 |

## Constraint

- Baseline: `clean_native_yolo_default_seed42`
- constraint_failed: `false`
