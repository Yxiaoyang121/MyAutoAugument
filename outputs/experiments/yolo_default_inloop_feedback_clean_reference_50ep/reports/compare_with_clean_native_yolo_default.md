# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7536 | 0.7262 | 0.0274 |
| Recall | 0.7124 | 0.6844 | 0.0281 |
| mAP50 | 0.7723 | 0.7616 | 0.0106 |
| mAP50-95 | 0.5153 | 0.5250 | -0.0098 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7536 | 0.7262 | 0.0274 |
| Recall | 0.7124 | 0.6844 | 0.0281 |
| mAP50 | 0.7723 | 0.7616 | 0.0106 |
| mAP50-95 | 0.5153 | 0.5250 | -0.0098 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
