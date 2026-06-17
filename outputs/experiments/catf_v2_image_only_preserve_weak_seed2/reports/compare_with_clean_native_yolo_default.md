# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7533 | 0.7132 | 0.0401 |
| Recall | 0.6942 | 0.7600 | -0.0658 |
| mAP50 | 0.7727 | 0.7759 | -0.0032 |
| mAP50-95 | 0.5151 | 0.5241 | -0.0089 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7533 | 0.7262 | 0.0270 |
| Recall | 0.6942 | 0.6844 | 0.0099 |
| mAP50 | 0.7727 | 0.7616 | 0.0111 |
| mAP50-95 | 0.5151 | 0.5250 | -0.0099 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
