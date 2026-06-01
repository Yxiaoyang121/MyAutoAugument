# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7691 | 0.7725 | -0.0034 |
| Recall | 0.6950 | 0.6477 | 0.0474 |
| mAP50 | 0.7549 | 0.7542 | 0.0007 |
| mAP50-95 | 0.4898 | 0.4799 | 0.0099 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7691 | 0.7725 | -0.0034 |
| Recall | 0.6950 | 0.6477 | 0.0474 |
| mAP50 | 0.7549 | 0.7542 | 0.0007 |
| mAP50-95 | 0.4898 | 0.4799 | 0.0099 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
