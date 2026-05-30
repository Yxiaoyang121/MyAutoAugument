# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7376 | 0.7725 | -0.0349 |
| Recall | 0.7098 | 0.6477 | 0.0621 |
| mAP50 | 0.7550 | 0.7542 | 0.0008 |
| mAP50-95 | 0.5185 | 0.4799 | 0.0386 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7376 | 0.7725 | -0.0349 |
| Recall | 0.7098 | 0.6477 | 0.0621 |
| mAP50 | 0.7550 | 0.7542 | 0.0008 |
| mAP50-95 | 0.5185 | 0.4799 | 0.0386 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
