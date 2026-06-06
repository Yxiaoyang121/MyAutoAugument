# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7550 | 0.7725 | -0.0175 |
| Recall | 0.7261 | 0.6477 | 0.0784 |
| mAP50 | 0.7653 | 0.7542 | 0.0111 |
| mAP50-95 | 0.4938 | 0.4799 | 0.0139 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7550 | 0.7725 | -0.0175 |
| Recall | 0.7261 | 0.6477 | 0.0784 |
| mAP50 | 0.7653 | 0.7542 | 0.0111 |
| mAP50-95 | 0.4938 | 0.4799 | 0.0139 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
