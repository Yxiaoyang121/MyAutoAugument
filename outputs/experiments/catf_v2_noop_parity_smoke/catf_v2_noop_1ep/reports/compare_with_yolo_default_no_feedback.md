# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.8277 | 0.7725 | 0.0552 |
| Recall | 0.1977 | 0.6477 | -0.4499 |
| mAP50 | 0.2127 | 0.7542 | -0.5415 |
| mAP50-95 | 0.1464 | 0.4799 | -0.3335 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.8277 | 0.7725 | 0.0552 |
| Recall | 0.1977 | 0.6477 | -0.4499 |
| mAP50 | 0.2127 | 0.7542 | -0.5415 |
| mAP50-95 | 0.1464 | 0.4799 | -0.3335 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
