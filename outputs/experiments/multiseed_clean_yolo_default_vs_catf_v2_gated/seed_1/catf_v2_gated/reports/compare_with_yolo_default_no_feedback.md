# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7852 | 0.7725 | 0.0127 |
| Recall | 0.7005 | 0.6477 | 0.0528 |
| mAP50 | 0.7826 | 0.7542 | 0.0284 |
| mAP50-95 | 0.5189 | 0.4799 | 0.0390 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7852 | 0.7725 | 0.0127 |
| Recall | 0.7005 | 0.6477 | 0.0528 |
| mAP50 | 0.7826 | 0.7542 | 0.0284 |
| mAP50-95 | 0.5189 | 0.4799 | 0.0390 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
