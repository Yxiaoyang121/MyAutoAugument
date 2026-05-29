# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7756 | 0.7725 | 0.0031 |
| Recall | 0.7276 | 0.6477 | 0.0799 |
| mAP50 | 0.7763 | 0.7542 | 0.0221 |
| mAP50-95 | 0.5147 | 0.4799 | 0.0347 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7756 | 0.7725 | 0.0031 |
| Recall | 0.7276 | 0.6477 | 0.0799 |
| mAP50 | 0.7763 | 0.7542 | 0.0221 |
| mAP50-95 | 0.5147 | 0.4799 | 0.0347 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `false`
