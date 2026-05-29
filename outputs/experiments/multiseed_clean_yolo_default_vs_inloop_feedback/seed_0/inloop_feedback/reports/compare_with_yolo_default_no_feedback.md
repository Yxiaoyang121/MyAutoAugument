# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7188 | 0.7846 | -0.0658 |
| Recall | 0.7273 | 0.6765 | 0.0509 |
| mAP50 | 0.7687 | 0.7347 | 0.0340 |
| mAP50-95 | 0.5276 | 0.4759 | 0.0517 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7188 | 0.7846 | -0.0658 |
| Recall | 0.7273 | 0.6765 | 0.0509 |
| mAP50 | 0.7687 | 0.7347 | 0.0340 |
| mAP50-95 | 0.5276 | 0.4759 | 0.0517 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
