# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6356 | 0.6962 | -0.0606 |
| Recall | 0.4952 | 0.7286 | -0.2334 |
| mAP50 | 0.5294 | 0.7692 | -0.2398 |
| mAP50-95 | 0.3436 | 0.5224 | -0.1787 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6356 | 0.6962 | -0.0606 |
| Recall | 0.4952 | 0.7286 | -0.2334 |
| mAP50 | 0.5294 | 0.7692 | -0.2398 |
| mAP50-95 | 0.3436 | 0.5224 | -0.1787 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
