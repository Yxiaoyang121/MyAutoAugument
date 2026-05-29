# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6765 | 0.6962 | -0.0197 |
| Recall | 0.6741 | 0.7286 | -0.0546 |
| mAP50 | 0.7411 | 0.7692 | -0.0281 |
| mAP50-95 | 0.4993 | 0.5224 | -0.0231 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6765 | 0.6962 | -0.0197 |
| Recall | 0.6741 | 0.7286 | -0.0546 |
| mAP50 | 0.7411 | 0.7692 | -0.0281 |
| mAP50-95 | 0.4993 | 0.5224 | -0.0231 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
