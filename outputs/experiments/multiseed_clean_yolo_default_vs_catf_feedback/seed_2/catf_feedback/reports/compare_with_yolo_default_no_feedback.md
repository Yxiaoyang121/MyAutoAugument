# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6846 | 0.6962 | -0.0116 |
| Recall | 0.7166 | 0.7286 | -0.0120 |
| mAP50 | 0.7608 | 0.7692 | -0.0084 |
| mAP50-95 | 0.4800 | 0.5224 | -0.0424 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6846 | 0.6962 | -0.0116 |
| Recall | 0.7166 | 0.7286 | -0.0120 |
| mAP50 | 0.7608 | 0.7692 | -0.0084 |
| mAP50-95 | 0.4800 | 0.5224 | -0.0424 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
