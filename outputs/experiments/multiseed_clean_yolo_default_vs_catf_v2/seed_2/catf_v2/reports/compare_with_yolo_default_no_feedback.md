# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.8358 | 0.6962 | 0.1395 |
| Recall | 0.6039 | 0.7286 | -0.1247 |
| mAP50 | 0.7423 | 0.7692 | -0.0269 |
| mAP50-95 | 0.4938 | 0.5224 | -0.0285 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.8358 | 0.6962 | 0.1395 |
| Recall | 0.6039 | 0.7286 | -0.1247 |
| mAP50 | 0.7423 | 0.7692 | -0.0269 |
| mAP50-95 | 0.4938 | 0.5224 | -0.0285 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
