# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7637 | 0.6962 | 0.0674 |
| Recall | 0.6863 | 0.7286 | -0.0423 |
| mAP50 | 0.7582 | 0.7692 | -0.0110 |
| mAP50-95 | 0.4967 | 0.5224 | -0.0257 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7637 | 0.6962 | 0.0674 |
| Recall | 0.6863 | 0.7286 | -0.0423 |
| mAP50 | 0.7582 | 0.7692 | -0.0110 |
| mAP50-95 | 0.4967 | 0.5224 | -0.0257 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
