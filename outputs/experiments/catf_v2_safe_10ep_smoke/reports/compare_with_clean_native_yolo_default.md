# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6744 | 0.6962 | -0.0219 |
| Recall | 0.4789 | 0.7286 | -0.2497 |
| mAP50 | 0.5404 | 0.7692 | -0.2288 |
| mAP50-95 | 0.3443 | 0.5224 | -0.1780 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6744 | 0.6962 | -0.0219 |
| Recall | 0.4789 | 0.7286 | -0.2497 |
| mAP50 | 0.5404 | 0.7692 | -0.2288 |
| mAP50-95 | 0.3443 | 0.5224 | -0.1780 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
