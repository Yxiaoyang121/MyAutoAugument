# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7850 | 0.6962 | 0.0887 |
| Recall | 0.6795 | 0.7286 | -0.0491 |
| mAP50 | 0.7521 | 0.7692 | -0.0171 |
| mAP50-95 | 0.5083 | 0.5224 | -0.0141 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7850 | 0.6962 | 0.0887 |
| Recall | 0.6795 | 0.7286 | -0.0491 |
| mAP50 | 0.7521 | 0.7692 | -0.0171 |
| mAP50-95 | 0.5083 | 0.5224 | -0.0141 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
