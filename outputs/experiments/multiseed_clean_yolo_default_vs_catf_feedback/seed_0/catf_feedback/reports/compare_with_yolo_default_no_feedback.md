# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7574 | 0.7846 | -0.0272 |
| Recall | 0.6751 | 0.6765 | -0.0014 |
| mAP50 | 0.7526 | 0.7347 | 0.0179 |
| mAP50-95 | 0.5204 | 0.4759 | 0.0445 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7574 | 0.7846 | -0.0272 |
| Recall | 0.6751 | 0.6765 | -0.0014 |
| mAP50 | 0.7526 | 0.7347 | 0.0179 |
| mAP50-95 | 0.5204 | 0.4759 | 0.0445 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
