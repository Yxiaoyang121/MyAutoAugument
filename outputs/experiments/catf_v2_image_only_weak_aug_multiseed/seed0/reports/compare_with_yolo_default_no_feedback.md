# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6790 | 0.7846 | -0.1055 |
| Recall | 0.7118 | 0.6765 | 0.0354 |
| mAP50 | 0.7222 | 0.7347 | -0.0125 |
| mAP50-95 | 0.4760 | 0.4759 | 0.0001 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6790 | 0.7846 | -0.1055 |
| Recall | 0.7118 | 0.6765 | 0.0354 |
| mAP50 | 0.7222 | 0.7347 | -0.0125 |
| mAP50-95 | 0.4760 | 0.4759 | 0.0001 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
