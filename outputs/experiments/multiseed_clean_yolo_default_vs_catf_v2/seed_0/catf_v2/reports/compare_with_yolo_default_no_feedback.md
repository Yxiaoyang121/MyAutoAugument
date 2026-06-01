# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7429 | 0.7846 | -0.0417 |
| Recall | 0.6982 | 0.6765 | 0.0218 |
| mAP50 | 0.7570 | 0.7347 | 0.0223 |
| mAP50-95 | 0.5023 | 0.4759 | 0.0264 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7429 | 0.7846 | -0.0417 |
| Recall | 0.6982 | 0.6765 | 0.0218 |
| mAP50 | 0.7570 | 0.7347 | 0.0223 |
| mAP50-95 | 0.5023 | 0.4759 | 0.0264 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
