# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6394 | 0.7132 | -0.0738 |
| Recall | 0.7231 | 0.7600 | -0.0369 |
| mAP50 | 0.7552 | 0.7759 | -0.0206 |
| mAP50-95 | 0.5073 | 0.5241 | -0.0167 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6394 | 0.6962 | -0.0569 |
| Recall | 0.7231 | 0.7286 | -0.0056 |
| mAP50 | 0.7552 | 0.7692 | -0.0140 |
| mAP50-95 | 0.5073 | 0.5224 | -0.0150 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
