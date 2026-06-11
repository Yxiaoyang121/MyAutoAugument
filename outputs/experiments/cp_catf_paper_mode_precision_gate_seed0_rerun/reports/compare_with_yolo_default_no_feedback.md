# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7513 | 0.7132 | 0.0381 |
| Recall | 0.6763 | 0.7600 | -0.0837 |
| mAP50 | 0.7566 | 0.7759 | -0.0192 |
| mAP50-95 | 0.5114 | 0.5241 | -0.0126 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7513 | 0.7262 | 0.0251 |
| Recall | 0.6763 | 0.6844 | -0.0081 |
| mAP50 | 0.7566 | 0.7616 | -0.0050 |
| mAP50-95 | 0.5114 | 0.5250 | -0.0136 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
