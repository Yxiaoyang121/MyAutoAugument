# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7005 | 0.7132 | -0.0127 |
| Recall | 0.6235 | 0.7600 | -0.1365 |
| mAP50 | 0.7125 | 0.7759 | -0.0634 |
| mAP50-95 | 0.4566 | 0.5241 | -0.0675 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7005 | 0.7262 | -0.0257 |
| Recall | 0.6235 | 0.6844 | -0.0608 |
| mAP50 | 0.7125 | 0.7616 | -0.0491 |
| mAP50-95 | 0.4566 | 0.5250 | -0.0684 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
