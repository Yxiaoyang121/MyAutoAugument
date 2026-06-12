# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7062 | 0.7132 | -0.0070 |
| Recall | 0.6512 | 0.7600 | -0.1088 |
| mAP50 | 0.6843 | 0.7759 | -0.0916 |
| mAP50-95 | 0.4427 | 0.5241 | -0.0814 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7062 | 0.7262 | -0.0200 |
| Recall | 0.6512 | 0.6844 | -0.0331 |
| mAP50 | 0.6843 | 0.7616 | -0.0774 |
| mAP50-95 | 0.4427 | 0.5250 | -0.0824 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
