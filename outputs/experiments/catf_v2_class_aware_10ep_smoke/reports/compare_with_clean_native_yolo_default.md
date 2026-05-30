# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6097 | 0.7132 | -0.1034 |
| Recall | 0.5103 | 0.7600 | -0.2497 |
| mAP50 | 0.5359 | 0.7759 | -0.2400 |
| mAP50-95 | 0.3522 | 0.5241 | -0.1719 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6097 | 0.7262 | -0.1165 |
| Recall | 0.5103 | 0.6844 | -0.1741 |
| mAP50 | 0.5359 | 0.7616 | -0.2258 |
| mAP50-95 | 0.3522 | 0.5250 | -0.1729 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
