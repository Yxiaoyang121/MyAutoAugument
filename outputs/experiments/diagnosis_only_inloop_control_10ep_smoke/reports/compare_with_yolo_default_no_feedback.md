# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6952 | 0.7132 | -0.0180 |
| Recall | 0.5176 | 0.7600 | -0.2425 |
| mAP50 | 0.5749 | 0.7759 | -0.2010 |
| mAP50-95 | 0.3701 | 0.5241 | -0.1539 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6952 | 0.7262 | -0.0311 |
| Recall | 0.5176 | 0.6844 | -0.1668 |
| mAP50 | 0.5749 | 0.7616 | -0.1867 |
| mAP50-95 | 0.3701 | 0.5250 | -0.1549 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
