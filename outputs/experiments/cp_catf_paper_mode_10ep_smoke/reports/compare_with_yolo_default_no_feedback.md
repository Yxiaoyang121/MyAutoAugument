# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7728 | 0.7132 | 0.0596 |
| Recall | 0.4190 | 0.7600 | -0.3410 |
| mAP50 | 0.5140 | 0.7759 | -0.2618 |
| mAP50-95 | 0.3242 | 0.5241 | -0.1999 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7728 | 0.7262 | 0.0466 |
| Recall | 0.4190 | 0.6844 | -0.2654 |
| mAP50 | 0.5140 | 0.7616 | -0.2476 |
| mAP50-95 | 0.3242 | 0.5250 | -0.2008 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
