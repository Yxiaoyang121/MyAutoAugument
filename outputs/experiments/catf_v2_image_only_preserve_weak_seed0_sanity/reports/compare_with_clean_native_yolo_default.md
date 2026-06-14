# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6666 | 0.7132 | -0.0466 |
| Recall | 0.7300 | 0.7600 | -0.0300 |
| mAP50 | 0.6999 | 0.7759 | -0.0759 |
| mAP50-95 | 0.4667 | 0.5241 | -0.0573 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6666 | 0.7262 | -0.0597 |
| Recall | 0.7300 | 0.6844 | 0.0457 |
| mAP50 | 0.6999 | 0.7616 | -0.0617 |
| mAP50-95 | 0.4667 | 0.5250 | -0.0583 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
