# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7572 | 0.7262 | 0.0309 |
| Recall | 0.4420 | 0.6844 | -0.2423 |
| mAP50 | 0.6073 | 0.7616 | -0.1544 |
| mAP50-95 | 0.4204 | 0.5250 | -0.1046 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.7572 | 0.7262 | 0.0309 |
| Recall | 0.4420 | 0.6844 | -0.2423 |
| mAP50 | 0.6073 | 0.7616 | -0.1544 |
| mAP50-95 | 0.4204 | 0.5250 | -0.1046 |

## Constraint

- Baseline: `no_feedback_control`
- constraint_failed: `true`
