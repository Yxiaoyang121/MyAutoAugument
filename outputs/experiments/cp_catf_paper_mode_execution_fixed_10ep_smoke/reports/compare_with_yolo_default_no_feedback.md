# Compare In-Loop Feedback With YOLO Default / No-Feedback

## Versus YOLO Default Reference

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6942 | 0.7132 | -0.0190 |
| Recall | 0.4739 | 0.7600 | -0.2861 |
| mAP50 | 0.4405 | 0.7759 | -0.3353 |
| mAP50-95 | 0.2865 | 0.5241 | -0.2376 |

## Versus In-Loop No-Feedback Control

| metric | value | reference | delta |
|---|---:|---:|---:|
| Precision | 0.6942 | 0.7262 | -0.0321 |
| Recall | 0.4739 | 0.6844 | -0.2105 |
| mAP50 | 0.4405 | 0.7616 | -0.3211 |
| mAP50-95 | 0.2865 | 0.5250 | -0.2385 |

## Constraint

- Baseline: `clean_native_yolo_default`
- constraint_failed: `true`
